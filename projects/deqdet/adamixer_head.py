from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import (build_activation_layer,
                      build_norm_layer)
from mmengine.model.weight_init import bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.dii_head import DIIHead
from .sampling_3d_operator import sampling_3d
from .adaptive_mixing_operator import AdaptiveMixing
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps



# @torch.compile()
def decode_box(xyzr):
    # return xyzr
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

# @torch.compile()
def encode_xyzr(roi):
    xy = (roi[...,0:2] + roi[...,2:4]) * 0.5
    wh = torch.abs(roi[...,-2:] - roi[...,:2])
    logwh = torch.log2(wh+1e-7)
    scale = (logwh[...,0:1]+logwh[...,1:2])*0.5
    ratio = (logwh[...,1:2]-logwh[...,0:1])
    xyzr = torch.cat([xy, scale, ratio], dim=-1)
    return xyzr
# @torch.compile()
def make_sample_points(offset, num_group, xyzr):
    '''
        offset_yx: [B, L, num_group*3], normalized by stride

        return: [B, H, W, num_group, 3]
        '''
    B, L, _ = offset.shape

    offset = offset.view(B, L, 1, num_group, 3)

    roi_cc = xyzr[..., :2]
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                               xyzr[..., 3:4] * 0.5], dim=-1)
    roi_wh = scale * ratio

    roi_lvl = xyzr[..., 2:3].view(B, L, 1, 1, 1)

    offset_yx = (offset[..., :2]) * roi_wh.view(B, L, 1, 1, 2) #TODO
    # offset_yx = (2*offset[..., :2].sigmoid()-1) * roi_wh.view(B, L, 1, 1, 2) #TODO
    sample_yx = roi_cc.contiguous().view(B, L, 1, 1, 2) \
        + offset_yx

    sample_lvl = roi_lvl + offset[..., 2:3]

    return torch.cat([sample_yx, sample_lvl], dim=-1)

class AdaptiveSamplingMixing(nn.Module):
    def __init__(self,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 content_dim=256,
                 feat_channels=None
                 ):
        super(AdaptiveSamplingMixing, self).__init__()
        self.in_points = in_points
        self.out_points = out_points
        self.n_groups = n_groups
        self.content_dim = content_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.content_dim

        self.sampling_offset_generator = nn.Sequential(
            nn.Linear(content_dim, in_points * n_groups * 3)
        )
        self.norm = nn.LayerNorm(content_dim)
        self.adaptive_mixing = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.content_dim,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups,
        )

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator[-1].weight)
        nn.init.zeros_(self.sampling_offset_generator[-1].bias)

        bias = self.sampling_offset_generator[-1].bias.data.view(
            self.n_groups, self.in_points, 3)

        # if in_points are squared number, then initialize
        # to sampling on grids regularly, not used in most
        # of our experiments.
        if int(self.in_points ** 0.5) ** 2 == self.in_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)

        # initialize sampling delta z
        nn.init.constant_(bias[:, :, 2:3], -1.0)

        self.adaptive_mixing.init_weights()
    #@torch.compile()
    def forward(self, x, query_feat, query_xyzr, featmap_strides):
        offset = self.sampling_offset_generator(query_feat)

        sample_points_xyz = make_sample_points(
            offset, self.n_groups * self.in_points,
            query_xyzr,
        )
        sampled_feature, _ = sampling_3d(sample_points_xyz, x,
                                         featmap_strides=featmap_strides,
                                         n_points=self.in_points,
                                         )
        query_feat = self.adaptive_mixing(sampled_feature, query_feat)
        query_feat = self.norm(query_feat)

        return query_feat

# @torch.compile()
def position_embedding(token_xyzr, num_feats, temperature=10000):
    assert token_xyzr.size(-1) == 4
    term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=4).flatten(2)
    return pos_x


@MODELS.register_module()
class AdaMixerHead(DIIHead):
    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 use_operator=False,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(AdaMixerHead, self).__init__(
            num_classes=num_classes,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_iou = MODELS.build(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 4)

        self.in_points = in_points
        self.n_groups = n_groups
        self.out_points = out_points

        self.sampling_n_mixing = AdaptiveSamplingMixing(
            content_dim=content_dim,  # query dim
            feat_channels=feat_channels,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups
        )

        self.iof_tau = nn.Parameter(torch.ones(self.attention.num_heads, ))
        self.use_operator_predictor = use_operator
    @torch.no_grad()
    def init_weights(self):
        super(AdaMixerHead, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                nn.init.xavier_uniform_(m.weight)

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        nn.init.uniform_(self.iof_tau, 0.0, 4.0)

        self.sampling_n_mixing.init_weights()

    def forward(self,
                x,
                query_position,
                query_content,
                attn_mask,
                cls_scores,
                bbox_preds,
                featmap_strides):
        query_xyzr = query_position
        N, n_query = query_content.shape[:2]
        with torch.no_grad():
            rois = bbox_preds  # decode_box(query_xyzr)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                  :, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(query_xyzr, query_content.size(-1) // 4)
        '''IoF'''
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1))
        attn_mask = attn_mask + attn_bias if attn_mask is not None else attn_bias
        query_content = query_content.permute(1, 0, 2)
        pe = pe.permute(1, 0, 2)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe
        query_content = self.attention(
            query_content_attn,
            attn_mask=attn_mask.flatten(0, 1),
        )
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)
        attn_content = query_content
        ''' adaptive 3D sampling and mixing '''
        query_content = self.sampling_n_mixing(
            x, query_content, query_xyzr, featmap_strides)

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content))
        cls_feat = query_content
        reg_feat = query_content
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_scores = self.fc_cls(cls_feat).view(N, n_query, -1)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)
        query_position, bbox_preds = self.refine_xyzr(query_xyzr, xyzr_delta)
        return query_position, query_content.view(N, n_query, -1), cls_scores, bbox_preds

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr