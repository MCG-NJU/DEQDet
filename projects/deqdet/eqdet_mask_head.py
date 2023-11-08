import torch
import torch.nn as nn
from mmengine.model import BaseModule, Sequential, ModuleList
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness, point_sample
from mmdet.structures.mask import mask_target, BitmapMasks
from mmdet.models.layers.transformer.utils import DynamicConv
from torch import Tensor
@MODELS.register_module()
class DenseMaskHead(BaseModule):
    def __init__(self, 
                 embed_dim,
                 loss_mask: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),

                 init_cfg: OptConfigType = None,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_cfg = init_cfg
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def init_weights(self):
        super().init_weights()

    def forward(self, dense_feat, position, content):
        raise NotImplementedError

    def _get_targets_single(self, sampling_result, gt_masks):
        pos_inds = sampling_result.pos_inds
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_targets.new_zeros((sampling_result.masks.shape[0],))
        mask_weights[pos_inds] = 1.0
        return mask_targets, mask_weights

    def get_targets(self, sampling_results, concat=True):
        gt_masks = [res.gt_masks for res in sampling_results]
        mask_targets, mask_weights  = multi_apply(self._get_targets_single, sampling_results, gt_masks)
        # import pdb; pdb.set_trace()
        if concat :
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
        return mask_targets, mask_weights

    def loss_and_target(self, mask_preds, sampling_results, train_cfg, concat=True):
         num_points = train_cfg.get('num_points', 12544)
         oversample_ratio = train_cfg.get('oversample_ratio', 3.0)
         importance_sample_ratio = train_cfg.get(
            'importance_sample_ratio', 0.75)

         mask_targets, mask_weights = self.get_targets(sampling_results, concat)
         mask_preds = mask_preds[mask_weights > 0]

         avg_factor = sum(
             [results.avg_factor for results in sampling_results])
         num_total_masks = max(reduce_mean(mask_preds.new_tensor(avg_factor)),1)
         if mask_targets.shape[0] == 0:
             # zero match
             loss_dice = mask_preds.sum()
             loss_mask = mask_preds.sum()
             return  dict(loss_mask=loss_mask, loss_dice=loss_dice)
         with torch.no_grad():
             points_coords = get_uncertain_point_coords_with_randomness(
                 mask_preds.unsqueeze(1), None, num_points,
                 oversample_ratio, importance_sample_ratio)
             # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
             mask_point_targets = point_sample(
                 mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
         # shape (num_queries, h, w) -> (num_queries, num_points)
         mask_point_preds = point_sample(
             mask_preds.unsqueeze(1), points_coords).squeeze(1)

         # dice loss
         loss_dice = self.loss_dice(
             mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

         # mask loss
         # shape (num_queries, num_points) -> (num_queries * num_points, )
         mask_point_preds = mask_point_preds.reshape(-1)
         # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
         mask_point_targets = mask_point_targets.reshape(-1)
         loss_mask = self.loss_mask(
             mask_point_preds,
             mask_point_targets,
             avg_factor=num_total_masks * num_points)

         return dict(loss_mask=loss_mask, loss_dice=loss_dice)
    
    
@MODELS.register_module()
class DeCoupleHead(DenseMaskHead):
    def __init__(self, embed_dim, loss_mask, loss_dice, init_cfg: OptConfigType = None,):
        super().__init__(embed_dim, loss_mask, loss_dice, init_cfg)
        self.mask_embed = Sequential(
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, dense_feats, mlvl_feats, position, content, filter, bbox, cls, mask):
        filter = content
        mask_embed = self.mask_embed(content)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, dense_feats[0])
        return content, filter, mask_pred

@MODELS.register_module()
class DeCoupleMLVLEHead(DenseMaskHead):
    def __init__(self, embed_dim, loss_mask, loss_dice, init_cfg: OptConfigType = None,mlvl=4, share_filter=False):
        super().__init__(embed_dim, loss_mask, loss_dice, init_cfg)
        if share_filter:
            self.mask_embed = nn.ModuleList(
                [self.mask_embed[0] for _ in range(mlvl)]
            )
        else:
            self.mask_embed = nn.ModuleList(
                [nn.Linear(embed_dim, embed_dim) for _ in range(mlvl)]
            )

        self.mlvl_weight_generator = nn.Sequential(
            nn.Linear(embed_dim, mlvl),
        )

    def forward(self, dense_feats, mlvl_feats, position, content, filter, bbox, cls, mask):
        filter = content
        lvl_weight = (self.mlvl_weight_generator(content)).softmax(dim=-1)
        mlvl_preds = []
        for lvl, lvl_feat in enumerate(dense_feats):
            mask_embed = self.mask_embed[lvl](content)
            lvl_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, lvl_feat)
            lvl_pred = torch.nn.functional.interpolate(lvl_pred, size=dense_feats[0].shape[-2:], mode='bilinear', align_corners=False)
            mlvl_preds.append(lvl_pred)
        mlvl_preds = torch.stack(mlvl_preds, dim=-1)
        mask_pred = torch.einsum('bqhwl,bql->bqhw', mlvl_preds, lvl_weight)
        return content, filter, mask_pred


from mmdet.models.roi_heads.mask_heads.dynamic_mask_head import DynamicMaskHead
@MODELS.register_module()
class WarpedDynamicMaskHead(DynamicMaskHead):
    def __init__(self,
                 *args,
                 mask_roi_extractor= dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32]),
                 loss_mask= dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        reduction='mean',
                        loss_weight=5.0),
                 loss_dice=dict(
                        type='DiceLoss',
                        loss_weight=8.0,
                        use_sigmoid=True,
                        activate=False,
                        eps=1e-5),
                 **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.roi_extractor = MODELS.build(mask_roi_extractor)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def forward(self, dense_feats, mlvl_feats, position, content, filter, bbox, cls, mask):
        batch_ids = torch.arange(bbox.shape[0], device=bbox.device).view(-1, 1, 1).repeat(1, bbox.shape[1], 1)
        rois = torch.cat([batch_ids, bbox], dim=-1).view(-1, 5)
        roi_feats = self.roi_extractor(dense_feats, rois)

        mask_pred = super().forward(roi_feats, content)
        mask_pred =  mask_pred.view(bbox.shape[0], bbox.shape[1], *mask_pred.shape[1:])
        return content, filter, mask_pred

    def loss_and_target(self, mask_preds, sampling_results, train_cfg, concat=True):
        pos_proposals = [res.pos_priors.detach() for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.gt_masks for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        num_pos = pos_labels.new_ones(pos_labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        if mask_preds.size(0) == 0:
            # loss_mask = mask_preds.sum()
            loss_dice = mask_preds.sum()
        else:
            pos_inds =  []
            for i in range(len(sampling_results)):
                pos_inds.append(sampling_results[i].pos_inds + (i * mask_preds.shape[0] // len(sampling_results)))
            pos_inds = torch.cat(pos_inds)
            mask_scores = mask_preds[pos_inds, pos_labels].sigmoid()
            loss_dice = self.loss_dice(
                mask_scores,
                mask_targets,
                avg_factor=avg_factor
            )
        return dict(loss_dice=loss_dice)

@MODELS.register_module()
class WarpedEfficientDynamicMaskHead(DynamicMaskHead):
    def __init__(self,
                 *args,
                 mask_roi_extractor=None,
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs,
    ):
        if mask_roi_extractor is None:
            mask_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
        if loss_mask is None:
            loss_mask = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0)
        if loss_dice is None:
            loss_dice = dict(
                type='DiceLoss',
                loss_weight=8.0,
                use_sigmoid=True,
                activate=False,
                eps=1e-5)
        super().__init__(*args, **kwargs)
        self.roi_extractor = MODELS.build(mask_roi_extractor)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
    def forward(self, dense_feats, mlvl_feats, position, content, filter, bbox, cls, mask):
        batch_ids = torch.arange(bbox.shape[0], device=bbox.device).view(-1, 1, 1).repeat(1, bbox.shape[1], 1)
        rois = torch.cat([batch_ids, bbox], dim=-1).view(-1, 5)
        roi_feats = self.roi_extractor(dense_feats, rois)
        mask_pred = super().forward(roi_feats, content[...,:256])
        mask_pred =  mask_pred.view(bbox.shape[0], bbox.shape[1], *mask_pred.shape[1:])
        return content, filter, mask_pred

    def loss_and_target_positive(self,  intermediate, sampling_results, train_cfg, concat=True):
        dense_feats, mlvl_feats, positions, contents, _, _, _, bboxes, _ = intermediate
        pos_proposals = [res.pos_priors.detach() for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.gt_masks for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        num_pos = pos_labels.new_ones(pos_labels.size()).float().sum()

        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        if num_pos < 1:
            return dict(loss_dice=contents.sum()*0)
        def make_positive_samples(sampling_results, contents, rois):
            pos_inds, batch_inds = [], []
            for bid, res in enumerate(sampling_results):
                _pos_inds = res.pos_inds
                _batch_inds = torch.ones(len(res.pos_inds), dtype=torch.float, device=rois.device) * bid
                pos_inds.append(_pos_inds)
                batch_inds.append(_batch_inds)
            pos_inds = torch.cat(pos_inds, dim=0).to(int)
            batch_inds = torch.cat(batch_inds, dim=0).to(int)
            positive_rois_wbid = torch.cat([batch_inds[:, None], rois[batch_inds, pos_inds]], dim=-1)
            postive_content = contents[batch_inds, pos_inds]

            return positive_rois_wbid, postive_content
        postive_rois, positive_contents = make_positive_samples(sampling_results, contents, bboxes)
        roi_feats = self.roi_extractor(dense_feats, postive_rois)
        positive_mask_pred = super().forward(roi_feats, positive_contents[...,:256])

        ins_inds = torch.arange(0, len(pos_labels))
        mask_scores = positive_mask_pred[ins_inds,pos_labels].sigmoid()
        loss_dice = self.loss_dice(
            mask_scores,
            mask_targets,
            avg_factor=avg_factor
        )
        return dict(loss_dice=loss_dice)

        
