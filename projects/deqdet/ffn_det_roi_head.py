# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from mmcv.ops import point_sample
from mmengine.model import ModuleList
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.structures.mask import mask2bbox
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from mmdet.models.utils import preprocess_panoptic_gt, multi_apply
from .adamixer_head import encode_xyzr, decode_box
from mmdet.models.task_modules.assigners import AssignResult
import functools


def register_stash_grad_hook(v_list):
    v_leaf_list = [v.detach().requires_grad_(True) for v in v_list]
    def recover_stashed_grad_from_leaf(v_grad, v_leaf):
        if v_leaf.grad is not None:
            return v_grad + v_leaf.grad
        return v_grad
    for v, v_leaf in zip(v_list, v_leaf_list):
        v.register_hook(
            functools.partial(recover_stashed_grad_from_leaf, v_leaf=v_leaf)
        )
    return v_leaf_list

@MODELS.register_module()
class FFNDetRoIHead(CascadeRoIHead):
    def __init__(self,
                 featmap_strides=(4, 8, 16, 32),
                 dense_stride=4,
                 encode_box=encode_xyzr,
                 decode_box=decode_box,
                 num_stages=6,
                 frozen_stages=0,
                 bbox_roi_extractor: ConfigType = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        self.featmap_strides = featmap_strides
        self.mask_enable = mask_head is not None
        self.num_stages = num_stages
        super().__init__(
            num_stages=self.num_stages,
            stage_loss_weights=(1.0,)*self.num_stages,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.dense_stride = dense_stride
        self.encode_box = encode_box
        self.decode_box = decode_box
        self.frozen_stages = frozen_stages

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_stages > 0:
            for i in range(self.frozen_stages):
                self.bbox_head[i] = self.bbox_head[i].eval()
                for param in self.bbox_head[i].parameters():
                    param.requires_grad = False
        return self

    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head: MultiConfig) -> None:
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()

        # init
        for i in range(self.num_stages):
            self.bbox_roi_extractor.append(MODELS.build(bbox_roi_extractor))
            self.bbox_head.append(MODELS.build(bbox_head[i]))

    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        self.mask_roi_extractor = ModuleList()
        self.mask_head = ModuleList()


    def layer_forward(self, layer_idx, mask_feats, mlvl_feats, position, content, attn_mask, filter, cls, bbox,
                      mask=None, need_mask=False):
        position = position.detach()
        updated_position, updated_content, updated_cls, updated_bbox = \
            self.bbox_head[layer_idx](mlvl_feats, position, content, attn_mask, cls, bbox, self.featmap_strides)
        if self.mask_enable and need_mask:
            updated_content, updated_filter, updated_mask = \
                self.mask_head[layer_idx](mask_feats, mlvl_feats, updated_position, updated_content, filter,
                                          updated_bbox, updated_cls, mask)
        else:
            updated_mask = None
            updated_filter = None
        return updated_position, updated_content, attn_mask, updated_filter, updated_cls, updated_bbox, updated_mask

    def layer_loss(self, prefix, layer_idx, assigner, sampler, feats ,results, batch_metas):
        outs = dict()
        position, content, _, filter, cls_scores, bbox_results, mask_results = results
        bboxes_list = [bboxes for bboxes in bbox_results]
        cls_list = [cls for cls in cls_scores]
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas, imgs_whwh \
            = batch_metas[:4]

        sampling_results = []
        for i, gt_instances in enumerate(batch_gt_instances):
            pred_instances = InstanceData(scores=cls_list[i], bboxes=bboxes_list[i], priors=bboxes_list[i])
            assign_result = assigner.assign(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                gt_instances_ignore=None,
                img_meta=batch_img_metas[i],
            )
            sampling_result = sampler.sample(
                assign_result, pred_instances, gt_instances)
            sampling_results.append(sampling_result)
        flatten_cls_scores = torch.cat(cls_list, dim=0)
        flatten_decoded_bboxes = torch.cat(bboxes_list, dim=0)
        loss_with_target = self.bbox_head[layer_idx].loss_and_target(flatten_cls_scores, flatten_decoded_bboxes,
                                                                     sampling_results, rcnn_train_cfg=self.train_cfg[0],
                                                                     imgs_whwh=imgs_whwh, concat=True)
        for k, v in loss_with_target["loss_bbox"].items():
            outs[prefix + k] = v

        if self.mask_enable:
            if hasattr(self.mask_head[layer_idx], "loss_and_target_positive"):
                intermediate = list(feats) + list(results) # TODO
                loss_with_target = self.mask_head[layer_idx].loss_and_target_positive(intermediate, sampling_results,
                                                                         self.train_cfg[0], concat=True)
            else:
                flatten_mask_preds = mask_results.view(-1, mask_results.shape[-2], mask_results.shape[-1])
                loss_with_target = self.mask_head[layer_idx].loss_and_target(flatten_mask_preds, sampling_results,
                                                                         self.train_cfg[0], concat=True)
            for k, v in loss_with_target.items():
                outs[prefix + k] = v

        return outs

    def preprocess_gt(self, batch_data_samples):
        # import pdb; pdb.set_trace()
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)
        return batch_gt_instances, batch_gt_semantic_segs, batch_img_metas


    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = self.preprocess_gt(batch_data_samples)
        imgs_whwh = torch.cat([res.imgs_whwh[None, ...] for res in rpn_results_list], dim=0)
        batch_metas = (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas, imgs_whwh)
        if self.mask_enable and isinstance(x, dict):
            mlvl_feats = x["mlvl_feats"]
            mask_feats = x["mask_feats"]
        else:
            mlvl_feats = x
            mask_feats = x

        init_content = torch.cat([res.features[None, ...] for res in rpn_results_list], dim=0)
        init_bbox = torch.cat([res.bboxes[None, ...] for res in rpn_results_list], dim=0)
        init_position = self.encode_box(init_bbox)
        loss_dict = {}

        feats = (mask_feats, mlvl_feats)
        # init and extra supervision
        results = (init_position, init_content, None, None, None, init_bbox, None)
        for i in range(self.num_stages):
            layer, assigner, sampler = self.bbox_head[i], self.bbox_assigner[i], self.bbox_sampler[i]
            results = self.layer_forward(i, mask_feats, mlvl_feats, *results, need_mask=False)
            loss_dict.update(self.layer_loss(f'stage_{i}', i, assigner, sampler, feats, results, batch_metas))
        return loss_dict

    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        raise NotImplementedError

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        if self.mask_enable and isinstance(x, dict):
            mlvl_feats = x["mlvl_feats"]
            mask_feats = x["mask_feats"]
        else:
            mlvl_feats = x
            mask_feats = x
        init_content = torch.cat([res.features[None, ...] for res in rpn_results_list], dim=0)
        init_bbox = torch.cat([res.bboxes[None, ...] for res in rpn_results_list], dim=0)
        init_position = self.encode_box(init_bbox)
        results = (init_position, init_content, None, None, None, init_bbox, None)
        for i in range(self.num_stages):
            results = self.layer_forward(i, mask_feats, mlvl_feats, *results, need_mask=False)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        _, _, _, _, cls_score, bbox_pred, mask_pred = results

        bboxes_list = [bboxes for bboxes in bbox_pred]
        num_classes = self.bbox_head[-1].num_classes
        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]
        topk_inds_list = []
        results_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_inds = cls_score_per_img.flatten(0, 1).topk(
                self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_inds % num_classes
            bboxes_per_img = bboxes_list[img_id][topk_inds // num_classes]
            topk_inds_list.append(topk_inds)

            results = InstanceData()
            if self.mask_enable:
                mask_pred_roi_per_img = mask_pred[img_id][topk_inds // num_classes]
                mask_pred_per_img = self.mask_head[-1]._predict_by_feat_single(
                    mask_preds=mask_pred_roi_per_img,
                    bboxes=bboxes_per_img,
                    labels=labels_per_img,
                    img_meta=batch_img_metas[img_id],
                    rcnn_test_cfg=self.test_cfg,
                    rescale=rescale, # rescale here for mask enable situation
                    activate_map=True)
                results.masks = mask_pred_per_img
            else :
                if rescale and bboxes_per_img.size(0) > 0 :
                    assert batch_img_metas[img_id].get('scale_factor') is not None
                    scale_factor = bboxes_per_img.new_tensor(
                        batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                    bboxes_per_img = (
                            bboxes_per_img.view(bboxes_per_img.size(0), -1, 4) /
                            scale_factor).view(bboxes_per_img.size()[0], -1)
            results.bboxes = bboxes_per_img
            results.scores = scores_per_img
            results.labels = labels_per_img
            results_list.append(dict(ins_results=results))

        return results_list
