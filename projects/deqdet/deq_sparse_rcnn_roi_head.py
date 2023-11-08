# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, OptConfigType, MultiConfig
from mmdet.models.utils.misc import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.sparse_roi_head import SparseRoIHead


@MODELS.register_module()
class DEQ_SparseRoIHead(SparseRoIHead):
    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head: MultiConfig) -> None:
        super().init_bbox_head(bbox_roi_extractor, bbox_head)
        for i in range(1, self.num_stages):
            self.bbox_head[i] = self.bbox_head[1]
    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        super().init_mask_head(mask_roi_extractor, mask_head)
        for i in range(1, self.num_stages):
            self.mask_head[i] = self.mask_head[1]
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:

        losses = {}
        def refine_func(latent_dict):
            results_list = latent_dict["results_list"]
            latent_dict = self._bbox_forward(1, x, bbox2roi([res.bboxes for res in results_list]),
                                         latent_dict['object_feats'],
                                         batch_img_metas)
            # propose for the new proposal_list
            proposal_list = []
            for idx in range(len(batch_img_metas)):
                res = InstanceData()
                res.imgs_whwh = results_list[idx].imgs_whwh
                res.bboxes = latent_dict['detached_proposals'][idx]
                proposal_list.append(res)
            latent_dict['results_list'] = proposal_list
            return latent_dict
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = unpack_gt_instances(batch_data_samples)

        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        latent_dict = dict(
            object_feats=object_feats,
            results_list=rpn_results_list,
        )
        init_latent_dict = None
        for stage in range(3):
            stage_loss_weight = self.stage_loss_weights[stage]
            # bbox head forward and loss
            latent_dict = self.bbox_loss(
                stage=stage,
                x=x,
                object_feats=latent_dict["object_feats"],
                results_list=latent_dict["results_list"],
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances)

            for name, value in latent_dict['loss_bbox'].items():
                losses[f'init^{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            if self.with_mask:
                mask_results = self.mask_loss(
                    stage=stage,
                    x=x,
                    bbox_results=latent_dict,
                    batch_gt_instances=batch_gt_instances,
                    rcnn_train_cfg=self.train_cfg[stage])

                for name, value in mask_results['loss_mask'].items():
                    losses[f'init^{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)
            if stage == 0:
                init_latent_dict = latent_dict # init results

        stage_loss_weight = self.stage_loss_weights[1]
        latent_dict = init_latent_dict
        for stage in range(1, self.num_stages+1):
            with torch.no_grad():
                if (torch.rand(1) < 0.2).item():
                    for res in latent_dict['results_list']:
                        res.bboxes = res.bboxes + torch.randn_like(res.bboxes) * 25
                        l = torch.min(res.bboxes[:, ::2], dim=1)[0]
                        t = torch.min(res.bboxes[:, 1::2], dim=1)[0]
                        r = torch.max(res.bboxes[:, ::2], dim=1)[0]
                        b = torch.max(res.bboxes[:, 1::2], dim=1)[0]
                        res.bboxes = torch.stack([l, t, r, b], dim=1)
                if (torch.rand(1) < 0.2).item():
                    obj_feats = latent_dict["object_feats"]
                    obj_feats = 0.9 * obj_feats + torch.norm(obj_feats, dim=-1, keepdim=True) * torch.randn_like(obj_feats) * 0.1
                    latent_dict["object_feats"] = obj_feats
                latent_dict = refine_func(latent_dict)
            if stage in [1,3,6,9,12, self.num_stages]:
                sup_latent_dict = refine_func(latent_dict)
                # bbox head forward and loss
                sup_latent_dict = self.bbox_loss(
                    stage=1,
                    x=x,
                    object_feats=sup_latent_dict['object_feats'],
                    results_list=sup_latent_dict['results_list'],
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances=batch_gt_instances)

                for name, value in sup_latent_dict['loss_bbox'].items():
                    losses[f'iter{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)
                if self.with_mask:
                    mask_results = self.mask_loss(
                        stage=1,
                        x=x,
                        bbox_results=sup_latent_dict,
                        batch_gt_instances=batch_gt_instances,
                        rcnn_train_cfg=self.train_cfg[1])
                    for name, value in mask_results['loss_mask'].items():
                        losses[f'iter{stage}.{name}'] = (
                            value * stage_loss_weight if 'loss' in name else value)

        return losses
