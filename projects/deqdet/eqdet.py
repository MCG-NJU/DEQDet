# Copyright (c) OpenMMLab. All rights reserved.
import torch.amp
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector


@MODELS.register_module()
class DEQDet(TwoStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        torch.set_float32_matmul_precision('high')
        # TODO: compile not work
        # self.backbone = torch.compile(self.backbone)
    def predict(self,
                batch_inputs,
                batch_data_samples,
                rescale: bool = True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True, cache_enabled=True):
            x = list(map(lambda x: x.type(torch.float32), self.extract_feat(batch_inputs)))
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)
        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            data_sample.pred_instances = pred_instances["ins_results"]
        return batch_data_samples
