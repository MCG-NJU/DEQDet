import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
from mmdet.models.task_modules.samplers.pseudo_sampler import PseudoSampler
from mmdet.models.task_modules.samplers.mask_pseudo_sampler import MaskPseudoSampler

@TASK_UTILS.register_module()
class UnifiedPseudoSampler(PseudoSampler, MaskPseudoSampler):
    def __init__(self, **kwargs):
        PseudoSampler().__init__(**kwargs)
        MaskPseudoSampler().__init__(**kwargs)

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, *args, **kwargs):
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        pred_masks = getattr(pred_instances, "masks", None)
        gt_masks = gt_instances.masks
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        sampling_result.masks = pred_masks
        sampling_result.gt_masks = gt_masks
        return sampling_result

