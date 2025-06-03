from typing import List
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.det_data_sample import SampleList
from mmdet.utils import InstanceList, OptConfigType
from mmdet.models.dense_heads.embedding_rpn_head import EmbeddingRPNHead

@MODELS.register_module()
class FixPositionEmbeddingRPNHead(EmbeddingRPNHead):
    def __init__(self, frozen=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen = frozen
    def train(self, mode: bool = True):
        super().train()
        if self.frozen:
            self.init_proposal_bboxes.eval()
            self.init_proposal_features.eval()
            for param in self.parameters():
                param.requires_grad = False
        return self
    def _init_layers(self) -> None:
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4, _freeze=True)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)
    def _decode_init_proposals(self, x: List[Tensor],
                               batch_data_samples: SampleList) -> InstanceList:
        rpn_results_list = super()._decode_init_proposals(list(x), batch_data_samples)
        for rpn_results in rpn_results_list:
            rpn_results.features = torch.layer_norm(rpn_results.features, normalized_shape=[self.proposal_feature_channel])
        return rpn_results_list

