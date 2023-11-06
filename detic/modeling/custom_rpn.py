from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN_HEAD_REGISTRY
import torch.nn as nn
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple, Union

# rpn changing
# the outputing proposals
# regression scheme
@RPN_HEAD_REGISTRY.register()
class SAMRpnHead(StandardRPNHead):
    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)):
        super().init(in_channels=in_channels, num_anchors=num_anchors, box_dim=box_dim, conv_dims=conv_dims)

# def forward()