from detectron2.modeling.proposal_generator.rpn import RPN, StandardRPNHead, RPN_HEAD_REGISTRY
import torch.nn as nn
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.layers import cat, get_norm
import torch.nn.functional as F
from detectron2.utils.memory import retry_if_cuda_oom
# rpn changing objectness loss, use centerness loss
from detectron2.layers import Conv2d, ShapeSpec, cat

@RPN_HEAD_REGISTRY.register()
class CustomStandardRPNHead(StandardRPNHead):
    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=get_norm('SyncBN', out_channels),
            activation=nn.ReLU(),
        )