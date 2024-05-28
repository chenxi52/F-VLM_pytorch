from .modeling.detector import ClipOpenDetector
from .modeling.meta_arch.custom_rcnn import CustomRCNN
from .modeling.roi_heads import res5_roi_heads
from .modeling.proposal_generator import CustomStandardRPNHead
from .modeling.backbone import *

from .data.datasets import coco_zeroshot
from .data.datasets import lvis_v1_zeroshot

try:
    from .modeling.meta_arch import d2_deformable_detr
except:
    pass