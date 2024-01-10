# # Copyright (c) Facebook, Inc. and its affiliates.
# from .modeling.meta_arch import custom_rcnn
# from .modeling.roi_heads import detic_roi_heads
# from .modeling.roi_heads import res5_roi_heads
# from .modeling.backbone import swintransformer
# from .modeling.backbone import timm
from .modeling.sam.modeling.Samfpn import SAMAggregatorNeck
from .modeling.sam.detector import ClipOpenDetector
from .modeling.sam.modeling.image_encoder import ImageEncoderViT
from .modeling.sam.modeling.sam_roi_heads import samAnchorPromptRoiHeads
from .modeling.meta_arch.custom_rcnn import CustomRCNN
from .modeling.roi_heads import res5_roi_heads
from .modeling.proposal_generator import CustomStandardRPNHead

# from .data.datasets import lvis_v1
# from .data.datasets import imagenet
# from .data.datasets import cc
# from .data.datasets import objects365
# from .data.datasets import oid
from .data.datasets import coco_zeroshot
from .data.datasets import lvis_v1_zeroshot

try:
    from .modeling.meta_arch import d2_deformable_detr
except:
    pass