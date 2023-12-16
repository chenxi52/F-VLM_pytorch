# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes, Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
import numpy as np
import pickle
__all__ = ["SamRCNNOutputLayers"]
logger = logging.getLogger(__name__)

class ClipRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. change last layer of classifier to clip text encoder
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        text_feats: torch.Tensor,
        **kwargs
    ):
        super().__init__(input_shape, **kwargs)
        self.text_feats = text_feats
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls_score = None

    def forward(self,x):
        if x.dim()>2:
            x = torch.flatten(x, start_dim=1) 
        x_norm = x/x.norm(dim=1,keepdim=True)
        logits_scale = self.logit_scale.exp()
        scores = logits_scale * x_norm @ (self.text_feats.t().to(x.device))
        proposal_deltas = self.bbox_pred(x)
        return  scores, proposal_deltas
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        with open(cfg.MODEL.CLIP_TEXT_FEATS_PATH,'rb') as f:
            ret['text_feats'] = pickle.load(f)
        return ret 