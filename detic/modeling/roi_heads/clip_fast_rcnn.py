# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import torch
import torch.nn as nn
from typing import  List, Tuple
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.layers import cat, ciou_loss, diou_loss
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Instances, Boxes
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, _log_classification_stats
from detic.modeling.utils import load_class_freq
import numpy as np
import pickle
from detectron2.modeling.poolers import ROIPooler
from torch.cuda.amp import autocast
from detic.data.datasets.coco_zeroshot import get_contigous_ids
__all__ = ["SamRCNNOutputLayers"]
logger = logging.getLogger(__name__)

class ClipRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. change last layer of classifier to clip text encoder
    2. set classifier weights of novel class to 0
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        text_feats: torch.Tensor,
        ignore_zero_cats: bool,
        cat_freq_path: str,
        use_fed_loss: bool,
        fed_loss_num_cat: int,
        fed_loss_freq_weight: float,
        base_alpha: float,
        novel_beta: float,
        test_pooler: ROIPooler,
        background_weight: float,
        **kwargs
    ):
        super().__init__(input_shape, **kwargs)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls_score = None
        if ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        self.ignore_zero_cats = ignore_zero_cats
        self.use_fed_loss = use_fed_loss
        self.fed_loss_num_cat = fed_loss_num_cat
        self.base_alpha = base_alpha
        self.novel_beta = novel_beta
        self.test_pooler = test_pooler
        self.background_weight = background_weight
        self.register_buffer('text_feats', text_feats) 
        # base_ones: 0 or 1 [0,79]
        base_ones = (freq_weight.view(-1) > 1e-4).float()
        self.register_buffer('base_ones', base_ones)
        unused_index = get_contigous_ids('unused') # [0-79]
        self.register_buffer('unused_index', torch.tensor(unused_index))


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        with open(cfg.MODEL.CLIP_TEXT_FEATS_PATH,'rb') as f:
            text_feats = pickle.load(f)
        # seen_unseen + background
        ret['text_feats'] = text_feats
        ret['ignore_zero_cats'] = cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS
        ret['cat_freq_path'] = cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        ret['use_fed_loss'] = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        ret['fed_loss_num_cat'] = cfg.MODEL.NUM_SAMPLE_CATS
        ret['fed_loss_freq_weight'] = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT
        ret['base_alpha'] = cfg.MODEL.ROI_BOX_HEAD.BASE_ALPHA
        ret['novel_beta'] = cfg.MODEL.ROI_BOX_HEAD.NOVEL_BETA
        ret['background_weight'] = cfg.MODEL.ROI_BOX_HEAD.BACKGROUND_WEIGHT
        test_pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            scales=[1./32,],
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        )
        ret['test_pooler'] = test_pooler
        return ret 
    
    def forward(self,x):
        if x.dim()>2:
            x = torch.flatten(x, start_dim=1) 
        x_norm = x/x.norm(dim=1,keepdim=True)
        logits_scale = self.logit_scale.exp()
        with autocast():
            scores = logits_scale * x_norm @ (self.text_feats.t().to(x.device))
            if not self.training:
                scores[:, self.unused_index] = float('-inf')
        proposal_deltas = self.bbox_pred(x)
        return  scores, proposal_deltas
    
    def losses(self, predictions, proposals):
        """
        change cross_entropy weight of novel class to 0
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
        
        if self.use_sigmoid_ce:
            if self.ignore_zero_cats:
                assert NotImplementedError
            else:
                loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            if self.ignore_zero_cats:
                w = self.base_ones
                w = torch.cat([w, w.new_ones(1)*self.background_weight])
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean", weight=w)
            else:
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
            
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        和之前的 reg_loss不同的是，只计算和对应的 proposal的 delta 作为目标回归对象
        而不是原来的和所有的 propsal_boxes 的 loss 
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            # actually it's mean loss finally in the output of return
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction='sum'
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_box_reg / max(gt_classes.numel(), 1.0)


    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], 
                  proposals: List[Instances], clip_feats: torch.Tensor,
                  attenpool: nn.AdaptiveAvgPool2d):
        """
        align vlm_box_features with text_feats
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        vlm_box_features = self.test_pooler([clip_feats], [Boxes(box) for box in boxes])
        # vlm pooler layer: clip attenpool
        vlm_box_features = attenpool(vlm_box_features)
        vlm_box_features = vlm_box_features / vlm_box_features.norm(dim=1,keepdim=True)
        logits_scale = 1/0.01
        vlm_scores = logits_scale * vlm_box_features @ (self.text_feats.t().to(vlm_box_features.device))
        num_inst_per_image = [len(p) for p in proposals]
        vlm_scores[:, self.unused_index] = float('-inf')
        vlm_scores = torch.nn.functional.softmax(vlm_scores, dim=1)
        vlm_scores = vlm_scores.split(num_inst_per_image, dim=0)
        # scores are differnent for base and novel class, and background score comes from the detector

        return self.ov_fast_rcnn_inference(
            boxes,
            scores,
            vlm_scores,
            image_shapes,
        )
    
    def ov_fast_rcnn_inference(
            self,
            boxes: List[torch.Tensor],
            scores: List[torch.Tensor],
            vlm_scores: List[torch.Tensor],
            image_shapes: List[Tuple[int, int]],
        ):
        """
        add vlm_scores to fast_rcnn_inference
        """
        result_per_image = [
            self.ov_fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, vlm_scores_per_image, image_shape
            )
            for scores_per_image, vlm_scores_per_image, boxes_per_image, image_shape in zip(scores, vlm_scores, boxes, image_shapes)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def ov_fast_rcnn_inference_single_image(
            self,
            boxes,
            scores,
            vlm_scores,
            image_shape: Tuple[int, int],
        ):
        """
        add vlm_scores to fast_rcnn_inference_single_image
        final_score_base = vlm_score^0.65 * vlm_scores^0.35    
        final_score_novel = vlm_score^0.35 * vlm_scores^0.65    
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            vlm_scores = vlm_scores[valid_mask]
        scores = scores[:, :-1]
        vlm_scores = vlm_scores[:, :-1]
        # 0 会自动过滤掉
        base_scores = ((scores * self.base_ones)**(1-self.base_alpha)) * ((vlm_scores*self.base_ones)**(self.base_alpha))
        novel_scores = ((scores * (1-self.base_ones))**(1-self.novel_beta)) * ((vlm_scores*(1-self.base_ones))**(self.novel_beta))
        scores = base_scores + novel_scores

        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = scores > self.test_score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, scores, filter_inds[:, 1], self.test_nms_thresh)
        if self.test_topk_per_image >= 0:
            keep = keep[:self.test_topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]
