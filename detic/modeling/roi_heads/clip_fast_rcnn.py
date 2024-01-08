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
import fvcore.nn.weight_init as weight_init
from detic.data.datasets.coco_zeroshot import get_contigous_ids, _get_metadata
import torch.nn.functional as F
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
        text_feats_path: str,
        ignore_zero_cats: bool,
        use_fed_loss: bool,
        fed_loss_num_cat: int,
        base_alpha: float,
        novel_beta: float,
        test_pooler: ROIPooler,
        background_weight: float,
        use_focal_ce: bool,
        **kwargs
    ):
        super().__init__(input_shape, **kwargs)
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07))
        del self.cls_score
        if ignore_zero_cats:
            # 输出基于总类别数量的 continuous id
            base_ones = torch.zeros(len(get_contigous_ids('all')))
            base_ones[get_contigous_ids('seen')] = 1
            base_ones = torch.cat([base_ones, torch.ones(1)]).to(torch.bool)
            self.register_buffer('base_ones', base_ones)
            unused_index = get_contigous_ids('unused') # [0-79]
            self.register_buffer('unused_index', torch.tensor(unused_index))
            novel_ones = torch.zeros(len(get_contigous_ids('all')))
            novel_ones[get_contigous_ids('unseen')] = 1
            novel_ones = torch.cat([novel_ones, torch.ones(1)]).to(torch.bool)
            self.register_buffer('novel_ones', novel_ones)
            
            del self.bbox_pred
            input_size = input_shape.channels * \
                (input_shape.width or 1) * (input_shape.height or 1)
            self.bbox_pred = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, 4, dtype=torch.float32)
            )
            weight_init.c2_xavier_fill(self.bbox_pred[0])
            nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
            nn.init.constant_(self.bbox_pred[-1].bias, 0)

        self.ignore_zero_cats = ignore_zero_cats
        self.use_fed_loss = use_fed_loss
        self.fed_loss_num_cat = fed_loss_num_cat
        self.base_alpha = base_alpha
        self.novel_beta = novel_beta
        self.test_pooler = test_pooler
        self.background_weight = background_weight
        text_feats = np.load(text_feats_path, allow_pickle=True)
        text_feats = torch.from_numpy(text_feats).to(torch.float32)
        self.register_buffer('text_feats', text_feats)
        self.use_focal_ce = use_focal_ce

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['text_feats_path'] = cfg.MODEL.CLIP_TEXT_FEATS_PATH
        ret['ignore_zero_cats'] = cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS
        ret['use_fed_loss'] = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        ret['fed_loss_num_cat'] = cfg.MODEL.NUM_SAMPLE_CATS
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
        ret['use_focal_ce'] = cfg.MODEL.ROI_BOX_HEAD.USE_FOCAL_CE
        return ret 
    
    def forward(self,x):
        scores = self.logit_scale.exp() * self.get_logits(x, self.text_feats)
        proposal_deltas = self.bbox_pred(x)
        return  scores, proposal_deltas
    
    def get_logits(self, img_feats, text_feats):
        img_feats = img_feats/img_feats.norm(dim=1, keepdim=True)
        text_feats = text_feats/text_feats.norm(dim=1, keepdim=True)
        logits = img_feats @ (text_feats.t().to(img_feats.device)).to(torch.float32)
        return logits

    def losses(self, predictions, proposals):
        """
        change cross_entropy weight of novel class to 0
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # sigmoid ce
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
        weight = 1.
        if self.use_sigmoid_ce and not self.use_focal_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes, self.background_weight>0)
        elif self.use_focal_ce and not self.use_sigmoid_ce:
            loss_cls = self.softmax_focal_loss(scores, gt_classes, gamma=0.5, reduction="mean")
        elif self.use_focal_ce and self.use_sigmoid_ce:
            loss_cls = self.sigmoid_focal_loss(scores, gt_classes)
        else:
            if self.ignore_zero_cats:
                weight = torch.ones(91).to(scores.device)
                weight[self.num_classes] *= self.background_weight
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean", weight=weight)
            
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes, consider_background=False):
        """
        Args:
            consifer background class
            self.background_weight>0 and use_sigmoid_ce=True means consider background class 
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        N = pred_class_logits.shape[0]
        K = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(N, K + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        
        if not consider_background:
            target = target[:, :K]
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_class_logits[:, :-1], target, reduction="none")
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_class_logits, target, reduction="none")

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1
        loss = torch.sum(cls_loss * weight) / N
        return loss
    
    def softmax_focal_loss(self, inputs, targets, gamma=0.5, reduction="mean"):
        """Inspired by RetinaNet implementation"""
        # use softmax score as  p of focal loss
        if targets.numel() == 0 and reduction == "mean":
            return input.sum() * 0.0  # connect the gradient
        
        # focal scaling
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = F.softmax(inputs, dim=-1)
        p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
        loss = ce_loss * ((1 - p_t) ** gamma)

        # bg loss weight
        if self.background_weight>0:
            loss_weight = torch.ones(loss.size(0)).to(p.device)
            loss_weight[targets == self.num_classes] = self.background_weight
            loss = loss * loss_weight

        if reduction == "mean":
            loss = loss.mean()

        return loss
    
    def sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """Compute the sigmoid focal loss."""
        prob = inputs.sigmoid()
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes + 1).float()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction="none")
        p_t = prob * targets_one_hot + (1 - prob) * (1 - targets_one_hot)
        loss = ce_loss * ((1 - p_t) ** gamma)
        B = inputs.shape[0]
        weight = 1
        if alpha >= 0:
            loss = (alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)) * loss

        # if self.ignore_zero_cats and (self.freq_weight is not None):
        #     w = (self.freq_weight.view(-1) > 1e-4).float()
        #     w = torch.cat([w, w.new_ones(1)])
        #     weight = weight * w
        return (loss * weight).mean(1).sum() / B

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        和之前的 reg_loss不同的是,只计算和对应的 proposal的 delta 作为目标回归对象
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
        scores = self.predict_probs(predictions, proposals) # already softmax or sigmoid
        image_shapes = [x.image_size for x in proposals]
        
        vlm_box_features = self.test_pooler([clip_feats], [Boxes(box) for box in boxes])
        # vlm pooler layer: clip attenpool
        vlm_box_features = attenpool(vlm_box_features)
        vlm_box_features = vlm_box_features / vlm_box_features.norm(dim=1,keepdim=True)
        logits_scale = 1/0.01
        vlm_scores = logits_scale * self.get_logits(vlm_box_features, self.text_feats)
        num_inst_per_image = [len(p) for p in proposals]
        vlm_scores[:, self.unused_index] = float('-inf')
        if not self.use_sigmoid_ce:
            vlm_scores = torch.nn.functional.softmax(vlm_scores, dim=1)
        else:
            vlm_scores = torch.sigmoid(vlm_scores)
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
        # for softmax
        # for simgmoid ce 
        # 0 会自动过滤掉; 0: unseen box or unused box
        base_scores = ((scores * self.base_ones)**(1-self.base_alpha)) * ((vlm_scores*self.base_ones)**(self.base_alpha))
        novel_scores = ((scores * self.novel_ones)**(1-self.novel_beta)) * ((vlm_scores*self.novel_ones)**(self.novel_beta))
        ensembled_socres = base_scores + novel_scores
        
        ensembled_socres = torch.cat([ensembled_socres[:,:-1], scores[:, -1:]], dim=1)
        # the unused index pron has been set to 0 after softmax
        ensembled_socres = ensembled_socres / ensembled_socres.sum(dim=1, keepdim=True)
        ensembled_socres = ensembled_socres[:, :-1]
        assert ensembled_socres[:, self.unused_index].max() < 1e-5, 'unused classes should not be evaluated'

        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = ensembled_socres > self.test_score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        ensembled_socres = ensembled_socres[filter_mask]

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, ensembled_socres, filter_inds[:, 1], self.test_nms_thresh)
        if self.test_topk_per_image >= 0:
            keep = keep[:self.test_topk_per_image]
        boxes, ensembled_socres, filter_inds = boxes[keep], ensembled_socres[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = ensembled_socres
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]
