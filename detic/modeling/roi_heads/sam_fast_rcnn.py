# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import torch
from typing import List, Tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes, Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss

__all__ = ["SamRCNNOutputLayers"]
logger = logging.getLogger(__name__)

class SamRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:
    1. proposal-to-detection box regression deltas
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(input_shape, **kwargs)
        
    def forward(self, x):
        """
        remove the classification branch   
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        return None, proposal_deltas
    
    def losses(self, predictions, proposals):
        """
        remove the classification branch
        """
        _, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
        losses = {
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
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
    
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        
        """
        if self.loss_weight["loss_box_reg"] == 0.:
            boxes = [p.proposal_boxes.tensor for p in proposals]
        else:
            boxes = self.predict_boxes(predictions, proposals)
        objectness = [p.objectness_logits.sigmoid() for p in proposals]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            objectness,
            image_shapes,
        )
    
def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    objectness: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],

):
    """
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, objectness_per_image, image_shape, 
        )
        for boxes_per_image, objectness_per_image, image_shape in zip( boxes, objectness, image_shapes)
    ]
    return result_per_image, None



def fast_rcnn_inference_single_image(
    boxes,
    objectness,
    image_shape: Tuple[int, int],
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    if num_bbox_reg_classes == 1:
        boxes = boxes[:, 0]
    else:
        boxes = boxes
    # the number is equal to the RPN output proposals
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.objectness = objectness
    return result
