from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN, PROPOSAL_GENERATOR_REGISTRY
import torch.nn as nn
from detectron2.config import configurable
from typing import Dict, List, Optional, Tuple, Union
import torch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import _dense_box_regression_loss, Box2BoxTransform
from detectron2.layers import Conv2d, ShapeSpec, cat
import torch.nn.functional as F
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.matcher import Matcher
# rpn changing objectness loss, use centerness loss

@PROPOSAL_GENERATOR_REGISTRY.register()
class SAMRPN(RPN):
    """
    add objectness_loss_type
    add objectness cfg
    """
    @configurable
    def __init__(self, 
                 *,
                objectness_loss_type: str = 'sigmoid_focal_loss',
                **kwargs):
        super().__init__(**kwargs)
        self.objectness_loss_type = objectness_loss_type

    @classmethod
    def from_config(cls, cfg, input_shape):
        # add objectness_loss_type
        ret = super().from_config(cfg, input_shape)
        ret['objectness_loss_type'] = cfg.MODEL.RPN.OBJECTNESS_LOSS_TYPE
        return ret

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        add self.objectness_loss_type == 'centerness':
        """
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            decide whether the anchor is inside the gt_boxes, and compute the centerness target.
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)

            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            # gt_lable=-1: ignore, not the centerness too
            # gt_lable=0: negative sample, centernesss=0 
            # centerness_target for each anchor. multiply the centerness_target_i with the gt_label_i
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:    
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            if self.objectness_loss_type == 'centerness':
                gt_centerness = retry_if_cuda_oom(get_centerness)(anchors, matched_gt_boxes_i)
                gt_centerness = gt_centerness.to(device=gt_boxes_i.device)
                gt_labels_i = gt_labels_i.to(torch.float32)
                gt_labels_i = torch.where(gt_centerness>0, gt_centerness*gt_labels_i, gt_labels_i)
                # only negtive boxes are set to gt_labels=0
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes
    
    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # only difference, to suit for centerness 
        pos_mask = gt_labels>0
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)
        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
    
def get_centerness(anchors, gt_boxes_i):
    anchors_centers = anchors.get_centers()
    gt_reg = torch.zeros_like(anchors.tensor, device=gt_boxes_i.device)
    # gt_boxes: xyxy
    gt_reg[:,0] = anchors_centers[:,0] - gt_boxes_i[:,0]
    gt_reg[:,1] = anchors_centers[:,1] - gt_boxes_i[:,1]
    gt_reg[:,2] = gt_boxes_i[:,2] - anchors_centers[:,0]
    gt_reg[:,3] = gt_boxes_i[:,3] - anchors_centers[:,1]
    is_in_boxes = gt_reg.min(dim=-1)[0] > 0
    gt_centerness = compute_centerness_targets(gt_reg, is_in_boxes)
    return gt_centerness
    
def compute_centerness_targets(reg_targets, is_inside_box):
    # reg_targets is [N, 4]， （x-x0, y-y0, x1-x, y1-y)
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = torch.zeros_like(reg_targets[:, 0])
    centerness[is_inside_box] = torch.sqrt(
        (left_right[is_inside_box].min(dim=-1)[0] / left_right[is_inside_box].max(dim=-1)[0]) *
        (top_bottom[is_inside_box].min(dim=-1)[0] / top_bottom[is_inside_box].max(dim=-1)[0])
    )
    return centerness
