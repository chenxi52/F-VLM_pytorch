# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler, ROIAlign
from detectron2.modeling import build_roi_heads, ROI_HEADS_REGISTRY, StandardROIHeads, build_mask_head
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Instances, ImageList, pairwise_iou, BitMasks
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
import copy 

@ROI_HEADS_REGISTRY.register()
class samAnchorPromptRoiHeads(StandardROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.

    the returns of _int_*_head will be the input of __init__
    """
    @configurable
    def __init__(
        self,
        *,
        positional_encoding = dict(num_feats=128, normalize=True),
        mask_on: bool=True,
        input_size: int = 1024,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.generator_pe = SinePositionalEncoding(**positional_encoding)
        self.mask_on = mask_on 
        self.input_size = input_size

    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        # super().from_config and _init_box_head, _init_mask_head
        _init_box_head has in_features, 
        """
        # fmt: off
        ret = super().from_config(cfg, input_shape)
        mask_on   = cfg.MODEL.MASK_ON
        input_size = cfg.INPUT.TRAIN_SIZE
        # fmt: on
        # maybe mask_forward need sam
        ret['mask_on'] = mask_on
        ret['input_size'] = input_size
        return ret


    def _forward_mask(self, sam: nn.Module, img_features: torch.Tensor, features: List[torch.Tensor], 
                      instances: List[Instances],origin_img_size: List[Tuple[int,int]]):
        """
        Forward logic of the mask prediction branch.
        Args:
            img_features: features output by image_encoder
            features: Multi-level features
            instances (list[Instances]): the per-image instances to train/predict masks. 
                have predicted_boxes of _forward_box_head
                        In training, they can be the proposals.
                        In inference, they can be the boxes predicted by R-CNN box head.
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            # len(instances) = bz
        if self.mask_pooler is not None:
            # x = [features[f] for f in self.mask_in_features]
            boxes = [i.proposal_boxes if self.training else i.pred_boxes for i in instances]
            # List[bz * List[19*Boxes, 3*Boxes]]
            img_flags_freq = [len(box) for box in boxes]
            features = self.mask_pooler(features, boxes)
            # len(Boxes)* Torch.tensor[256,14,14]
        else:
            features = {f: features[f] for f in self.mask_in_features}
        # prompt_head + mask_decoder
        return self.mask_head(features, img_features, instances, sam, img_flags_freq, origin_img_size)


    def forward( 
            self,
            sam: nn.Module,
            images: ImageList,
            img_features: torch.Tensor,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            origin_img_size: List[Tuple[int, int]] = None
            )-> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
            img_features: output of image_encoder
            features: multi-level features output by FPN
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        Return: pred_instances
            list[Instances]: length `N` list of `Instances` containing the
                detected instances. Returned during inference only; may be [] during training.
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        del images
        # len(targets[0]) = 5; len(targets[1]) = 1
        # proposals[0][0]: Instances(num_instances=1, image_height=683, image_width=1024, 
            # fields=[proposal_boxes: Boxes(tensor([[ 44.6688, 107.2670,  50.0604, 118.8706]], device='cuda:0')), 
            # objectness_logits: tensor([0.1904], device='cuda:0')])
        # targets[0][0]: Instances(num_instances=1, image_height=683, image_width=1024, 
            # fields=[gt_boxes: Boxes(tensor([[ 59.3920, 407.0560, 121.1200, 500.9760]], device='cuda:0')), 
            # gt_classes: tensor([0], device='cuda:0'), gt_masks: PolygonMasks(num_instances=1)])

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            #Instances(num_instances=1, image_height=683, image_width=1024, 
                # fields=[proposal_boxes: Boxes(tensor([[196.3040, 415.1680, 223.5360, 464.7520]], device='cuda:0')), 
                # objectness_logits: tensor([23.0259], device='cuda:0'), gt_classes: tensor([0], device='cuda:0'), 
                # gt_boxes: Boxes(tensor([[196.3040, 415.1680, 223.5360, 464.7520]], device='cuda:0')), 
                # gt_masks: PolygonMasks(num_instances=1)])
            # proposals: List[bz * List[512*Instances]]
        del targets
        # pe map
        x = [item[1] for item in list(features.items())]
        bs, _, h, w = x[0].shape
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feat_pe = self.generator_pe(mask_pe)

        for i in range(len(features)):
            x[i] = x[i] + torch.nn.functional.interpolate(img_feat_pe, size=x[i].shape[-2:], mode='bilinear')
        
        if self.training:
            losses = self._forward_box(features, proposals)
            # losses: Instance
            # {'loss_cls': , 'loss_box_reg'}
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head. proposal_boxes are replaced by boxes predicted by box_head
            # using sam
            losses.update(loss_mask=self._forward_mask(sam, img_features, x, proposals, origin_img_size)['loss_mask'])
            return proposals, losses
        else:
            #proposals=None
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(sam, img_features, features, pred_instances, origin_img_size)
            return pred_instances, {}
    
    def forward_with_given_boxes(
        self, sam: nn.Module, img_features: torch.Tensor, features: Dict[str, torch.Tensor], instances: List[Instances], origin_img_size: Tuple[int,int]
        ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [item[1] for item in features.items()]
        instances = self._forward_mask(sam, img_features, features, instances, origin_img_size)
        return instances

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes, gt_masks = self.sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, 
                targets_per_image.gt_masks, (self.input_size, self.input_size)
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]

            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_masks = gt_masks

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    

    def sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, gt_masks: torch.Tensor, img_shape: Tuple[int,int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_masks = gt_masks[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            # NOTE: only for bitmask mask labeling-format
            # Mask has [1024,1024] in training
            gt_masks = BitMasks(
                    torch.stack([torch.zeros(img_shape) for i in matched_idxs])
                ).to(gt_classes.device)
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs], gt_masks[sampled_idxs]


import math
from torch import Tensor
class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.) -> None:
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str    