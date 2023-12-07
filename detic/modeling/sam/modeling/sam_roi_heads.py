# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.modeling import build_roi_heads, ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Instances, ImageList, pairwise_iou, BitMasks, Boxes
from detectron2.modeling.matcher import Matcher
from detic.modeling.roi_heads.sam_fast_rcnn import SamRCNNOutputLayers
from torch import Tensor
from detic.modeling.custom_poolers import customRoiPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
import math
from detectron2.utils.events import get_event_storage
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.modeling.roi_heads.box_head import build_box_head
@ROI_HEADS_REGISTRY.register()
class samAnchorPromptRoiHeads(StandardROIHeads):
    """
    The roi heads controls the boxes head and mask head.
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
            positional_encoding: added to FPN features
            mask_on: whether to use mask head
            input_size: input size for sam image_encoder
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
        ret = super().from_config(cfg, input_shape)
        mask_on   = cfg.MODEL.MASK_ON
        input_size = cfg.INPUT.TRAIN_SIZE
        ret['mask_on'] = mask_on
        ret['input_size'] = input_size
        # add allow_quality to the cfg.
        ret['proposal_matcher'] = Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=cfg.MODEL.ROI_HEADS.ALLOW_LOW_QUALITY_MATCHES,
            )
        # update box_predictor for bbox_loss 
        return ret
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        ################
        #update the rcnn output layer
        box_predictor = SamRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


    def _forward_mask(self, sam: nn.Module,  img_features: torch.Tensor,features: Dict[str, torch.Tensor],
                      instances: List[Instances], clip:nn.Module, clip_images: torch.Tensor, clip_texts: torch.Tensor,
                      context_former_pe:nn.Module=None):
        """
        Forward logic of the mask prediction branch.
        Args:
            img_features: features output by image_encoder
            features: Multi-level features
            instances (list[Instances]): 
                proposals from rpn. the per-image instances to train/predict masks. 
                with fg_proposals and bg_proposals
                have predicted_boxes of _forward_box_head
                        In training, they can be the proposals.
                        In inference, they can be the boxes predicted by R-CNN box head.
        """
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        boxes = [i.proposal_boxes if self.training else i.pred_boxes for i in instances]
        if self.mask_pooler is not None:
            # the box here are fused together, but will be assigned to each level in mask_pooler
            features = [features[f] for f in self.mask_in_features]
            features = self.mask_pooler(features, boxes)
            # mask_roi_inds = [box.tensor.size(0).to(box.device) for box in boxes]
            if features.size(0)==0:
                results_instances = []
                for ins in instances:
                    ins.pred_masks = torch.tensor([], device=ins.pred_classes.device)
                    results_instances.append(ins)
                return results_instances
        else:
            features = [features[f] for f in self.mask_in_features]
        return self.mask_head(features, img_features, instances, sam, clip, clip_images, clip_texts, context_former_pe)


    def forward( 
            self,
            sam: nn.Module,
            img_features: torch.Tensor,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            clip: nn.Module=None,
            clip_images: torch.Tensor=None,
            clip_texts: torch.Tensor=None,
            context_former_pe: nn.Module=None,
            )-> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
            img_features: output of image_encoder
            features: multi-level features output by FPN
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
        Return: pred_instances
            list[Instances]: length `N` list of `Instances` containing the
                detected instances. Returned during inference only; may be [] during training.
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # import ipdb; ipdb.set_trace()
        if self.training:
            assert targets, "'targets' argument is required during training"
            # ROI assigner and sampler works.
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        # pe map
        # confusing
        x = [item[1] for item in list(features.items())]
        bs, _, h, w = x[-1].shape #
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feat_pe = self.generator_pe(mask_pe)

        for i in range(len(x)):
            x[i] = x[i] + torch.nn.functional.interpolate(img_feat_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
        x = {list(features.keys())[i]: x[i] for i in range(len(features))}
        
        if self.training:
            losses = self._forward_box(x, proposals)
            # print(len(proposals[0].proposal_boxes))
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head. proposal_boxes are replaced by boxes predicted by box_head
            losses.update(self._forward_mask(sam, img_features, x, proposals, clip, clip_images, clip_texts, context_former_pe))
            # self._forward_mask(sam, img_features, x, proposals)

            return proposals, losses
        else:
            # dscard the nms from fast_rcnn
            pred_instances = self._forward_box(x, proposals)
            # pred_boxes = Boxes(boxes)   result.scores = scores  pred_classes
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(sam, img_features, x, pred_instances, clip, clip_images, clip_texts, context_former_pe)
            return pred_instances, {}
        
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
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

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

    def forward_with_given_boxes(
        self, sam: nn.Module, img_features: torch.Tensor, features: Dict[str, torch.Tensor], instances: List[Instances],
        clip:nn.Module, clip_images: torch.Tensor, clip_texts: torch.Tensor, context_former_pe:nn.Module=None
        ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()` multi-level roi features
            img_features: image features from image encoder
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes")

        instances = self._forward_mask(sam, img_features, features, instances, clip, clip_images, clip_texts, context_former_pe=context_former_pe)
        # NMS , this is semantic token classification
        return instances


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
        dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_feats)
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