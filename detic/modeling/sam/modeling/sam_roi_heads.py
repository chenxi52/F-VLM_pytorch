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
import math


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
        
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        if cfg.MODEL.MASK_ON:
            del ret['mask_pooler']

            ret['mask_pooler'] = (
                customRoiPooler(
                    output_size=pooler_resolution,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
                if pooler_type
                else None
            )

        del ret['box_predictor']
        # update box_predictor for bbox_loss 
        ret.update(cls.init_box_head(ret['box_head'].output_shape, cfg))
        return ret

    @classmethod
    def init_box_head(cls, box_out_shape, cfg):
        box_predictor = SamRCNNOutputLayers(cfg, box_out_shape)
        return {
            "box_predictor": box_predictor,
        }

    def _forward_mask(self, sam: nn.Module, img_features: torch.Tensor, features: Dict[str, torch.Tensor], 
                      instances: List[Instances]):
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
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            # len(instances) = bz
        if self.mask_pooler is not None:
            # the box here are fused together, but will be assigned to each level in mask_pooler
            boxes = [i.proposal_boxes if self.training else i.pred_boxes for i in instances]
            # List[bz * List[19*Boxes, 3*Boxes]]
            features = [features[f] for f in self.mask_in_features]
            features, mask_roi_inds = self.mask_pooler(features, boxes)
            if features.size(0)==0:
                results_instances = []
                for ins in instances:
                    ins.pred_masks = torch.tensor([], device=ins.pred_classes.device)
                    results_instances.append(ins)
                return results_instances
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, img_features, instances, mask_roi_inds, sam)


    def forward( 
            self,
            sam: nn.Module,
            img_features: torch.Tensor,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
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
        if self.training:
            assert targets, "'targets' argument is required during training"
            # ROI assigner and sampler works
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        # pe map
        x = [item[1] for item in list(features.items())]
        bs, _, h, w = x[-1].shape #
        mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
        img_feat_pe = self.generator_pe(mask_pe)

        for i in range(len(x)):
            x[i] = x[i] + torch.nn.functional.interpolate(img_feat_pe, size=x[i].shape[-2:], mode='bilinear')
        x = {list(features.keys())[i]: x[i] for i in range(len(features))}
        
        if self.training:
            losses = self._forward_box(x, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head. proposal_boxes are replaced by boxes predicted by box_head
            if self.mask_on:
                losses.update(loss_mask=self._forward_mask(sam, img_features, x, proposals)['loss_mask'])
            return proposals, losses
        else:
            pred_instances = self._forward_box(x, proposals)
            # pred_boxes = Boxes(boxes)   result.scores = scores  pred_classes
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            if self.mask_on:
                pred_instances = self.forward_with_given_boxes(sam, img_features, x, pred_instances)
            return pred_instances, {}
    
    def forward_with_given_boxes(
        self, sam: nn.Module, img_features: torch.Tensor, features: Dict[str, torch.Tensor], instances: List[Instances]
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
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(sam, img_features, features, instances)
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