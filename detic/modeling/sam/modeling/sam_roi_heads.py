from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals, ROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Instances, Boxes
from detectron2.modeling.matcher import Matcher
from detic.modeling.roi_heads import ClipRCNNOutputLayers, SamRCNNOutputLayers
from torch import Tensor
import math
import inspect


@ROI_HEADS_REGISTRY.register()
class samAnchorPromptRoiHeads(StandardROIHeads):
    """
    The roi heads controls the boxes head and mask head.
    """
    @configurable
    def __init__(
        self,
        *,
        mask_on: bool=True,
        input_size: int = 1024,
        sam_on: bool = False,
        select_fore_cls: bool = False,
        box_prompter: bool = False,
        generate_pe: Tensor = None,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            positional_encoding: added to FPN features
            input_size: input size for sam image_encoder
        """
        super().__init__(**kwargs)
        self.mask_on = mask_on 
        self.input_size = input_size
        self.sam_on = sam_on
        self.select_fore_cls = select_fore_cls 
        self.box_prompter = box_prompter
        self.generate_pe = generate_pe
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        """
        ret = ROIHeads.from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))

        ret['mask_on'] = cfg.MODEL.MASK_ON
        ret['input_size'] = cfg.INPUT.TRAIN_SIZE
        # add allow_quality to the cfg.
        ret['proposal_matcher'] = Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=cfg.MODEL.ROI_HEADS.ALLOW_LOW_QUALITY_MATCHES,
            )
        ret['select_fore_cls'] = cfg.MODEL.ROI_MASK_HEAD.SELECT_FORE_CLS
        ret['box_prompter'] = cfg.MODEL.ROI_MASK_HEAD.BOX_PROMPTER
        ret['generate_pe'] = nn.Parameter(torch.randn(1, 256, 32, 32))
        return ret
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_head = ret["box_head"]
        #update the rcnn output layer
        if not cfg.MODEL.SAM_ON:
            ret.update(box_predictor = ClipRCNNOutputLayers(cfg, box_head.output_shape))
        else:
            # remove loss_cls
            ret.update(box_predictor = SamRCNNOutputLayers(cfg, box_head.output_shape))
        ret['sam_on'] = cfg.MODEL.SAM_ON
        ################
        return ret
    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        ret = super()._init_mask_head(cfg, input_shape) 
        if cfg.MODEL.SAM_ON:
            ret["mask_pooler"] = (
                ROIPooler(
                    output_size=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
                    scales=[1./16,],
                    sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
                    pooler_type=cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE,
                )
                if cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
                else None
            )
            
        return ret

    def forward_sam_mask(
            self, 
            instances: List[Instances], 
            clip_final_feats: torch.Tensor,
            sam: nn.Module,
            sam_features: torch.Tensor,
            attnpool: nn.Module,
            ):
        """
        Args:
            img_features: features output by image_encoder
            features: Multi-level features
            instances (list[Instances]): 
                proposals from rpn. the per-image instances to train/predict masks. 
                have predicted_boxes of _forward_box_head
                        In training, they can be the proposals.
                        In inference, they can be the boxes predicted by R-CNN box head.
        """
        if self.training and self.select_fore_cls:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        boxes = [i.proposal_boxes if self.training else i.pred_boxes for i in instances]
        if not self.box_prompter:
            if self.mask_pooler is not None:
                # sam_features 大小和 clip_features不一样
                # mask pool 修改
                features = self.mask_pooler([sam_features], boxes)
                if features.size(0)==0:
                    results_instances = []
                    for ins in instances:
                        ins.pred_masks = torch.tensor([], device=ins.pred_classes.device)
                        results_instances.append(ins)
                    return results_instances
            else:
                assert NotImplementedError
                features = [features[f] for f in self.mask_in_features]
        else:
            features = None
        return self.mask_head(roi_features=features, 
                              instances=instances,
                              sam=sam, 
                              sam_features=sam_features, 
                              clip_final_feats=clip_final_feats, 
                              boxes=boxes, 
                              attnpool=attnpool,
                              select_fore_cls=self.select_fore_cls)

    def _forward_box(self, attenpool, clip_final_feats: torch.Tensor, 
                     features: List[torch.Tensor], 
                     proposals: List[Instances]):
        """
        add VLM pooling layer
        clip_feats: [bsz, 2048, 32, 32]
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features) # here, box
        predictions = self.box_predictor(box_features)
        del box_features
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            # propsal_boxes is relative to the original image size.
            # roi align will assign level
            if not self.sam_on:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals, clip_final_feats, attenpool)
            else: 
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
        

    def forward( 
            self,
            sam,
            sam_features: torch.Tensor,
            attnpool: nn.Module,
            clip_final_feats: torch.Tensor,
            fpn_features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            )-> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
            clip_features: output of image_encoder
            fpn_features: multi-level features output by FPN
        Return: pred_instances
        """
        if self.training:
            assert targets, "'targets' argument is required during training"
            # ROI assigner and sampler works.
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        x = [fpn_features[f] for f in self.box_in_features]
        # add pos to fpn features
        for i in range(len(x)):
            x[i] = x[i] + torch.nn.functional.interpolate(self.generate_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False).to(x[i].device)
        if self.training:
            losses = self._forward_box(attnpool, clip_final_feats, x, proposals)
            if self.mask_on:
                if sam_features is not None:
                    losses.update(self.forward_sam_mask(instances=proposals, 
                                                        clip_final_feats=clip_final_feats, 
                                                        sam=sam, 
                                                        sam_features=sam_features,
                                                        attnpool=None))
                else: losses.update(self._forward_mask(x, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(attnpool, clip_final_feats, x, proposals)
            if self.mask_on:
                if sam_features is not None:
                    assert pred_instances[0].has("pred_boxes")
                    pred_instances = self.forward_sam_mask(pred_instances, 
                                                           clip_final_feats, 
                                                           sam, 
                                                           sam_features, 
                                                           attnpool=attnpool)
                else:
                    pred_instances = self.forward_with_given_boxes(x, pred_instances)
            return pred_instances, {}
    

    
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