from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals, ROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Instances, Boxes
from detectron2.modeling.matcher import Matcher
from .clip_fast_rcnn import ClipRCNNOutputLayers
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
        select_fore_cls: bool = False,
        box_prompter: str='Roi',
        add_pe_before_mask_pool: bool = False,
        roi_prompter: str = "",
        roi_prompter_fuse_type: str = "",
        add_fpn_pe: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            positional_encoding: added to FPN features
        """
        super().__init__(**kwargs)
        for name, value in locals().items():
            if name == 'self':
                continue
            else:
                setattr(self, name, value)
        self.generator_pe = SinePositionalEncoding(num_feats=128, normalize=True)

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
        ret['add_pe_before_mask_pool'] = cfg.MODEL.ROI_MASK_HEAD.ADD_PE_BEFORE_POOL
        ret['roi_prompter'] = cfg.MODEL.ROI_MASK_HEAD.ROI_PROMPTER
        ret['roi_prompter_fuse_type'] = cfg.MODEL.ROI_MASK_HEAD.ROI_PROMPTER_FUSE_TYPE
        ret['add_fpn_pe'] = cfg.MODEL.FPN.ADD_PE
        return ret
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        box_head = ret["box_head"]
        #update the rcnn output layer
        ret.update(box_predictor = ClipRCNNOutputLayers(cfg, box_head.output_shape))
       
        ################
        return ret

    def _forward_box(self, attenpool, clip_final_feats: torch.Tensor, 
                     fpn_feats: Dict[str, torch.Tensor], 
                     proposals: List[Instances]):
        """
        add VLM pooling layer
        clip_feats: [bsz, 2048, 32, 32]
        """
        features = [fpn_feats[f] for f in self.box_in_features]
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
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, clip_final_feats, attenpool)
            return pred_instances
        

    def forward( 
            self,
            clip_features,
            attnpool: nn.Module,
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
        clip_img_feats, clip_fpn_feats = clip_features
        ###########
        if self.add_fpn_pe:
            x = [item[1] for item in list(clip_fpn_feats.items())]
            bs, _, h, w = x[-1].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feat_pe = self.generator_pe(mask_pe)
            for i in range(len(x)):
                x[i] = x[i] + torch.nn.functional.interpolate(img_feat_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
            clip_fpn_feats = {list(clip_fpn_feats.keys())[i]: x[i] for i in range(len(clip_fpn_feats))}
        ############
        if self.training:
            losses = self._forward_box(attnpool, clip_final_feats=None, fpn_feats=clip_fpn_feats, proposals=proposals)
            if self.mask_on:
                losses.update(self._forward_mask(clip_fpn_feats, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(attnpool, clip_img_feats, clip_fpn_feats, proposals)
            if self.mask_on:
                pred_instances = self.forward_with_given_boxes(clip_fpn_feats, pred_instances)
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