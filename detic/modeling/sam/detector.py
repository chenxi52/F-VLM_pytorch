# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detic.modeling.sam.modeling.postprocess_sam_mask import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from torch.cuda.amp import autocast
from detectron2.modeling import build_backbone, build_proposal_generator, build_roi_heads
from detic.modeling.sam import sam_model_registry
import copy
import torch.nn.functional as F
import ipdb

@META_ARCH_REGISTRY.register()
class SamDetector(GeneralizedRCNN):
    """
    build roi and fpn in from_config()
    """
    @configurable
    def __init__(
        self,
        fp16=False,
        sam=None,
        **kwargs
    ):
        """
        kwargs:
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
        """
        self.fp16=fp16
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        self.sam = sam
    
    @classmethod
    def from_config(cls, cfg):
        # ret = super().from_config(cfg)
        sam = sam_model_registry[cfg.MODEL.BACKBONE.TYPE](cfg.MODEL.WEIGHTS)
        # sam_img_encoder = copy.deepcopy(sam.image_encoder)

        # the img_encoder and img_feat are not passes to buil_backbone
        backbone = build_backbone(cfg)# fpn+image_encoder
        ret=({
            'backbone':backbone,
            'proposal_generator':build_proposal_generator(cfg, backbone.output_shape),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape),
            'fp16': cfg.FP16,
            'input_format': cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "sam": sam
        })
        return ret

    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]]=None,
            do_postprocess:bool= True,
    ):
        assert not self.training
        assert detected_instances is None
        # normalize images
        images = self.preprocess_image(batched_inputs)
        # the augmentation is not sure?
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor)

        fpn_features = self.backbone(inter_feats)
        # proposal_generator need to be trained before testing
        proposals, _ = self.proposal_generator(images, fpn_features, None)
        
        results, _ = self.roi_heads(self.sam, images, img_embedding_feat, fpn_features, proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return self._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor)
        ann_type = 'bbox'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        origin_size = [(x['height'], x['width']) for x in batched_inputs]
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                fpn_features = self.backbone([feat.half() for feat in inter_feats]) #Feature aggretor
            fpn_features = {k: v.float() for k, v in fpn_features.items()}
        else:
            fpn_features = self.backbone(inter_feats)
        # fpn_features: Dict{'feat0': Tuple[2*Tensor[256,32,32]], 'feat1': Tuple[2*Tensor[256,64,64]], ...}

        proposals, proposal_losses = self.proposal_generator(
            images, fpn_features, gt_instances)
        # proposals: List[bz * Instance[1000 * Instances(num_instances, image_height, image_width, fields=[proposal_boxes: Boxes(tensor([1,4])), objectness_logits:tensor[1],])]]
        
        predictions, detector_losses = self.roi_heads(self.sam, images, img_embedding_feat, fpn_features, proposals, gt_instances, origin_size)
        return detector_losses
        

    @torch.no_grad()
    def extract_feat(self, batched_inputs):
        # forward sam.image_encoder
        feat, inter_features = self.sam.image_encoder(batched_inputs)
        # feat: Tensor[bz, 256, 64, 64]  inter_feats: List[32*Tensor[bz,64,64,1280]]
        return feat, inter_features
    
    @staticmethod
    def _postprocess(self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        import ipdb
        ipdb.set_trace()
        sam_img_size = (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size)
        processed_results = []
        for results_per_img, input_per_img, img_size in zip(
            instances, batched_inputs, image_sizes
        ):
            ori_height = input_per_img.get("height")
            ori_width = input_per_img.get("width")
            masks = F.interpolate(
                results_per_img.mask_pred,
                sam_img_size,
                mode="bilinear",
                align_corners=False,
            )
            masks = masks[..., : img_size[0], : img_size[1]]
            masks = F.interpolate(masks, (ori_height, ori_width), mode="bilinear", align_corners=False) 
            results_per_img.mask_pred = masks
            processed_results.append({"instances": results_per_img})

        return processed_results