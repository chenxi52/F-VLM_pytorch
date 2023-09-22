# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes, BitMasks,ROIMasks
import detectron2.utils.comm as comm

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import build_backbone, build_proposal_generator, build_roi_heads, build_mask_head
from detic.modeling.sam import sam_model_registry
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
        mask_thr_binary=0.5,
        do_postprocess=True,
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
        
        self.mask_thr_binary = mask_thr_binary
        self.do_postprocess = do_postprocess

    @classmethod
    def from_config(cls, cfg):
        sam = sam_model_registry[cfg.MODEL.BACKBONE.TYPE]()
        backbone = build_backbone(cfg)# fpn+image_encoder
        # roi_heads include box_heads, mask_heads
        ret=({
            'backbone':backbone,
            'proposal_generator':build_proposal_generator(cfg, backbone.output_shape),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape),
            'fp16': cfg.FP16,
            'input_format': cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "sam": sam,
            "do_postprocess": cfg.TEST.DO_POSTPROCESS
        })
        ret.update(mask_thr_binary = cfg.TEST.MASK_THR_BINARY)
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
        # batched_inputs is a dict
        images = self.preprocess_image(batched_inputs) #padding and size_divisiable
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor)

        fpn_features = self.backbone(inter_feats)
        # proposal_generator need to be trained before testing
        proposals, _ = self.proposal_generator(images, fpn_features, None) #samFpn # proposals: img_height=img_width=1024
        results, _ = self.roi_heads(self.sam, images, img_embedding_feat, fpn_features, proposals)
        # batched_inputs have ori_image_sizes
        # images.image_sizes have input_image_sizes
        img_input_sizes = [(inp['input_height'], inp['input_width']) for inp in batched_inputs]
        ori_sizes = [(inp['height'], inp['width']) for inp in batched_inputs]
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return self._postprocess(
                instances=results, ori_sizes=ori_sizes, image_sizes=img_input_sizes)
        else:
            return results
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=self.do_postprocess)
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor)
        fpn_features = self.backbone(inter_feats)
        # fpn_features: Dict{'feat0': Tuple[2*Tensor[256,32,32]], 'feat1': Tuple[2*Tensor[256,64,64]], ...}
        # resize the img_size in gt_instances to (1024,1024)
        proposals, _ = self.proposal_generator(
            images, fpn_features, gt_instances)
        # proposals: img_width=img_height=1024,  gt_instance:max(h,w)=1024
        # proposals: List[bz * Instance[1000 * Instances(num_instances, image_height, image_width, fields=[proposal_boxes: Boxes(tensor([1,4])), objectness_logits:tensor[1],])]]
        
        _, detector_losses = self.roi_heads(self.sam, images, img_embedding_feat, fpn_features, proposals, gt_instances)
        return detector_losses
        

    def extract_feat(self, batched_inputs):
        # forward sam.image_encoder
        
        feat, inter_features = self.sam.image_encoder(batched_inputs)
        # feat: Tensor[bz, 256, 64, 64]  inter_feats: List[32*Tensor[bz,64,64,1280]]
        return feat, inter_features
    
    def _postprocess(self, instances: List[Dict[str,Instances]], ori_sizes: List[Dict[str, torch.Tensor]], image_sizes: List[Tuple[int,int]]):
        """
        instances: with instance(mask_preds, iou_preds)
        Rescale the output instances to the target size.
        Return: processed_results: List[bz*Dict['instances':Instances('pred_boxes', 'scores', pred_classes', 'pred_masks', 'pred_ious')]]
        """
        # note: private function; subject to changes
        sam_img_size = (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size)
        processed_results = []
        for results_per_img, ori_size, img_size in zip(
            instances, ori_sizes, image_sizes
        ):  
            results = Instances(ori_size, **results_per_img.get_fields())
            
            if results.has('pred_masks'):
                pred_masks = results.pred_masks
                if pred_masks.size(0) == 0:
                    processed_results.append({'instances':results})
                    continue

                mask_per_img = pred_masks.sigmoid()
                masks = F.interpolate(
                    mask_per_img.unsqueeze(1),
                    sam_img_size,
                    mode="bilinear",
                    align_corners=False)
                masks = masks[..., : img_size[0], : img_size[1]]
                masks = F.interpolate(masks, ori_size, mode="bilinear", align_corners=False)
                masks = masks.squeeze(1)
                if self.mask_thr_binary>=0:
                    masks = masks >= self.mask_thr_binary
                else: 
                    raise ValueError('The mask_thr_binary<0')
                # img_size: longest=1024
                results.pred_masks = masks
                
            scale_x, scale_y = (
                ori_size[1] / img_size[1],
                ori_size[0] / img_size[0] 
            )

            if results.has("pred_boxes"):
                output_boxes = results.pred_boxes
            elif results.has("proposal_boxes"):
                output_boxes = results.proposal_boxes
            else:
                output_boxes = None
            assert output_boxes is not None, "Predictions must contain boxes!"

            # scaled to the (ori_height, ori_width)
            output_boxes.scale(scale_x, scale_y)
            output_boxes.clip(ori_size)

            results.pred_boxes = output_boxes
            results = results[output_boxes.nonempty()]
            processed_results.append({'instances':results})

        return processed_results
    