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
from torchvision.utils import save_image

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN, detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import build_backbone, build_proposal_generator, build_roi_heads, build_mask_head
from detic.modeling.sam import sam_model_registry
import torch.nn.functional as F
import matplotlib.pyplot as plt
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
from detic.modeling.clip import clip
from detic.prompt_engineering import get_prompt_templates
from detic import constants

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
        backbone_name=None,
        **kwargs
    ):
        """
        kwargs:
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            input_format: Optional[str] = None,
            vis_period: int = 0,
        """
        self.fp16=fp16
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        self.sam = sam
        
        self.mask_thr_binary = mask_thr_binary
        self.do_postprocess = do_postprocess
        self.backbone_name = backbone_name

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
            "do_postprocess": cfg.TEST.DO_POSTPROCESS,
            "backbone_name":cfg.MODEL.BACKBONE.NAME
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

        results, _ = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals)
        ###########################
        # batched_inputs have ori_image_sizes
        # images.image_sizes have input_image_sizes
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return self.postprocess(pred_instances=results, batched_inputs=batched_inputs, mask_threshold=self.mask_thr_binary)
        else:
            return results
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=self.do_postprocess)
        # batched_inputs have longes_side=1024, and prepocess need the shortest_side%32==0
        # images.size() is origin image size

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor) 
        fpn_features = self.backbone(inter_feats)
        # fpn_features: Dict{'feat0': Tuple[2*Tensor[256,32,32]], 'feat1': Tuple[2*Tensor[256,64,64]], ...}
        # resize the img_size in gt_instances to (1024,1024)
        proposals, proposal_losses = self.proposal_generator(
            images, fpn_features, gt_instances)
        # proposals:max(h,w)=1024,  gt_instance:max(h,w)=1024
        del images
        _, detector_losses = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    
    def extract_feat(self, batched_inputs):
        # forward sam.image_encoder
        if 'det' in self.backbone_name:
            feat,inter_features = self.sam.image_encoder(batched_inputs)
            inter_features = feat
        else:
            # tiny image encoder are not implemented now
            feat, inter_features = self.sam.image_encoder(batched_inputs)
        # feat: Tensor[bz, 256, 64, 64]  inter_feats: List[32*Tensor[bz,64,64,1280]]
        return feat, inter_features
    
    def postprocess(self, pred_instances, batched_inputs: List[Dict[str, torch.Tensor]], mask_threshold:float):
        """
        Rescale the output instances to the target size.
        image_sizes should be the origin size of images
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            pred_instances, batched_inputs
        ):
            # height, width: input_image_sizes， 原本的 img_size
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = custom_detector_postprocess(results_per_image, height, width, mask_threshold=mask_threshold)
            processed_results.append({"instances": r})
        return processed_results
    
@META_ARCH_REGISTRY.register()
class SamOpenDetector(SamDetector):
    @configurable
    def __init__(
        self,
        fp16=False,
        sam_type=None,
        clip_type=None,
        mask_thr_binary=0.5,
        do_postprocess=True,
        backbone_name=None,
        class_name=constants.COCO_INSTANCE_CLASSES,
        add_unfrozen='xxx',
        **kwargs
    ):
        self.fp16=fp16
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        self.sam = sam_model_registry[sam_type]()
        self.clip = clip.load(clip_type, jit=False)[0]
        self.mask_thr_binary = mask_thr_binary
        self.do_postprocess = do_postprocess
        self.backbone_name = backbone_name
        self.class_name = class_name
        self.text_feats =  self.get_custom_text_feat(self.class_name)
        # set params in sam and clip to no_grad
        for name, params in self.sam.named_parameters():
            if add_unfrozen in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        for name, params in self.clip.named_parameters():
            params.requires_grad = False
        self.register_buffer("clip_pixel_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(1).unsqueeze(2), False)
        self.register_buffer("clip_pixel_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(1).unsqueeze(2), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)# fpn+image_encoder
        # roi_heads include box_heads, mask_heads
        if 'coco' in cfg.DATASETS.TRAIN[0]:
            class_name = constants.COCO_INSTANCE_CLASSES
        ret=({
            'backbone':backbone,
            'proposal_generator':build_proposal_generator(cfg, backbone.output_shape),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape),
            'fp16': cfg.FP16,
            'input_format': cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "sam_type": cfg.MODEL.BACKBONE.TYPE,
            "clip_type": cfg.MODEL.BACKBONE.CLIP_TYPE,
            "do_postprocess": cfg.TEST.DO_POSTPROCESS,
            "backbone_name":cfg.MODEL.BACKBONE.NAME,
            "class_name":class_name,
            "add_unfrozen":cfg.MODEL.BACKBONE.ADD_UNFROZEN
        })
        ret.update(mask_thr_binary = cfg.TEST.MASK_THR_BINARY)
        return ret
    
    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]]=None,
            do_postprocess:bool= True,
            resized_images:torch.Tensor=None
        ):
        assert not self.training
        assert detected_instances is None
        # normalize images
        # batched_inputs is a dict
        images = self.preprocess_image(batched_inputs) #padding and size_divisiable
        img_embedding_feat, inter_feats, clip_images = self.extract_feat(images.tensor, resized_images)
        
        fpn_features = self.backbone(inter_feats)
        # proposal_generator need to be trained before testing
        proposals, _ = self.proposal_generator(images, fpn_features, None) #samFpn # proposals: img_height=img_width=1024
        results, _ = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals, targets=None,
                                    clip=self.clip, clip_images=clip_images, clip_texts=self.text_feats )
        # batched_inputs have ori_image_sizes
        # images.image_sizes have input_image_sizes
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return self.postprocess(pred_instances=results, batched_inputs=batched_inputs, mask_threshold=self.mask_thr_binary)
        else:
            return results
        
    def extract_feat(self, batched_inputs, resized_images):
        # forward sam.image_encoder
        if 'det' in self.backbone_name:
            feat,inter_features = self.sam.image_encoder(batched_inputs)
            inter_features = feat
        else:
            # tiny image encoder are not implemented now
            feat, inter_features = self.sam.image_encoder(batched_inputs)
        clip_img = self.clip.encode_image_feature(resized_images)
    
        # feat: Tensor[bz, 256, 64, 64]  inter_feats: List[32*Tensor[bz,64,64,1280]]
        return feat, inter_features, clip_img
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        resized_images = self.resize_norm(batched_inputs)
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=self.do_postprocess, resized_images=resized_images)
        # batched_inputs have longes_side=1024, and prepocess need the shortest_side%32==0
        # images.size() is origin image size
        # preprocess are normalized
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        
        img_embedding_feat, inter_feats, clip_feats = self.extract_feat(images.tensor, resized_images) 
        fpn_features = self.backbone(inter_feats)
        # fpn_features: Dict{'feat0': Tuple[2*Tensor[256,32,32]], 'feat1': Tuple[2*Tensor[256,64,64]], ...}
        # resize the img_size in gt_instances to (1024,1024)
        proposals, proposal_losses = self.proposal_generator(
            images, fpn_features, gt_instances)
        # proposals:max(h,w)=1024,  gt_instance:max(h,w)=1024
        del images
        _, detector_losses = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals, gt_instances, self.clip, clip_feats, self.text_feats )
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
            
    def resize_norm(self, batched_inputs, target_size=(224, 224)):
        # Convert the numpy image to a torch tensor and ensure it is in CxHxW format
        images = [self._move_to_current_device((x["image"]/255.).to(torch.float)) for x in batched_inputs]
        resized_images = [F.interpolate(x.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(1) for x in images]
        # Apply normalization
        resized_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in resized_images]
        return torch.cat(resized_images,dim=0)
    
    @torch.no_grad()
    def get_custom_text_feat(self, class_names):
        def extract_mean_emb(text):
            tokens = clip.tokenize(text).cuda()
            if len(text) > 10000:
                text_features = torch.cat([
                    self.clip.encode_text(text[:len(text) // 2]),
                    self.clip.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = self.clip.encode_text(tokens)
            
            text_features = torch.mean(text_features, 0, keepdims=True)
            return text_features[0]

        templates = get_prompt_templates()
        clss_embeddings = []
        for clss in class_names:
            txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
            clss_embeddings.append(extract_mean_emb(txts))
        txts = ['background']
        clss_embeddings.append(extract_mean_emb(txts))
        text_emb = torch.stack(clss_embeddings, dim=0)
        text_emb /= text_emb.norm(dim=-1, keepdim=True) 
    
        return text_emb

def custom_detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Inputs:
        results: the pred_masks of (1024,1024),results.image_size: (1024, x) or (x,1024)
        output_height, output_width: the original img sie 
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    input_size = results.image_size
    results = Instances(new_size, **results.get_fields())
    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    # resize box from 
    # output_boxes.scale(scale_x, scale_y)
    # output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]
    #1. paste mask to [1024,1024], original is [1024,1024]
    if results.has("pred_masks"):
        # if isinstance(results.pred_masks, ROIMasks):
        #     roi_masks = results.pred_masks
        # else:
        #     # pred_masks is a tensor of shape (N, 1, M, M)
        #     roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        mask_tensor = results.pred_masks
        # mask_tensor = roi_masks.tensor
        # results.pred_masks = roi_masks.to_bitmasks(
        #     results.pred_boxes, 1024, 1024, mask_threshold
        # ).tensor  # TODO return ROIMasks/BitMask object in the future
        #2. clip up the paddings
        mask_tensor = mask_tensor[:, :, :input_size[0], :input_size[1]]
        #3. resize the box and mask, give it to the results
        mask_tensor = F.interpolate(mask_tensor, size=new_size, mode="bilinear", align_corners=False).squeeze(1)
        mask_tensor = (mask_tensor>=mask_threshold).to(torch.bool)
        results.pred_masks = mask_tensor
    output_boxes.scale(output_width_tmp/input_size[1], output_height_tmp/input_size[0])
    results.pred_boxes = output_boxes
    return results