# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import build_backbone, build_proposal_generator, build_roi_heads
from detic.modeling.sam import sam_model_registry
import torch.nn.functional as F
from detic.modeling.clip import clip
from detic.prompt_engineering import get_prompt_templates
from detic import constants
from torch.cuda.amp import autocast
import detectron2.utils.comm as comm
import pickle
from detic.modeling.utils import load_class_freq
import sys
@META_ARCH_REGISTRY.register()
class ClipOpenDetector(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        fp16=False,
        sam_type=None,
        mask_thr_binary=0.5,
        do_postprocess=True,
        clip_model=None,
        backbone_name=None,
        add_unfrozen='xxx',
        fpn_in_features=[],
        clip_train_size=1024,
        sam_on = False,
        sam_pixel_mean=None,
        sam_pixel_std=None,
        sam_weights=None,
        eval_ar= False,
        zs_weight_path=None,
        **kwargs
    ):
        self.fp16=fp16
        super().__init__(**kwargs)
        self.register_buffer("sam_pixel_mean", torch.tensor(sam_pixel_mean).view(-1,1,1), False)
        self.register_buffer("sam_pixel_std", torch.tensor(sam_pixel_std).view(-1,1,1), False)
        assert self.proposal_generator is not None
        if sam_on:
            self.sam = sam_model_registry[sam_type](checkpoint=sam_weights)
            for name, params in self.sam.named_parameters():
                params.requires_grad = False
        else: self.sam = None
        self.sam_on = sam_on
        self.clip = clip_model
        self.mask_thr_binary = mask_thr_binary
        self.do_postprocess = do_postprocess
        self.backbone_name = backbone_name
        self.fpn_in_features = fpn_in_features
        ####可以从这里保存 text features 、
        # self.text_feats =  self.get_custom_text_feat(constants.COCO_SEEN_CLS)
        # if comm.is_main_process():
        #     with open('datasets/coco/coco_cls_seen.pkl', 'wb') as f:
        #         pickle.dump(self.text_feats, f)
        #     sys.exit()

        ##########
        # set params in sam and clip to no_grad
        for name, params in self.clip.named_parameters():
            params.requires_grad = False
        self.clip_train_size = clip_train_size
        self.eval_ar = eval_ar

    @classmethod
    def from_config(cls, cfg):
        # roi_heads include box_heads, mask_heads
        clip_model,  _ = clip.load(cfg.MODEL.BACKBONE.CLIP_TYPE)
        # FPN backbone
        backbone = build_backbone(cfg, clip_model.visual.output_shape)
        ret=({
            "backbone": backbone, 
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "sam_pixel_mean": cfg.MODEL.SAM_PIXEL_MEAN,
            "sam_pixel_std": cfg.MODEL.SAM_PIXEL_STD,
            "clip_model": clip_model,
            'fp16': cfg.FP16,
            "sam_type": cfg.MODEL.BACKBONE.TYPE,
            "do_postprocess": cfg.TEST.DO_POSTPROCESS,
            "backbone_name":cfg.MODEL.BACKBONE.NAME,
            "add_unfrozen":cfg.MODEL.BACKBONE.ADD_UNFROZEN,
            "clip_train_size":cfg.INPUT.CLIP_TRAIN_SIZE,
            "mask_thr_binary":cfg.TEST.MASK_THR_BINARY,
            'fpn_in_features': cfg.MODEL.FPN.IN_FEATURES,
            "sam_on": cfg.MODEL.SAM_ON,
            "sam_weights": cfg.MODEL.SAM_WEIGHTS,
            "eval_ar": cfg.EVAL_AR,
        })
        return ret
    
    @torch.no_grad()
    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]]=None,
            do_postprocess:bool= True,
            sam_feats=None,
            clip_images=None,
            clip_final_feats=None,
            fpn_feats=None, 
        ):
        assert not self.training
        assert detected_instances is None
        proposals, _ = self.proposal_generator(clip_images, fpn_feats, None) #samFpn # proposals: img_height=img_width=1024
        results, _ = self.roi_heads(self.sam, 
                                    sam_features=sam_feats,
                                    attnpool=self.clip.visual.attnpool,
                                    clip_final_feats=clip_final_feats,
                                    fpn_features=fpn_feats, 
                                    proposals=proposals, 
                                    targets=None)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            if self.sam_on:
                return self.postprocess(pred_instances=results, batched_inputs=batched_inputs, mask_threshold=self.mask_thr_binary)
            return GeneralizedRCNN._postprocess(results, batched_inputs, clip_images.image_sizes)
        else:
            return results
            
    @torch.no_grad()    
    def extract_sam_feat(self, images):
        # extrac feat from clip(multi-level feats) and sam;
        # preprocess: padding 
        if self.fp16: 
            with autocast():
                sam_feat, _ = self.sam.image_encoder(images.half())
        else:
            sam_feat, _ = self.sam.image_encoder(images.float())
        return sam_feat
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        if not self.sam_on:
            clip_images = self.to_imageList(images)
        sam_image_feats = None
        if self.sam_on: 
            clip_images = self.norm_imageList(images, self.pixel_mean, self.pixel_std, norm_val=255.) # ImageList
            sam_images = self.norm_imageList(images, self.sam_pixel_mean, self.sam_pixel_std, norm_val=1.)
            sam_image_feats = self.extract_sam_feat(sam_images.tensor)
        if self.fp16:
            with autocast():
                clip_features = self.clip.encode_image_feature(clip_images.tensor.half())
                fpn_features = self.backbone(clip_features)
        else:
            clip_features = self.clip.encode_image_feature(clip_images.tensor.float())
            fpn_features = self.backbone(clip_features)

        if not self.training:
            with autocast():
                return self.inference(batched_inputs, 
                                    do_postprocess=self.do_postprocess,
                                    sam_feats=sam_image_feats, 
                                    clip_images=clip_images,
                                    clip_final_feats=clip_features['res5'],
                                    fpn_feats=fpn_features)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] 
        proposals, proposal_losses = self.proposal_generator(
            clip_images, fpn_features, gt_instances)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        del images
        _, detector_losses = self.roi_heads(sam=self.sam, 
                                            sam_features=sam_image_feats,
                                            attnpool=self.clip.visual.attnpool, 
                                            clip_final_feats=clip_features['res5'], 
                                            fpn_features=fpn_features, 
                                            proposals=proposals, 
                                            targets=gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
            
    
    def norm_imageList(self, images, mean, std, norm_val):
        resized_images = [(x.to(torch.float)/norm_val - mean) / std for x in images]
        return self.to_imageList(resized_images)
    
    def to_imageList(self, images: List[torch.Tensor]):
        """
        padding by size_divisibility, 1024 by default
        """
        images = ImageList.from_tensors(
            images,
            self.clip.visual.size_divisibility,
            padding_constraints=self.clip.visual.padding_constraints,
        )
        return images
    
    def postprocess(self, pred_instances, batched_inputs: List[Dict[str, torch.Tensor]], mask_threshold:float):
        """
        Rescale the output instances to the target size.
        image_sizes should be the origin size of images
        """
        processed_results = []
        for results_per_image, input_per_image in zip(
            pred_instances, batched_inputs
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = custom_detector_postprocess(results_per_image, height, width, mask_threshold=mask_threshold)
            if not self.eval_ar:
                processed_results.append({"instances": r})
            else: 
                r.proposal_boxes = r.pred_boxes
                processed_results.append({"proposals": r})
        return processed_results
    

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
    
    def visualize_training(self, batched_inputs, proposals, pg_name=''):

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
        
@torch.jit.unused
def custom_detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Inputs: 
        cut padding mask, and resize boxes and masks
        results: the pred_masks of (1024,1024), results.image_size: (1024, x) or (x,1024)
        output_height, output_width: the original img sie 
    """
    
    if isinstance(output_width, torch.Tensor):
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

    results = results[output_boxes.nonempty()]
    if results.has("pred_masks"):
        mask_tensor = results.pred_masks
        #2. clip up the paddings
        mask_tensor = mask_tensor[:, :, :input_size[0], :input_size[1]]
        #3. resize the box and mask, give it to the results
        mask_tensor = F.interpolate(mask_tensor, size=new_size, mode="bilinear", align_corners=False).squeeze(1)
        mask_tensor = (mask_tensor>=mask_threshold).to(torch.bool)
        results.pred_masks = mask_tensor
    output_boxes.scale(output_width_tmp/input_size[1], output_height_tmp/input_size[0])
    results.pred_boxes = output_boxes
    return results