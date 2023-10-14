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
import ipdb
import matplotlib.pyplot as plt
import os
import time

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
        #########################
        # import ipdb; ipdb.set_trace()
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        # boxes_prompt = gt_instances[0].get("gt_boxes")
        # #sam 
        # sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
        #         points=None,
        #         boxes=boxes_prompt.tensor,
        #         masks=None,
        #     )
        # low_res_masks, iou_predictions = self.sam.mask_decoder(
        #     image_embeddings=img_embedding_feat,
        #     image_pe=self.sam.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=True,
        # )
        # masks = F.interpolate(
        #     low_res_masks,
        #     (self.sam.image_encoder.img_size, self.sam.image_encoder.img_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # masks = masks > self.sam.mask_threshold
        # # Assuming img_tensor is your normalized image tensor with shape [3, 1024, 1024]
        # # Define a directory to save the images and masks
        # save_dir = "visualized_results"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # # Convert normalized tensor to range [0, 1] for image visualization
        # img_np = images[0].cpu().numpy().transpose(1, 2, 0)
        # img_np = img_np.clip(0, 1)
        # # Save the image
        # img_filename = os.path.join(save_dir, "image.png")
        # # import ipdb; ipdb.set_trace()
        # # plt.imsave(img_filename,  img_np)
        # plt.figure(figsize=(10,10))
        # plt.imshow(img_np)
        # plt.axis('on')
        # plt.show()
        # plt.close()


        # fig, ax = plt.subplots(figsize=(10,10))
        # ax.imshow(img_np)

        # # Now, save each of the 20 masks
        # for idx in range(masks.shape[0]):
        #     # Extract mask
        #     mask_np = masks[idx].cpu().numpy().transpose(1, 2, 0)
            
        #     # Normalize mask to range [0, 1] 
        #     # mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-5)  # added small epsilon to avoid division by zero
            
        #     # Ensure mask is in range [0, 1]
        #     mask_np = mask_np.clip(0, 1)
        #     # Convert to float
        #     mask_np = mask_np.astype(float)
        #     ax.imshow(mask_np)
        #     # Save the mask
        #     # mask_filename = os.path.join(save_dir, f"mask_{idx+1}.png")
        #     # plt.imsave(mask_filename, mask_np)
        # ax.axis('off')
        # plt.show()
        # # Saving the image with points before mask
        # fig.savefig('result.png')
        # plt.close(fig)

        # save_image(images.tensor[0], 'output_image.png') # nrow determines the number of images in a row
        # #########################

        # sssss
        fpn_features = self.backbone(inter_feats)
        # proposal_generator need to be trained before testing
        proposals, _ = self.proposal_generator(images, fpn_features, None) #samFpn # proposals: img_height=img_width=1024

        results, _ = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals)
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        ###########################
        # Replacing content for each pair of gt_instance and result
        # Assuming results and gt_instances are lists of Instances and have the same length
        

        # This will give you a list of results in the desired format
        ###########################
        # batched_inputs have ori_image_sizes
        # images.image_sizes have input_image_sizes
        # img_input_sizes = [(inp['input_height'], inp['input_width']) for inp in batched_inputs]
        # ori_sizes = [(inp['height'], inp['width']) for inp in batched_inputs]
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            # return self._postprocess(
            #     instances=results, ori_sizes=ori_sizes, image_sizes=img_input_sizes)
            return self._postprocess(instances=results, batched_inputs=batched_inputs, mask_threshold=self.mask_thr_binary)
        else:
            return results
        
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        startTime = time.time()
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=self.do_postprocess)
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #instance have img_Size with longest-size = 1024
        
        img_embedding_feat, inter_feats = self.extract_feat(images.tensor) 
        fpn_features = self.backbone(inter_feats)
        # fpn_features: Dict{'feat0': Tuple[2*Tensor[256,32,32]], 'feat1': Tuple[2*Tensor[256,64,64]], ...}
        # resize the img_size in gt_instances to (1024,1024)
        proposals, proposal_losses = self.proposal_generator(
            images, fpn_features, gt_instances)
        # proposals:max(h,w)=1024,  gt_instance:max(h,w)=1024
        # proposals: List[bz * Instance[1000 * Instances(num_instances, image_height, image_width, fields=[proposal_boxes: Boxes(tensor([1,4])), objectness_logits:tensor[1],])]]
        del images
        _, detector_losses = self.roi_heads(self.sam, img_embedding_feat, fpn_features, proposals, gt_instances)
        end_time = time.time()
        dua_time = end_time-startTime
        print('dua_time:',dua_time)
        
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
    
    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], mask_threshold:float):
        """
        Rescale the output instances to the target size.
        image_sizes should be the origin size of images
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = detector_postprocess(results_per_image, height, width, mask_threshold=mask_threshold)
            processed_results.append({"instances": r})
        return processed_results
