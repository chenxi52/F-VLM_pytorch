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
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss

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
            # height, width: input_image_sizes， 原本的 img_size
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            # results_per_image..image_size(): one is 1024
            # the mask have zeor padding but image_size indicts that not 
            r = custom_detector_postprocess(results_per_image, height, width, mask_threshold=mask_threshold)
            processed_results.append({"instances": r})
        return processed_results

def custom_detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
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

    # results = results[output_boxes.nonempty()]
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
        mask_tensor = mask_tensor[:,:, :results.image_size[1], :results.image_size[0]]

        #3. resize the box and mask, give it to the results
        mask_tensor = F.interpolate(mask_tensor, size=new_size, mode="bilinear", align_corners=False)
        mask_tensor = (mask_tensor>=mask_threshold).to(torch.bool)
        # roi_masks.tensor = mask_tensor
        # results.pred_masks = mask_tensor
    
    output_boxes.scale(output_width_tmp/results.image_size[1], output_height_tmp/results.image_size[0])
    # output_boxes.clip(results.image_size)
    return results