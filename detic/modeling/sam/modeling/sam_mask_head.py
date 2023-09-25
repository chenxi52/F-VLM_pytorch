import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
from detectron2.modeling import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, ImageList
import torch.nn.functional as F
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference

@ROI_MASK_HEAD_REGISTRY.register()
class samMaskHead(BaseMaskRCNNHead):
    @configurable
    def __init__(
            self,
            class_agnostic: bool=True,
            per_query_point: int=5,
            with_sincos: bool=True,
            train_size: int=1024,
            num_classes: int = 1,
            mask_loss_weight: float=1.0,
            vis_period: int = 0
            ) -> None:
        super().__init__()
        if with_sincos:
            sincos = 2
        else:
            sincos = 1
        point_emb = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(7*7*256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256*sincos*per_query_point)
        )
        self.point_emb = point_emb
        self.class_agnostic = class_agnostic
        self.per_query_point = per_query_point
        self.with_sincos = with_sincos
        self.train_size = train_size
        self.num_classes = num_classes
        self.vis_period = vis_period
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        with_sincos = cfg.MODEL.ROI_MASK_HEAD.WITH_SINCOS
        per_query_point = cfg.MODEL.ROI_MASK_HEAD.PER_QUERY_POINT
        class_agnostic = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        mask_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        else:
            num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        train_size = cfg.INPUT.TRAIN_SIZE
        return {'class_agnostic': class_agnostic,
                'per_query_point': per_query_point,
                'with_sincos': with_sincos,
                'train_size': train_size,
                'mask_loss_weight': mask_loss_weight,
                'num_classes': num_classes,
                'vis_period': cfg.VIS_PERIOD}
    
    def forward(
            self,
            roi_feature: List[torch.Tensor],
            img_features: torch.Tensor,
            instances: List[Instances],
            mask_roi_inds: torch.Tensor,
            sam: nn.Module,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        firstly, inference, and then calculate losses
        Args:
            roi_feature: features after maskroi, multi-level
            features: features from image encoder
            instances: Instances(num_instances=1, image_height=1024, image_width=664,
                fields=[proposal_boxes: Boxes(tensor([[214.0800, 907.2640, 235.6640, 963.2800]], device='cuda:0')), 
                objectness_logits: tensor([23.0259], device='cuda:0'), gt_classes: tensor([0], device='cuda:0'), 
                gt_boxes: Boxes(tensor([[214.0800, 907.2640, 235.6640, 963.2800]], device='cuda:0')), 
                gt_masks: PolygonMasks(num_instances=1)])
        Returns:
            A dict of losses in training. The predicted "instances" in inference(List[Dict['instances': Instances]]).
        """
        # first, select positive rois
        batch_size = roi_feature.shape[0]
        point_emd = self.point_emb(roi_feature) #prompt head 
        
        point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        if self.with_sincos: 
            point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        #::2, 从 0 列开始+2 取列， 1::2, 从 1 列开始+2 取列
        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emd.shape[0], -1, *img_features.shape[-2:]
        )

        # the index must have 
        img_flag_ids = torch.bincount(mask_roi_inds.long())
        padding = torch.zeros((len(img_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        # padding: what if no_mask exist in the 
        img_flag_ids = torch.cat([img_flag_ids, padding])
        
        img_embeddings = torch.repeat_interleave(img_features, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        res_img_feat = None
        low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emd,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
            res_img_feat=res_img_feat,
        )
        iou_predictions = iou_predictions.squeeze(1)
        mask_result = dict(mask_preds = low_res_masks.squeeze(1), mask_iou = iou_predictions)
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
        if self.training:
            return {"loss_mask": mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(low_res_masks, instances)
            return instances
        
        if self.training:
            # gt_mask id [1024,1024]
            # first pad the mask_result, then the loss
            mask_preds = F.interpolate(
                low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
            if mask_preds.size(0) == 0:
                mask_loss_and_target = dict(loss_mask = low_res_masks.sum(), mask_target=None)
            else:
                mask_loss_and_target = self.loss_and_target(
                    mask_preds = mask_preds,
                    instances = instances,
                    rcnn_train_cfg = self.train_size
                )
            mask_result.update(loss_mask = mask_loss_and_target['loss_mask'])
            return mask_result
        else:
            mask_preds = low_res_masks.squeeze(1)
            results_instances = []
            img_flag_ids = img_flag_ids.cpu().numpy().tolist()
            mask_preds = mask_preds.split(img_flag_ids, 0)
            iou_predictions = iou_predictions.split(img_flag_ids, 0)
            for i, ins in enumerate(instances):
                ins.pred_masks = mask_preds[i]
                ins.pred_ious = iou_predictions[i]
                results_instances.append(ins)
            # then check the detector.inference.
            return results_instances
            