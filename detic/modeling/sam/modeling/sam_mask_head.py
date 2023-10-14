import torch
import torch.nn as nn
from typing import Tuple, List
from detectron2.modeling import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, ImageList
import torch.nn.functional as F
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from timm.models.layers import trunc_normal_


from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage


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
        # Prompt encoder
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

        self._init_weights(self.point_emb)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
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
    
    


    def select_points_from_box(self, box, ratio=0.25):
        x_min, y_min, x_max, y_max = box
    
        # Calculate the center of the box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate the width and height of the box
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate the upper-left and bottom-right points using the ratio
        upper_left_x = center_x - ratio * width
        upper_left_y = center_y - ratio * height
        bottom_right_x = center_x + ratio * width
        bottom_right_y = center_y + ratio * height
        
        # Create a tensor of shape (3, 2)
        points_tensor = torch.tensor([[center_x, center_y], 
                                    [upper_left_x, upper_left_y], 
                                    [bottom_right_x, bottom_right_y]])
        return points_tensor.cuda()
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
        ######################
        #1. get proposal_boxes from instances
        #2. pass it to prompt_encoder to get sparse_embeddings
        #3. pass sparse_embeddings to mask_decoder to get mask
        # import ipdb; ipdb.set_trace()
        if self.training:
            proposal_boxes = [x.proposal_boxes.tensor for x in instances]
        else:
            default_box = torch.tensor([0, 0, 1024, 1024])

            proposal_boxes = []
            for x in instances:
                if x.pred_boxes.tensor.shape[0] == 0:  # Check if the first dimension is 0
                    proposal_boxes.append(default_box)
                else:
                    proposal_boxes.append(x.pred_boxes.tensor)
            # proposal_boxes = [x.pred_boxes.tensor for x in instances]
        # Extract points for all boxes
        proposal_points = []
        for boxes_tensor in proposal_boxes:
            points_tensors_for_image = torch.stack([self.select_points_from_box(box) for box in boxes_tensor])
            proposal_points.append(points_tensors_for_image)
        low_res_masks = []
        
        for points_record, curr_embedding in zip(proposal_points, img_features):
            point_labels = torch.ones((points_record.shape[0], points_record.shape[1]), dtype=torch.int64)
            points = (points_record, point_labels)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            low_res_masks_curr, iou_predictions_curr = sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            # Find the channel index with the maximum IOU score for each instance
            max_iou_indices = iou_predictions_curr.argmax(dim=1)

            # Use advanced indexing to select the appropriate channel for each instance
            selected_masks = low_res_masks_curr[torch.arange(low_res_masks_curr.shape[0]), max_iou_indices]
            
            low_res_masks.append(selected_masks)

        # Concatenate along dimension 0
        low_res_masks = torch.cat(low_res_masks, dim=0)
        low_res_masks = low_res_masks.unsqueeze(1)
        # import ipdb; ipdb.set_trace()
        ######################
        ######################
        #ori version
        # first, select positive rois
        # batch_size = roi_feature.shape[0]
        # point_emd = self.point_emb(roi_feature) #prompt head 
        
        # point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        # if self.with_sincos: 
        #     point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        # #::2, 从 0 列开始+2 取列， 1::2, 从 1 列开始+2 取列
        # nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        #     point_emd.shape[0], -1, *img_features.shape[-2:]
        # )

        # # the index must have 
        # img_flag_ids = torch.bincount(mask_roi_inds.long())
        # padding = torch.zeros((len(img_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        # # padding: what if no_mask exist in the 
        # img_flag_ids = torch.cat([img_flag_ids, padding])
        
        # img_embeddings = torch.repeat_interleave(img_features, img_flag_ids, dim=0)
        # img_pe = sam.prompt_encoder.get_dense_pe()
        # img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        # res_img_feat = None
        
        # low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
        #     image_embeddings=img_embeddings, #n
        #     image_pe=img_pe,
        #     sparse_prompt_embeddings=point_emd,
        #     dense_prompt_embeddings=nomask_dense_embeddings,
        #     multimask_output=False,
        #     res_img_feat=res_img_feat,
        # )
        # # max_iou_indices = iou_predictions_curr.argmax(dim=1)

        # #     # Use advanced indexing to select the appropriate channel for each instance
        # #     selected_masks = low_res_masks_curr[torch.arange(low_res_masks_curr.shape[0]), max_iou_indices]
        
        
        # iou_predictions = iou_predictions.squeeze(1)
        ######################
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
        
        if self.training:
            # TODO: not right
            # return {"loss_mask": self.mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight}
            return None
        else:
            mask_rcnn_inference(low_res_masks, instances)
            return instances
    
    @torch.jit.unused
    def mask_rcnn_loss(self, pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)
            ###clone instances_per_image.proposal_boxes.tensor
            clone_proposal_boxes = instances_per_image.proposal_boxes.tensor.clone()
            clone_proposal_boxes = clone_proposal_boxes.detach().cpu()
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                clone_proposal_boxes, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
        storage.put_scalar("mask_rcnn/false_positive", false_positive)
        storage.put_scalar("mask_rcnn/false_negative", false_negative)
        if vis_period > 0 and storage.iter % vis_period == 0:
            pred_masks = pred_mask_logits.sigmoid()
            vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
            name = "Left: mask prediction;   Right: mask GT"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask)

        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        return mask_loss