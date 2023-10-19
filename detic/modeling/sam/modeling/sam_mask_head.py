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
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat
import time
from detectron2.structures.masks import polygons_to_bitmask
import copy

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
    
    def forward(
            self,
            roi_feature: List[torch.Tensor],
            img_features: torch.Tensor,
            instances: List[Instances],
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
        batch_size = roi_feature.shape[0]
        start_time = time.time()
        point_emd = self.point_emb(roi_feature) #prompt head 
        # print('dual_time1:', time.time()-start_time)
        point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        if self.with_sincos: 
            point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        #::2, 从 0 列开始+2 取列， 1::2, 从 1 列开始+2 取列
        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emd.shape[0], -1, *img_features.shape[-2:]
        )
        # the index must have 
        # print('dual_time2:', time.time()-start_time)

        img_flag_ids = torch.tensor([len(i) for i in instances], device=point_emd.device, dtype=torch.long)
        # print('img_flag_ids: ',img_flag_ids)
        # padding = torch.zeros((len(img_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        # padding: what if no_mask exist in the 
        # img_flag_ids = torch.cat([img_flag_ids, padding])
        
        img_embeddings = torch.repeat_interleave(img_features, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])
        # print('dual_time3:', time.time()-start_time)

        # print('img_pe device:',img_pe.device,'point_emd device:',point_emd.device, )
        low_res_masks = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emd,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
        )
        # print('dual_time2:', time.time()-start_time)

        ######################
        # Initialize the result storage lists
        # low_res_masks_list = []
        # iou_predictions_list = []

        # # Decide on the chunk size based on your requirements and memory constraints
        # chunk_size = 100

        # # Splitting the tensors into smaller chunks
        # point_emd_chunks = torch.split(point_emd, chunk_size, dim=0)
        # img_embeddings_chunks = torch.split(img_embeddings, chunk_size, dim=0)
        # img_pe_chunks = torch.split(img_pe, chunk_size, dim=0)
        # nomask_dense_embeddings_chunks = torch.split(nomask_dense_embeddings, chunk_size, dim=0)

        # # Iterate through each chunk
        # for point_emd_chunk, img_embeddings_chunk, img_pe_chunk, nomask_dense_chunk in zip(
        #     point_emd_chunks, img_embeddings_chunks, img_pe_chunks, nomask_dense_embeddings_chunks):

        #     # Processing each chunk through the mask decoder
        #     low_res_masks_chunk, iou_predictions_chunk = sam.mask_decoder.forward_batch(
        #         image_embeddings=img_embeddings_chunk,
        #         image_pe=img_pe_chunk,
        #         sparse_prompt_embeddings=point_emd_chunk,
        #         dense_prompt_embeddings=nomask_dense_chunk,
        #         multimask_output=False,
        #         res_img_feat=None  # As per your previous setup
        #     )

        #     # Append results from this chunk to the result storage lists
        #     low_res_masks_list.append(low_res_masks_chunk)
        #     iou_predictions_list.append(iou_predictions_chunk)

        # # Concatenate the results after processing all chunks
        # low_res_masks = torch.cat(low_res_masks_list, dim=0)
        # iou_predictions = torch.cat(iou_predictions_list, dim=0)
        ######################
        low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
        # iou_predictions = iou_predictions.squeeze(1)
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
        if self.training:
            # TODO: not right
            loss ={"loss_mask": mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight}
            # print('dual_time3:', time.time()-start_time)
            
            return loss
        else:
            mask_rcnn_inference(low_res_masks, instances)
            return instances



def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
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
    start_ = time.time()
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    # mask_side_len = pred_mask_logits.size(2)
    mask_side_len = 1024
    # assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    # print('loss_dual_time1:', time.time()-start_)
    
    gt_classes = []
    gt_masks = []
    # import ipdb;ipdb.set_trace()
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        # gt_mask resized to mask_size
        # gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
        #     instances_per_image.proposal_boxes.tensor, mask_side_len
        # ).to(device=pred_mask_logits.device)
        device = instances_per_image.proposal_boxes.device
        # boxes = instances_per_image.proposal_boxes.tensor.to(torch.device('cpu'))
        gt_masks_per_image = [torch.from_numpy(polygons_to_bitmask(copy.deepcopy(polygons), mask_side_len, mask_side_len))
                              for i, polygons in enumerate(instances_per_image.gt_masks.polygons)]
        if len(gt_masks_per_image) == 0:
            gt_masks_per_image = torch.empty(0, mask_side_len, mask_side_len, device=device, dtype=torch.bool)
        else:
            gt_masks_per_image = torch.stack(gt_masks_per_image, dim=0).to(device=device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)
    # print('loss_dual_time0:', time.time()-start_)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    # print('loss_dual_time1:', time.time()-start_)

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
    # print('loss_dual_time2:', time.time()-start_)

    # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)
    # print('loss_dual_time3:', time.time()-start_)

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
    # print('loss_dual_time5:', time.time()-start_)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")

    return mask_loss
