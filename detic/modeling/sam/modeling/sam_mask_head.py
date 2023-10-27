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
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss

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
        # point_emb = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(7*7*256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 256*sincos*per_query_point)
        # )
        # self.point_emb = point_emb
        self.class_agnostic = class_agnostic
        self.per_query_point = per_query_point
        self.with_sincos = with_sincos
        self.train_size = train_size
        self.num_classes = num_classes
        self.vis_period = vis_period

        # self._init_weights(self.point_emb)

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
            roi_boxes,
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
        # import ipdb;ipdb.set_trace()
        sparse_embeddings, dense_embeddings = sam.prompt_encoder.forward(
            points = None,
            boxes = roi_boxes, 
            masks = None
        )
        # point_emd = self.point_emb(roi_feature) #prompt head 
        # point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        # if self.with_sincos: 
        #     point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        # #::2, 从 0 列开始+2 取列， 1::2, 从 1 列开始+2 取列
        # nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        #     point_emd.shape[0], -1, *img_features.shape[-2:]
        # )
        img_flag_ids = torch.tensor([len(i) for i in instances], device=sparse_embeddings.device, dtype=torch.long)
        padding = torch.zeros((len(img_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        # padding: what if no_mask exist in the 
        img_flag_ids = torch.cat([img_flag_ids, padding])
        
        img_embeddings = torch.repeat_interleave(img_features, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        low_res_masks = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

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
        # iou_predictions = iou_predictions.squeeze(1)
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
        # low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
        low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)

        if self.training:
            # TODO: not right
            loss ={"loss_mask": custom_mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight}
            return loss
        else:
            # low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
            mask_rcnn_inference(low_res_masks, instances)
            return instances
        

def custom_mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    remove gt_masks.crop_and_resize from original mask_rcnn_loss 
    """
    # start_ = time.time()
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    # mask_side_len = pred_mask_logits.size(2)
    # mask_side_len = 1024
    # assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    # print('loss_dual_time1:', time.time()-start_)
    
    gt_classes = []
    gt_masks = []
    # store the gt_mask to gpu first 
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        # the mask are sampled by box grid with repspect to the mask_size_len when the gt_maks=polygonMask
        # no need the crop_and_resize
        # if gt_mask is bitMask, the crop_and_resize have align_ratio=1, and resize the roi to mask_side_len, which is totally wrong!!!
        # instances.gt_masks.tensor = torch.nn.functional.pad(instances.gt_masks.tensor, (0, mask_side_len-instances.gt_masks.tensor.shape[-1], 0, mask_side_len-instances.gt_masks.tensor.shape[-2]))
        
        
        # gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
        #     instances_per_image.proposal_boxes.tensor, mask_side_len
        # ).to(device=pred_mask_logits.device)
        ########
        # device = instances_per_image.proposal_boxes.device

        # # here the gt_mask is bitmask
        # gt_masks_per_image = [torch.from_numpy(polygons_to_bitmask(copy.deepcopy(polygons), mask_side_len, mask_side_len))
        #                       for i, polygons in enumerate(instances_per_image.gt_masks.polygons)]
        # import ipdb;ipdb.set_trace()
        # gt_masks_per_image = torch.tensor(torch.ones(size=(len(instances_per_image.gt_masks.polygons), mask_side_len, mask_side_len)),device=device)
        #########
        # if len(gt_masks_per_image) == 0:
        #     gt_masks_per_image = torch.empty(0, mask_side_len, mask_side_len, device=device, dtype=torch.bool)
        # else:
        #     gt_masks_per_image = torch.stack(gt_masks_per_image, dim=0).to(device=device)
        
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks_per_image = instances_per_image.gt_masks.tensor
        gt_masks.append(gt_masks_per_image)

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