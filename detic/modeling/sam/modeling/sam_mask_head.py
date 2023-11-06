import torch
import torch.nn as nn
from typing import Tuple, List
from detectron2.modeling import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, ImageList, Boxes
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat, cross_entropy
import time
from detectron2.layers.wrappers import move_device_like
from detectron2.structures.masks import polygons_to_bitmask
import torch.cuda.amp as amp
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
from detic.modeling.ContextFormer import build_contextformer
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
# add classification with clip text
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
            vis_period: int = 0,
            clip_type: str = 'CLIP_400M_Large',
            d_model: int=1024,
            score_thresh: float=0.02,
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
        
        if clip_type == 'ViT-B/16':
            self.text_dim = 512
            self.clip_dim = 768
        # elif clip_type == 'RN50':
        #     self.text_dim = 768
        #     self.clip_dim = 1024
        self.contextformer = build_contextformer(
          d_model=self.clip_dim
        )
        self.to_clip = nn.Linear(
            256, self.clip_dim
        )
        self.projector = nn.Linear(
            self.clip_dim, self.text_dim
        )
        self._init_weights(self.point_emb)
        self._init_weights(self.to_clip)
        self.score_thresh = score_thresh

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
        clip_type = cfg.MODEL.BACKBONE.CLIP_TYPE
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
                'vis_period': cfg.VIS_PERIOD,
                'clip_type': clip_type,
                'score_thresh': cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                }
    
    def forward(
            self,
            roi_features: torch.Tensor,
            img_features: torch.Tensor,
            instances: List[Instances],
            sam: nn.Module,
            clip: nn.Module,
            clip_images: torch.Tensor,
            clip_texts: torch.Tensor, 
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
            clip: clip model
            clip_images: vit-B/16: 512
            clip_texts: cit-B/16: 512
        Returns:
            A dict of losses in training. The predicted "instances" in inference(List[Dict['instances': Instances]]).
        """
        batch_size = roi_features.shape[0]
        point_emd = self.point_emb(roi_features) #prompt head 
        point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        if self.with_sincos: 
            point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        #::2, 从 0 列开始+2 取列， 1::2, 从 1 列开始+2 取列
        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emd.shape[0], -1, *img_features.shape[-2:]
        )
        img_flag_ids = torch.tensor([len(i) for i in instances], device=roi_features.device, dtype=torch.long)
        padding = torch.zeros((len(img_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        # padding: what if no_mask exist in the 
        img_flag_ids = torch.cat([img_flag_ids, padding])
        
        img_embeddings = torch.repeat_interleave(img_features, img_flag_ids, dim=0)
        clip_img_embeddings = torch.repeat_interleave(clip_images, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=img_embeddings.shape[0])

        low_res_masks, iou_preds, mask_tokens = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emd,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
        )
        # mask_tokens: (batch_size, 4, 256)
        # clip_texts and mask_tokens
        with amp.autocast(enabled=True):
            mask_tokens = self.to_clip(mask_tokens)
            logit_scale = clip.logit_scale.exp()
            semantic_token = self.contextformer(mask_tokens, clip_img_embeddings)#(batch_size, 4, self.clip_dim)
            semantic_token = self.projector(semantic_token)
        
            logits_image, logits_text = self.get_logits(semantic_token, clip_texts, logit_scale)
            logits_image = logits_image.squeeze(1) 
        low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
        if self.training:
            gt_classes = (
                cat([p.gt_classes for p in instances], dim=0) if len(instances) else torch.empty(0)
                )
            _log_classification_stats(logits_image, gt_classes)
            # TODO: not right
            loss ={"loss_mask": custom_mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight,
                   "loss_class": cross_entropy(logits_image, gt_classes, reduction='mean')}
            return loss

        else:
            # low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
            custom_mask_rcnn_inference(low_res_masks, instances, logits_image, self.score_thresh)
            return instances
        
    def get_logits(self, region_features, text_features, logit_scale):
        # 计算image_features @ text_features.T相似度矩阵
        region_features = region_features / (region_features.norm(dim=-1, keepdim=True) + 1e-7)
        logits_per_image = logit_scale * region_features @ (text_features.unsqueeze(0).transpose(1, 2))
        logits_per_text = logit_scale * text_features.unsqueeze(0) @ region_features.transpose(1, 2)
        return logits_per_image, logits_per_text
  

def custom_mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances], logits_image: torch.Tensor,
                               score_thresh: torch.Tensor):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        device = (
            class_pred.device
            if torch.jit.is_scripting()
            else ("cpu" if torch.jit.is_tracing() else class_pred.device)
        )
        indices = move_device_like(torch.arange(num_masks, device=device), class_pred)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    logits_image = logits_image.split(num_boxes_per_image, dim=0)
    return inference_single_image(mask_probs_pred, logits_image, pred_instances,score_thresh )

def inference_single_image(mask_probs_pred, logits_image, pred_instances, score_thresh):
    # batch nms for single instance
    instance_list = []
    
    for prob, logits, instances in zip(mask_probs_pred, logits_image, pred_instances):
        new_instance = Instances(instances.image_size)
        scores = F.softmax(logits, dim=-1)
        boxes = instances.pred_boxes
        pred_class = logits.argmax(dim=1)
        masks = prob
        filter_mask = scores>score_thresh
    
        boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        masks = masks[filter_mask]
        pred_class = pred_class[filter_mask]
        new_instance.pred_boxes = Boxes(boxes)  # (1, Hmask, Wmask)
        new_instance.scores = scores
        new_instance.pred_classes = pred_class
        new_instance.pred_masks = masks
    instance_list.append(new_instance)

    return instance_list

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