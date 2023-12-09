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
from detectron2.layers import cat, cross_entropy, batched_nms
from detectron2.layers.wrappers import move_device_like
import torch.cuda.amp as amp
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detic.modeling.ContextFormer import build_contextformer
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.layers import nonzero_tuple
from fvcore.nn import sigmoid_focal_loss_jit
from detic.modeling.utils import load_class_freq, get_fed_loss_inds
import fvcore.nn.weight_init as weight_init

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
            mask_loss_type: str = 'ce',
            mask_loss_weight: float=1.0,
            vis_period: int = 0,
            clip_type: str = 'CLIP_400M_Large',
            score_thresh: float=0.02,
            top_per_instance: int=100,
            test_nms_thresh: float=0.5,
            data_classes: int=80,
            test_score_type: str='score',
            test_geometric_fact: float= 0.5,
            ignore_zero_cats: bool=True,
            cat_freq_path: str='',
            fed_loss_freq_weight: float=0.5,
            use_fed_loss: bool=False,
            fed_loss_num_cat: int=50,
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
            self.emb_dim = 768
            self.down_dim = self.emb_dim
        elif clip_type == 'RN50':
            self.text_dim = 1024
            self.emb_dim = 2048
            self.down_dim = self.emb_dim
        elif clip_type == 'RN50x64':
            self.text_dim = 1024
            self.emb_dim = 4096
            self.down_dim = self.emb_dim
        
        self.contextformer = build_contextformer(
            mask_dim=256,
            d_model=self.text_dim, clip_txt_dim=self.text_dim, normalize_before=False
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                weight_init.c2_xavier_fill(m)
            elif type(m) == nn.Conv2d:
                weight_init.c2_msra_fill(m)
        self.point_emb.apply(init_weights)
        self.score_thresh = score_thresh
        self.top_per_instance = top_per_instance
        self.test_nms_thresh = test_nms_thresh
        self.mask_loss_type = mask_loss_type
        self.data_classes  = data_classes
        self.test_score_type = test_score_type
        self.test_geometric_fact = test_geometric_fact
        self.ignore_zero_cats = ignore_zero_cats
        if self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        self.use_fed_loss = use_fed_loss
        self.fed_loss_num_cat = fed_loss_num_cat
            
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
                'num_classes': num_classes,
                'vis_period': cfg.VIS_PERIOD,
                'clip_type': clip_type,
                'score_thresh': cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                'top_per_instance': cfg.TEST.DETECTIONS_PER_IMAGE,
                'test_nms_thresh': cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
                'mask_loss_type': cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_TYPE,
                'data_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
                'test_score_type': cfg.TEST.SCORE_TYPE,
                'test_geometric_fact': cfg.TEST.GEOMETRIC_FACT,
                'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
                'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
                'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
                'fed_loss_num_cat': cfg.MODEL.NUM_SAMPLE_CATS,
                }
    
    def forward(
            self,
            roi_features: torch.Tensor,
            img_features: torch.Tensor,
            instances: List[Instances],
            sam: nn.Module,
            clip_images: torch.Tensor,
            clip_texts: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        firstly, inference, and then calculate losses
        Args:
            roi_feature: features after maskroi, multi-level---> roi box
            features: features from image encoder
            instances: 
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
        # select foreGround proposals first will save computation here.
        low_res_masks, iou_preds, mask_tokens = sam.mask_decoder.forward_batch(
            image_embeddings=img_embeddings,
            image_pe=img_pe,
            sparse_prompt_embeddings=point_emd,
            dense_prompt_embeddings=nomask_dense_embeddings,
            multimask_output=False,
        )
        with amp.autocast(enabled=True):
            clip_texts = move_device_like(clip_texts, low_res_masks)
            logits_image,_ = self.contextformer(mask_tokens, clip_img_embeddings, clip_texts)#mask_tokens: (batch_size, 1, self.emb_dim),clip: [bz,self.emb_dim, 32,32]
        low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
        if self.training:
            gt_classes = (
                cat([p.gt_classes for p in instances], dim=0) if len(instances) else torch.empty(0)
                )
            target_classes_onehot = torch.zeros(logits_image.shape, dtype=logits_image.dtype, device=logits_image.device)
            if len(logits_image.shape) < 2:
                logits_image = logits_image.unsqueeze(0)
            assert len(logits_image.shape) == 2, print('logits_image.shape: ', logits_image.shape)
            target_classes_onehot.scatter_(1, gt_classes.unsqueeze(-1), 1)
            # what if the classification not include background. The classification will not be interupted?
            loss ={"loss_mask": self.custom_mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight,
                   "loss_cls": self.sigmoid_focal_loss(logits_image, target_classes_onehot,gt_classes)}
            del instances, low_res_masks, logits_image, mask_tokens, clip_img_embeddings, img_embeddings
            return loss
        else:
            new_instances = self.custom_mask_rcnn_inference(low_res_masks, instances, logits_image[:,:-1], self.score_thresh, self.top_per_instance, self.test_nms_thresh)
            del instances, low_res_masks, logits_image, mask_tokens, clip_img_embeddings, img_embeddings
            return new_instances
        
    def sigmoid_focal_loss(self, inputs, targets, gt_classes, alpha: float = 0.25, gamma: float = 2):
        """Compute the sigmoid focal loss."""
        _log_classification_stats(inputs, gt_classes, 'clip_fast_rcnn')
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        B = inputs.shape[0]
        C = inputs.shape[1] - 1
        weight = 1
        if alpha >= 0:
            loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
        # if use fed_loss, the background is not sampled ?
        if self.use_fed_loss and (self.freq_weight is not None): # fedloss
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            weight = appeared_mask.float() # 
        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            w = torch.cat([w, w.new_ones(1)])
            weight = weight * w
        return (loss*weight).mean(1).sum() / B
    
    def get_logits(self, region_features, text_features, logit_scale):
        # 计算image_features @ text_features.T相似度矩阵
        # set unseen class weights to zero
        region_features = region_features / (region_features.norm(dim=-1, keepdim=True) + 1e-7)
        logits_per_image = logit_scale * region_features @ (text_features.unsqueeze(0).transpose(1, 2))
        logits_per_text = logit_scale * (text_features.unsqueeze(0)) @ region_features.transpose(1, 2)
        return logits_per_image, logits_per_text
    
    def custom_mask_rcnn_loss(self, pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
        """
        remove gt_masks.crop_and_resize from original mask_rcnn_loss 
        with foreground selection
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)

        gt_classes = []
        gt_masks = []
        fg_inds_list = []
        num_instance_list = []
        # store the gt_mask to gpu first 
        for instances_per_image in instances:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            # duplicated when select foreground proposals before, but not a big deal
            fg_inds = nonzero_tuple((gt_classes_per_image >= 0) & (gt_classes_per_image < self.data_classes))[0]

            gt_classes.append(gt_classes_per_image[fg_inds])
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks_per_image = instances_per_image.gt_masks.tensor[fg_inds]
            gt_masks.append(gt_masks_per_image)
            fg_inds_list.append(fg_inds)
            num_instance_list.append(len(instances_per_image))
        pred_mask_per_logits = pred_mask_logits.split(num_instance_list, dim=0)
        pred_mask_logits_list = [pred_mask_per_logits[i][fg_inds_list[i]] for i in range(len(fg_inds_list))]
        pred_mask_logits = torch.cat(pred_mask_logits_list, dim=0)
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

        if self.mask_loss_type == 'ce':
            mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        elif self.mask_loss_type == 'focal_dice':
            # suitable for open-vocabulary setting
            focalLoss = sigmoid_focal_loss_jit(pred_mask_logits, 
                                            gt_masks,
                                            alpha=0.25,
                                            gamma=2.0,
                                            reduction="mean")
            diceLoss = dice_loss(pred_mask_logits,
                                gt_masks)
            mask_loss = focalLoss + diceLoss
        elif self.mask_loss_type == 'ce_dice':
            ceLoss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
            diceLoss = dice_loss(pred_mask_logits,
                                gt_masks)
            mask_loss = ceLoss + diceLoss
        else:
            assert False, 'mask loss type not supported'
        return mask_loss

    def custom_mask_rcnn_inference(self, pred_mask_logits: torch.Tensor, pred_instances: List[Instances], logits_image: torch.Tensor,
                                score_thresh: float, top_per_instance: int = 100, nms_thresh: float = 0.5):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.
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
        return self.inference_single_image(mask_probs_pred, logits_image, pred_instances, score_thresh, top_per_instance, nms_thresh )


    def inference_single_image(self, mask_probs_pred, logits_image, pred_instances, score_thresh, top_per_instance, nms_thresh):
        # batch nms for single instance, class-wisely
        instance_list = []
        for prob, logits, instances in zip(mask_probs_pred, logits_image, pred_instances):
            new_instance = Instances(instances.image_size).to(logits.device)
            scores = logits.sigmoid()
            boxes = instances.pred_boxes.tensor
            objectness = instances.objectness
            if self.test_score_type == 'ob_mul_cls':
                scores = scores * objectness[:, None]
            elif self.test_score_type == 'ob_geo_cls':
                scores = scores**(1-self.test_geometric_fact) * objectness[:, None]**self.test_geometric_fact
            elif self.test_score_type == 'cls':
                pass
            masks = prob
            filter_mask = scores>score_thresh
            num_bbox_reg_classes = boxes.shape[1] // 4
            filter_inds = filter_mask.nonzero()
            boxes = boxes.view(-1, num_bbox_reg_classes, 4)
            if num_bbox_reg_classes == 1:
                boxes = boxes[filter_inds[:, 0], 0]
            else:
                boxes = boxes[filter_mask]
            scores = scores[filter_mask]
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
            if top_per_instance >= 0:
                keep = keep[:top_per_instance]
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

            new_instance.pred_boxes = Boxes(boxes)  # (1, Hmask, Wmask)
            new_instance.scores = scores
            new_instance.pred_classes = filter_inds[:,1]
            new_instance.pred_masks = masks[filter_inds[:,0]]
            instance_list.append(new_instance)
        return instance_list

def dice_loss(pred,
            target,
            weight=None,
            eps=1e-3,
            reduction='mean',
            avg_factor=None):
    """
    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    input = pred.sigmoid().flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
