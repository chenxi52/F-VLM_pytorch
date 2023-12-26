import torch
import torch.nn as nn
from typing import Tuple, List
from detectron2.modeling import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, Boxes
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat, batched_nms
from detectron2.layers.wrappers import move_device_like
from detic.modeling.ContextFormer import build_contextformer, build_yhs_contextFormer
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.layers import nonzero_tuple
from fvcore.nn import sigmoid_focal_loss_jit
from detic.modeling.utils import load_class_freq, get_fed_loss_inds
import fvcore.nn.weight_init as weight_init
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
from torch.cuda.amp import autocast
import pickle
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import cross_entropy
from detic.data.datasets.coco_zeroshot import get_contigous_ids
@ROI_MASK_HEAD_REGISTRY.register()
class samMaskHead(BaseMaskRCNNHead):
    @configurable
    def __init__(
            self,
            vis_period: int = 0,
            with_sincos: bool = False,
            per_query_point: int = 4,
            clip_type: str = 'ViT-B/16',
            ignore_zero_cats: bool = False,
            text_feats: torch.Tensor = None,
            train_size: int = 224,
            add_pe_context: bool = False,
            **kwargs
            ) -> None:
        super().__init__(vis_period=vis_period)
        for name, value in locals().items():
            if name == 'self':
                continue
            elif name == 'kwargs':
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, name, value)

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
        self.contextformer = build_yhs_contextFormer(
            mask_dim=256,
            d_model=self.text_dim, 
            normalize_before=True,
            vis_dim=self.emb_dim,
        )
        def init_weights(m):
            if type(m) == nn.Linear:
                weight_init.c2_xavier_fill(m)
            elif type(m) == nn.Conv2d:
                weight_init.c2_msra_fill(m)
        self.point_emb.apply(init_weights)
        self.contextformer.apply(init_weights)
        if ignore_zero_cats:
            base_ones = torch.zeros(len(get_contigous_ids('all'))) 
            base_ones[get_contigous_ids('seen')] = 1
            base_ones = torch.cat([base_ones, torch.ones(1)])
            self.register_buffer('base_ones', base_ones)
            unused_index = get_contigous_ids('unused') # [0-79]
            self.register_buffer('unused_index', torch.tensor(unused_index))
            novel_ones = torch.zeros(len(get_contigous_ids('all')))
            novel_ones[get_contigous_ids('unseen')] = 1
            novel_ones = torch.cat([novel_ones, torch.ones(1)])
            self.register_buffer('novel_ones', novel_ones)
        del self.text_feats
        self.register_buffer('text_feats', text_feats)
        if add_pe_context:
            self.contextformer_pe = nn.Parameter(torch.randn(1, (train_size//32)**2+1, self.contextformer.d_model), requires_grad=True)
        else:
            self.contextformer_pe = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        else:
            num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        with open(cfg.MODEL.CLIP_TEXT_FEATS_PATH,'rb') as f:
            text_feats = pickle.load(f)
        test_pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            scales=[1./32,],
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        )
        return {'class_agnostic': cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK,
                'per_query_point': cfg.MODEL.ROI_MASK_HEAD.PER_QUERY_POINT,
                'with_sincos': cfg.MODEL.ROI_MASK_HEAD.WITH_SINCOS,
                'train_size':  cfg.INPUT.TRAIN_SIZE,
                'num_classes': num_classes,
                'vis_period': cfg.VIS_PERIOD,
                'clip_type': cfg.MODEL.BACKBONE.CLIP_TYPE,
                'score_thresh': cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                'top_per_instance': cfg.TEST.DETECTIONS_PER_IMAGE,
                'test_nms_thresh': cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
                'mask_loss_type': cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_TYPE,
                'data_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
                'test_score_type': cfg.TEST.SCORE_TYPE,
                'test_geometric_fact': cfg.TEST.GEOMETRIC_FACT,
                'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
                'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
                'fed_loss_num_cat': cfg.MODEL.NUM_SAMPLE_CATS,
                'mask_thr_binary': cfg.TEST.MASK_THR_BINARY,
                'mask_loss_weight':cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT,
                'text_feats': text_feats, 
                'test_pooler': test_pooler,
                'base_alpha': cfg.MODEL.ROI_BOX_HEAD.BASE_ALPHA,
                'novel_beta': cfg.MODEL.ROI_BOX_HEAD.NOVEL_BETA,
                'background_weight': cfg.MODEL.ROI_BOX_HEAD.BACKGROUND_WEIGHT,
                'eval_ar': cfg.EVAL_AR,
                'box_prompter': cfg.MODEL.ROI_MASK_HEAD.BOX_PROMPTER,
                'add_pe_context': cfg.MODEL.ROI_MASK_HEAD.ADD_PE_CONTEXT,
                }
    
    def forward(
            self,
            roi_features: torch.Tensor,
            instances: List[Instances],
            sam: nn.Module,
            sam_features: torch.Tensor,
            clip_final_feats: torch.Tensor,
            boxes: List[Boxes],
            attnpool: nn.Module = None,
            select_fore_cls: bool = True
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        firstly, inference, and then calculate losses
        Args:
            roi_feature: features after maskroi, multi-level---> roi box
        Returns:
            A dict of losses in training. The predicted "instances" in inference(List[Dict['instances': Instances]]).
        """
        if not self.box_prompter:
            batch_size = roi_features.shape[0]
            point_emd = self.point_emb(roi_features) #prompt head 
            point_emd = point_emd.view(batch_size, self.per_query_point, -1)
            if self.with_sincos: 
                point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
            nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                point_emd.shape[0], -1, *sam_features.shape[-2:]
            )
        else:
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points= None,
                boxes = cat([b.tensor for b in boxes], dim=0),
                masks = None
            )
        img_flag_ids = torch.tensor([len(i) for i in instances], device=clip_final_feats.device, dtype=torch.long)
        padding = torch.zeros((len(sam_features)-len(img_flag_ids),), device=img_flag_ids.device, dtype=img_flag_ids.dtype)
        img_flag_ids = torch.cat([img_flag_ids, padding])
        
        sam_features = torch.repeat_interleave(sam_features, img_flag_ids, dim=0)
        batch_clip_final_feats = torch.repeat_interleave(clip_final_feats, img_flag_ids, dim=0)
        img_pe = sam.prompt_encoder.get_dense_pe()
        img_pe = repeat(img_pe, 'b c h w -> (b n) c h w', n=sam_features.shape[0])
        
        # select foreGround proposals first will save computation here.
        with autocast():
            # box和 image features对应关系是，
            if self.box_prompter:
                low_res_masks, mask_tokens = sam.mask_decoder.forward_batch(
                    image_embeddings=sam_features,
                    image_pe=img_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            else:
                low_res_masks, mask_tokens = sam.mask_decoder.forward_batch(
                    image_embeddings=sam_features,
                    image_pe=img_pe,
                    sparse_prompt_embeddings=point_emd,
                    dense_prompt_embeddings=nomask_dense_embeddings,
                    multimask_output=False,)
        
            logits_image = self.contextformer(mask_tokens, 
                                            batch_clip_final_feats, 
                                            self.text_feats, 
                                            pos=self.contextformer_pe[:,1:] if self.add_pe_context else None, 
                                            query_pos=self.contextformer_pe[:,:1] if self.add_pe_context else None)
                
            if len(logits_image.shape) > 2: #[bzs, n_tokens, dim]
                logits_image = logits_image.squeeze()
        low_res_masks = torch.nn.functional.interpolate(low_res_masks, size=(self.train_size, self.train_size), mode='bilinear', align_corners=False)
        if self.training:
            del boxes
            gt_classes = (cat([p.gt_classes for p in instances], dim=0) )
            assert len(logits_image.shape) == 2, print('the fore proposal is zero in this batch', logits_image.shape)
            weight = torch.cat([torch.ones(self.data_classes), torch.ones(1)* self.background_weight], dim=0).to(logits_image.device)
            loss_cls = cross_entropy(logits_image, gt_classes, reduction="mean", weight=weight)
            # 当选前景 proposals 进入 mask head即 self.fore_mask_cls=True，这里 cls_accuracy=fg_cls_accuracy
            _log_classification_stats(logits_image, gt_classes , 'fast_rcnn')
            # if select_fore_cls=True, custom_mask_rcnn_loss should select foreground masks
            loss ={"loss_mask": self.custom_mask_rcnn_loss(low_res_masks, instances, self.vis_period, select_fore_cls) * self.mask_loss_weight,
                   "loss_cls": loss_cls
                   }
            return loss
        else:
            # low_res_masks, logits_image(scores, class), vlm_scores, 
            new_instances = self.custom_mask_rcnn_inference(pred_mask_logits = low_res_masks, 
                                                            pred_instances=instances, 
                                                            logits_image=logits_image, 
                                                            boxes=boxes,
                                                            clip_features=clip_final_feats,
                                                            attnpool=attnpool)
            return new_instances
        
    # @torch.jit.unused
    # def sigmoid_focal_loss(self, inputs, targets, gt_classes, alpha: float = 0.25, gamma: float = 2):
    #     """Compute the sigmoid focal loss."""
    #     _log_classification_stats(inputs, gt_classes, 'clip_fast_rcnn')
    #     prob = inputs.sigmoid()
    #     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    #     p_t = prob * targets + (1 - prob) * (1 - targets)
    #     loss = ce_loss * ((1 - p_t) ** gamma)
    #     B = inputs.shape[0]
    #     C = inputs.shape[1] - 1
    #     weight = 1
    #     if alpha >= 0:
    #         loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    #     # if use fed_loss, the background is not sampled ?
    #     if self.use_fed_loss and (self.freq_weight is not None): # fedloss
    #         appeared = get_fed_loss_inds(
    #             gt_classes, 
    #             num_sample_cats=self.fed_loss_num_cat,
    #             C=C,
    #             weight=self.freq_weight)
    #         appeared_mask = appeared.new_zeros(C + 1)
    #         appeared_mask[appeared] = 1 # C + 1
    #         weight = appeared_mask.float() # 
    #     if self.ignore_zero_cats and (self.freq_weight is not None):
    #         w = (self.freq_weight.view(-1) > 1e-4).float()
    #         w = torch.cat([w, w.new_ones(1)])
    #         weight = weight * w
    #     return (loss*weight).mean(1).sum() / B
    
    @torch.jit.unused
    def custom_mask_rcnn_loss(self, pred_mask_logits: torch.Tensor, 
                              instances: List[Instances], 
                              vis_period: int = 0,
                              select_fore_cls: bool = False):
        """
        remove gt_masks.crop_and_resize from original mask_rcnn_loss 
        with foreground selection
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        gt_masks = []
        pred_mask_list = []
        pred_mask_per_logits = pred_mask_logits.split([len(x) for x in instances], dim=0)
        for instances_per_image, pred_logits_per_image in zip(instances, pred_mask_per_logits):
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            ######因为之前并没有选出前景的 propsals做分类， 这里应该选出前景的masks
            if not select_fore_cls: 
                fg_inds = nonzero_tuple((gt_classes_per_image >= 0) & (gt_classes_per_image < self.data_classes))[0]
                if len(fg_inds) == 0:
                    continue
                gt_masks_per_image = instances_per_image[fg_inds].gt_masks.tensor
                pred_mask_list.append(pred_logits_per_image[fg_inds])
                # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            else:
                gt_masks_per_image = instances_per_image.gt_masks.tensor
            gt_masks_per_image = F.pad(gt_masks_per_image, (0, self.train_size-gt_masks_per_image.shape[-1], 0, self.train_size-gt_masks_per_image.shape[-2]), value=0)
            gt_masks.append(gt_masks_per_image)
        ###########
        gt_masks = cat(gt_masks, dim=0)
        pred_mask_logits = cat(pred_mask_list, dim=0)
        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            assert NotImplementedError
            # assert len(gt_classes)>0, print('gt_classes is empty when cls_agnostic_mask = False')
            # indices = torch.arange(total_num_masks)
            # gt_classes = cat(gt_classes, dim=0)
            # pred_mask_logits = pred_mask_logits[indices, gt_classes]
        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and sigmoid(0.0) == 0.5 threshold)
        mask_incorrect = (pred_mask_logits > self.mask_thr_binary) != gt_masks_bool
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
            pred_masks_thre = pred_masks > self.mask_thr_binary
            vis_masks = torch.cat([pred_masks, pred_masks_thre, gt_masks], axis=2)
            name = "Left: mask prediction;   Middle: thre0.5 ;Right: mask GT"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name, vis_mask)
                break

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

    def custom_mask_rcnn_inference(self, 
                                pred_mask_logits: torch.Tensor, 
                                pred_instances: List[Instances], 
                                logits_image: torch.Tensor,
                                boxes: List[Boxes],
                                clip_features: torch.Tensor = None,
                                attnpool: nn.AdaptiveAvgPool2d = None,
                                ):
        """
        boxes to crop vlm features and get vlm_scores
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
        num_boxes_per_image = [len(i) for i in pred_instances]
        
        vlm_box_features = self.test_pooler([clip_features], boxes)
        # vlm pooler layer: clip attenpool
        vlm_box_features = attnpool(vlm_box_features)
        vlm_box_features = vlm_box_features / vlm_box_features.norm(dim=1, keepdim=True)
        logits_scale = 1/0.01
        vlm_scores = logits_scale * vlm_box_features @ (self.text_feats.t().to(vlm_box_features.device))

        vlm_scores = vlm_scores.split(num_boxes_per_image, dim=0)
        logits_image = logits_image.split(num_boxes_per_image, dim=0)
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
        results = [self.inference_single_image(mask_preds_per_img, 
                                           scores_ori=scores_per_img, 
                                           instances=instances_per_img,
                                           vlm_scores_ori=vlm_scores_per_img) 
                    for vlm_scores_per_img, scores_per_img, mask_preds_per_img, instances_per_img in 
                         zip(vlm_scores, logits_image, mask_probs_pred, pred_instances)]
        return results

    # classification score consider vlm text score.
    def inference_single_image(self, mask_pred, scores_ori, instances, vlm_scores_ori):
        vlm_scores = vlm_scores_ori.clone()
        scores = scores_ori.clone()
        scores[:, self.unused_index] = float('-inf')
        vlm_scores[:, self.unused_index] = float('-inf')
        # 先去掉背景类别再 softmax
        vlm_scores = F.softmax(vlm_scores, dim=1)
        scores = F.softmax(scores, dim=1)

        new_instance = Instances(instances.image_size).to(scores.device)
        boxes = instances.pred_boxes.tensor
        objectness = instances.objectness
        if self.test_score_type == 'ob_mul_cls':
            assert NotImplementedError
            ensembled_socres = scores * objectness[:, None]
        elif self.test_score_type == 'ob_geo_cls':
            ensembled_socres = scores**(1-self.test_geometric_fact) * objectness[:, None]**self.test_geometric_fact
            ensembled_socres = ensembled_socres / ensembled_socres.sum(dim=1, keepdim=True)
            ensembled_socres = ensembled_socres[:, :-1]
            assert ensembled_socres[:, self.unused_index].max() < 1e-5, 'unused classes should not be evaluated'
        elif self.test_score_type == 'cls':
            # with vlm scores
            base_score = ((scores * self.base_ones)**(1-self.base_alpha)) * ((vlm_scores*self.base_ones)**(self.base_alpha))
            novel_score = ((scores * self.novel_ones)**(1-self.novel_beta)) * ((vlm_scores * self.novel_ones)**(self.novel_beta))
            ensembled_socres = base_score + novel_score
            ensembled_socres = torch.cat([ensembled_socres[:,:-1], scores[:,-1:]], dim=1)
            ensembled_socres = ensembled_socres / ensembled_socres.sum(dim=1, keepdim=True)
            ensembled_socres = ensembled_socres[:, :-1]
            assert ensembled_socres[:, self.unused_index].max() < 1e-5, 'unused classes should not be evaluated'

        filter_mask = ensembled_socres > self.score_thresh
        num_bbox_reg_classes = boxes.shape[1] // 4
        filter_inds = filter_mask.nonzero()
        boxes = boxes.view(-1, num_bbox_reg_classes, 4)
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        ensembled_socres = ensembled_socres[filter_mask]
        keep = batched_nms(boxes, ensembled_socres, filter_inds[:, 1], self.test_nms_thresh)
        if self.top_per_instance >= 0:
            keep = keep[:self.top_per_instance]
        boxes, ensembled_socres, filter_inds = boxes[keep], ensembled_socres[keep], filter_inds[keep]

        new_instance.pred_boxes = Boxes(boxes)  # (1, Hmask, Wmask)
        new_instance.scores = ensembled_socres
        new_instance.pred_classes = filter_inds[:,1]
        new_instance.pred_masks = mask_pred[filter_inds[:,0]]
        if self.eval_ar:
            new_instance.objectness_logits = objectness[filter_inds[:,0]]
        return new_instance

@torch.jit.unused
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
@torch.jit.unused
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
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
@torch.jit.unused
def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
