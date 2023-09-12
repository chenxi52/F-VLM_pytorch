import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
from detectron2.modeling import build_mask_head, BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, ImageList
import torch.nn.functional as F

@ROI_MASK_HEAD_REGISTRY.register()
class samPromptMaskHead(nn.Module):
    @configurable
    def __init__(
            self,
            class_agnostic: bool=True,
            per_query_point: int=5,
            with_sincos: bool=True,
            train_size: int=1024,
            mask_loss: str='ce',
            mask_loss_weight: float=1.0,
            num_classes: int = 1
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
        if mask_loss == 'ce':
            self.mask_loss = CrossEntropyLoss(use_mask=True, loss_weight=mask_loss_weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        with_sincos = cfg.MODEL.ROI_MASK_HEAD.WITH_SINCOS
        per_query_point = cfg.MODEL.ROI_MASK_HEAD.PER_QUERY_POINT
        class_agnostic = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        mask_loss = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_TYPE
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
                'mask_loss': mask_loss,
                'mask_loss_weight': mask_loss_weight,
                'num_classes': num_classes}
    
    def forward(
            self,
            roi_feature: torch.Tensor,
            features: torch.Tensor,
            instances: List[Instances],
            sam: nn.Module,
            img_flag_freq: List[int],
            origin_img_size: List[Tuple[int, int]]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        firstly, inference, and then calculate losses
        Args:
            roi_feature: features after maskroi
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
        point_emd = self.point_emb(roi_feature) #prompt head 
        point_emd = point_emd.view(batch_size, self.per_query_point, -1)
        if self.with_sincos: 
            point_emd = torch.sin(point_emd[..., ::2] + point_emd[..., 1::2])
        
        nomask_dense_embeddings = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_emd.shape[0], -1, *features.shape[-2:]
        )
        img_flag_freq = torch.tensor(img_flag_freq, dtype=torch.long,device=point_emd.device)
        img_embeddings = torch.repeat_interleave(features, img_flag_freq, dim=0 )
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
        mask_preds = low_res_masks.squeeze(1)
        iou_predictions = iou_predictions.squeeze(1)
        mask_result = dict(mask_preds = mask_preds, mask_iou = iou_predictions)
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
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
            results_instances = []
            start = end = 0
            for freq, ins in zip(img_flag_freq, instances):
                end += freq.cpu().numpy()
                ins.pred_masks = mask_preds[start: end]
                ins.pred_ious = iou_predictions[start: end]
                start = end
                results_instances.append({'instances':ins})
            # then check the detector.inference.
            return results_instances
            

    def loss_and_target(
            self,
            mask_preds:torch.Tensor,
            instances: List[Instances],
            rcnn_train_cfg: int
            ):
        """
        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.        
        """
        mask_targets, mask_classes = self.get_targets(
            instances=instances,
            rcnn_train_cfg=rcnn_train_cfg)
        
        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else: 
            if self.class_agnostic:
                loss_mask = self.mask_loss(mask_preds, mask_targets,
                                           torch.zeros_like(mask_classes))
            else:
                loss_mask = self.mask_loss(mask_preds, mask_targets, 
                                           mask_classes)
        loss['loss_mask'] = loss_mask
        loss['mask_target'] = mask_targets
        return loss

    def get_targets(self, instances, rcnn_train_cfg):
        """
        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
        """

        pos_assigned_gt_inds = [res.gt_classes for res in instances]
        gt_masks = [res.gt_masks for res in instances]
        device = pos_assigned_gt_inds[0].device

        mask_targets_list = []
        mask_class_list = []
        mask_size = (rcnn_train_cfg, ) * 2
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0, ) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask.tensor.to(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
            mask_class_list.append(pos_gt_inds)
        
        
        mask_targets  = torch.cat(mask_targets_list) # [n, 1024, 1024]
        mask_classes  = torch.cat(mask_class_list) # [n]
        return mask_targets, mask_classes
    

def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, H, W), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]



class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0,
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = None
        self.avg_non_ignore = avg_non_ignore

        if self.use_mask:
            self.cls_criterion = mask_cross_entropy


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls
