import torch
import torch.nn as nn
from typing import Tuple, List
from detectron2.modeling import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from einops import repeat
from detectron2.structures import Instances, ImageList
import torch.nn.functional as F
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference
from timm.models.layers import trunc_normal_

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

        # low_res_masks, iou_predictions = sam.mask_decoder.forward_batch(
        #     image_embeddings=img_embeddings,
        #     image_pe=img_pe,
        #     sparse_prompt_embeddings=point_emd,
        #     dense_prompt_embeddings=nomask_dense_embeddings,
        #     multimask_output=False,
        #     res_img_feat=res_img_feat,
        # )
        ######################
        # Initialize the result storage lists
        low_res_masks_list = []
        iou_predictions_list = []

        # Decide on the chunk size based on your requirements and memory constraints
        chunk_size = 100

        # Splitting the tensors into smaller chunks
        point_emd_chunks = torch.split(point_emd, chunk_size, dim=0)
        img_embeddings_chunks = torch.split(img_embeddings, chunk_size, dim=0)
        img_pe_chunks = torch.split(img_pe, chunk_size, dim=0)
        nomask_dense_embeddings_chunks = torch.split(nomask_dense_embeddings, chunk_size, dim=0)

        # Iterate through each chunk
        for point_emd_chunk, img_embeddings_chunk, img_pe_chunk, nomask_dense_chunk in zip(
            point_emd_chunks, img_embeddings_chunks, img_pe_chunks, nomask_dense_embeddings_chunks):

            # Processing each chunk through the mask decoder
            low_res_masks_chunk, iou_predictions_chunk = sam.mask_decoder.forward_batch(
                image_embeddings=img_embeddings_chunk,
                image_pe=img_pe_chunk,
                sparse_prompt_embeddings=point_emd_chunk,
                dense_prompt_embeddings=nomask_dense_chunk,
                multimask_output=False,
                res_img_feat=None  # As per your previous setup
            )

            # Append results from this chunk to the result storage lists
            low_res_masks_list.append(low_res_masks_chunk)
            iou_predictions_list.append(iou_predictions_chunk)

        # Concatenate the results after processing all chunks
        low_res_masks = torch.cat(low_res_masks_list, dim=0)
        iou_predictions = torch.cat(iou_predictions_list, dim=0)
        ######################
        iou_predictions = iou_predictions.squeeze(1)
        # sample pos_ind from box_features, this has been done in the roi's _forward_mask
        if self.training:
            # TODO: not right
            return {"loss_mask": mask_rcnn_loss(low_res_masks, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(low_res_masks, instances)
            return instances