#!/bin/bash
#SBATCH --job-name=my_multi_task
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.2 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta
    
srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.2 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.45 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.45 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR output/clipRpn/SamOn/eval_alpha_beta