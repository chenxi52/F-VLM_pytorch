#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=.output/slurm_out/%j.out
#SBATCH --error=.output/slurm_out/%j.error
path="output/clipRpn/SamOn/boxPrompt/"
model_weight=$path"model_final.pth"

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.2 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.45 \
    OUTPUT_DIR $path

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.2 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR $path
    
srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS  $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.2 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR $path

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.45 \
    OUTPUT_DIR $path

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR $path
    
srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS  $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.3 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR $path

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.45 \
    OUTPUT_DIR $path

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only --num-gpus 8 \
    MODEL.WEIGHTS $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.65 \
    OUTPUT_DIR $path
    
srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --eval-only  --num-gpus 8\
    MODEL.WEIGHTS  $model_weight \
    MODEL.ROI_BOX_HEAD.BASE_ALPHA 0.4 \
    MODEL.ROI_BOX_HEAD.NOVEL_BETA 0.85 \
    OUTPUT_DIR $path