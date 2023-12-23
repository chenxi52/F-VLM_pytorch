#!/bin/bash
#SBATCH --job-name=evalAR
#SBATCH --partition=vip1
#SBATCH --output=.output/slurm_out/%j.out
#SBATCH --error=.output/slurm_out/%j.error
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco_eval.yaml --num-gpus 8 --eval-only \
    OUTPUT_DIR output/clipRpn/SamOn/evalAR \
    EVAL_AR True \
    MODEL.WEIGHTS output/clipRpn/SamOn/model_final.pth