#!/bin/bash
#SBATCH --job-name=my_multi_task
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --error=.output/slurm_out/job.%J.err
#SBATCH --output=.output/slurm_out/job.%J.out

srun python train_net_stand.py --config-file configs/OpenDet_tiny_coco.yaml --num-gpus 8 \
    OUTPUT_DIR output/clipRpn/SamOn/fpn_lean_pe/ab_points PER_QUERY_POINT 6