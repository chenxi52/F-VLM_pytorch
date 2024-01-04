#!/bin/bash
#SBATCH --job-name=Fvlm
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --error=.output/slurm_out/runjob.%J.err
#SBATCH --output=.output/slurm_out/runjob.%J.out

srun python train_net_stand.py --config-file configs/Fvlm_tiny_coco.yaml --num-gpus 8 \
    OUTPUT_DIR output/clipRpn/FVLM/binaryCe \
    MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE True
    

