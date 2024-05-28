
# Open-Vocabulary segmentation
运行配置基本都在config-file

## dataset 以及 环境 准备
见 Reference

## 运行命令：
CUDA_VISIBLE_DEVICES=4,5,6,7  python train_net_stand.py --config-file configs/Fvlm_coco.yaml --num-gpus 4 (--eval-only)


## Reference
### [Detecting Twenty-thousand Classes using Image-level Supervision](https://github.com/facebookresearch/Detic)
