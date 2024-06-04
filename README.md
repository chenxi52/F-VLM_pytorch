
# F-VLM
运行配置基本都在config-file

## dataset 以及 环境 准备
见 Reference

## 运行命令：
CUDA_VISIBLE_DEVICES=4,5,6,7  python train_net_stand.py --config-file configs/Fvlm_coco.yaml --num-gpus 4 (--eval-only)


## Reference
### [Detecting Twenty-thousand Classes using Image-level Supervision](https://github.com/facebookresearch/Detic)

##  Model
Pytorch version of [F-VLM](http://arxiv.org/abs/2209.15639) (ICLR2023)

| Model | link | APr| AP|
|-------|-------|-------|----|
| F-VLM  |  [model](https://github.com/google-research/google-research/blob/master/fvlm/README.md)  | 28.0|39.6|
| Reproduce(RN50-coco)  |  [model](https://pan.baidu.com/s/16YWBCCkYuhBt148rGp9MkA?pwd=dwi9)  | 26.3|41.0|
