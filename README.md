<!-- Stage1: 
    train RPN in base classes
    (test rpn recall first)
Stage2:
    change contextformer -->

1. 将 rpn 移动到 clip image encoder
    1. fpn 构建到 clip resnet
    2. (先复现f-vlm) box head, mask head都是 mask-rcnn一样的
2. 不使用 mask roi 而是直接将 box送入 sam prompt encoder --> sam mask decoder
    2. 目的是，roi 是否可以适应到原来的位置空间，又是否可以和 sam mask 对齐。
    3. 优化的参数是 rpn box_head context_former
    4. 由原来的 box roi经过卷积的分类，变为 mask token和原clip image feature交互后的特征 和 text_features对齐。 
![Alt text](image.png)

# Open-Vocabulary segmentation
运行配置基本都在config-file

## dataset 以及 环境 准备
见 Reference

## 运行命令：
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net.py --config-file configs/RSPrompter_anchor.yaml --num-gpus 4 (--eval-only)

可视化：
CUDA_VISIBLE_DEVICES=6 python tools/visualize_json_results.py --input output/Detic-COCO/RSPrompter_anchor/inference_coco_2017_val/coco_instances_results.json --output output/visualize/

CUDA_VISIBLE_DEVICES=6 python visualize_gt.py --config-file configs/RSPrompter_anchor.yaml --eval-only
## Reference
### [Detecting Twenty-thousand Classes using Image-level Supervision](https://github.com/facebookresearch/Detic)
