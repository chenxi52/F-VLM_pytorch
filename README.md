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
