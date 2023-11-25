# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor.yaml  #--resume 

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor_eval.yaml  --eval-only  #--resume 

# python train_net.py --num-gpus 4 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 8
# CUDA_VISIBLE_DEVICES=1,2,3,4,7 python train_net_stand.py --num-gpus 5 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 10
# python train_net_stand.py --num-gpus 8 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 16
# CUDA_VISIBLE_DEVICES=5,6 python3 train_net_stand.py --num-gpus 2 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 4 
# CUDA_VISIBLE_DEVICES=6 python3 train_net_stand.py --num-gpus 1 --config-file configs/RSPrompter_anchor_tiny_Vitdet_cosWarm.yaml SOLVER.IMS_PER_BATCH 2 
# CUDA_VISIBLE_DEVICES=6 python3 train_net_stand.py  --num-gpus 1 --config-file configs/RSPrompter_anchor_tiny_Vitdet_cosWarm.yaml SOLVER.IMS_PER_BATCH 2 
# CUDA_VISIBLE_DEVICES=3,4,5,6 python3 train_net_stand.py --num-gpus 4 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml --eval-only SOLVER.IMS_PER_BATCH 8 MODEL.WEIGHTS ./output/full_tune_tiny_coswarm/model_final.pth
# CUDA_VISIBLE_DEVICES=0 python3 plain_train_net.py --num-gpus 1 --config-file configs/Base-RCNN-FPN.yaml SOLVER.IMS_PER_BATCH 2 

# CUDA_VISIBLE_DEVICES=3,4,5,6 python3 train_net_stand.py --num-gpus 4 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml --eval-only  MODEL.WEIGHTS output/full_tune_tiny_coswarm/model_0089999.pth

python3 train_net_stand.py --num-gpus 1 \
        --config-file configs/OpenDet_anchor_tiny_Vitdet_cosWarm_coco.yaml \
        OUTPUT_DIR ./output/ovcoco_1124 \
        