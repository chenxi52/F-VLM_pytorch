# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor.yaml  #--resume 

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor_eval.yaml  --eval-only  #--resume 

# python train_net.py --num-gpus 4 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 8
# CUDA_VISIBLE_DEVICES=1,2,3,4,7 python train_net_stand.py --num-gpus 5 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 10
python train_net_stand.py --num-gpus 8 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 16
# python3 train_net_stand.py --num-gpus 4 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 8
python3 train_net_stand.py --num-gpus 1 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 2 
python3 train_net_stand.py --num-gpus 1 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml --eval-only SOLVER.IMS_PER_BATCH 2 MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.01 
python3 plain_train_net.py --num-gpus 1 --config-file configs/Base-RCNN-FPN.yaml SOLVER.IMS_PER_BATCH 2 