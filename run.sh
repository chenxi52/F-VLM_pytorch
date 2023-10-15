# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor.yaml  #--resume 

# python3 train_net.py --num-gpus 8 \
#     --config-file configs/RSPrompter_anchor_eval.yaml  --eval-only  #--resume 

python train_net.py --num-gpus 8 --config-file configs/RSPrompter_anchor_tiny_Vitdet.yaml SOLVER.IMS_PER_BATCH 16