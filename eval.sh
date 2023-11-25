
python -u train_net_stand.py --num-gpus 8 \
        --config-file configs/OpenDet_anchor_tiny_Vitdet_cosWarm_coco.yaml --eval-only \
        MODEL.WEIGHTS output/ovcoco/model_0089999.pth DATALOADER.NUM_WORKERS 4