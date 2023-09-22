
# visualize mask, image, anchors_on_images
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
import re
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
from contextlib import ExitStack, contextmanager
import torch.nn as nn
import datetime
from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detic.custom_checkpointer import samCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler
from detectron2.structures import Instances, Boxes, BoxMode
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper, SamDatasetMapper
from detic.custom_solver import build_custom_optimizer, build_sam_optimizer
from detic.evaluation.oideval import OIDEvaluator
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator
from detic.modeling.utils import reset_cls_test
from detic.config import add_rsprompter_config
logger = logging.getLogger("Visulizer")
import ipdb
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation.evaluator import inference_context
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.build import custom_build_detection_test_loader

def create_instances(predictions, image_size):
    # for train_dataset
    ret = Instances(image_size)
    ret.pred_boxes = predictions.gt_boxes
    ret.pred_classes = predictions.gt_classes
    ret.pred_masks = predictions.gt_masks.tensor[:, :image_size[0], :image_size[1]]
    return ret


def create_pred_instances(predictions, image_size, metadata):
    ret = Instances(image_size)
    score = predictions.scores.cpu().numpy()
    bbox = predictions.pred_boxes.tensor.cpu().numpy()
    # bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    # labels = [metadata.thing_dataset_id_to_contiguous_id[predictions.pred_classes.cpu().numpy()[i]] for i in chosen]
    masks = predictions.pred_masks.cpu().numpy()
    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    # NOTE: should change the _postprocess without interpolating to the ori_img_size
    ret.pred_masks = masks
    ret.pred_classes = predictions.pred_classes.cpu().numpy()

    return ret

def do_train(cfg):
    train_dataset = cfg.DATASETS.TRAIN[0]
    metadata = MetadataCatalog.get(train_dataset)
    MapperClass = SamDatasetMapper
    mapper = MapperClass(cfg, True, augmentations = build_custom_augmentation(cfg, True))
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    for data in data_loader:
        train_img = [ins['image'].permute(1,2,0).numpy() for ins in data]
        for ind, img in enumerate(train_img):
            predictions = create_instances(data[ind]['instances'], img.shape[:2])
            vis = Visualizer(img, metadata)
            vis_gt  = vis.draw_instance_predictions(predictions).get_image()
            cv2.imwrite(f'output/visualize/gt_{ind}.png',vis_gt[:,:,::-1])
        break

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rsprompter_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def do_test(cfg, model):
    # test the testing image
    # test the anchor/proposals generated
    # test the inference result?
    test_dataset = cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(test_dataset)
    # thing 可数
    MapperClass = SamDatasetMapper
    mapper = MapperClass(cfg, False, augmentations = build_custom_augmentation(cfg, False))
    data_loader = custom_build_detection_test_loader(cfg, test_dataset, mapper=mapper)
    show_iter = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for data in data_loader:
            train_img = [ins['image'].permute(1,2,0).numpy() for ins in data]
            # the model 
            outs = model(data)

            # outs = outs.unsqueeze(1)
            # outs = torch.nn.functional.interpolate(outs, size=(img.shape[:2]), mode='bilinear', align_corners=False, )
            # outs = outs.squeeze(1)
            # NOTE: reback the postprocess in detector
            for ind, img in enumerate(train_img):
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
                resize_img = torch.nn.functional.interpolate(img_tensor, (data[ind]['height'],data[ind]['width']), mode='bilinear', align_corners=False)
                resize_img = resize_img.permute(0,2,3,1).squeeze().numpy()
                vis = Visualizer(resize_img, metadata)
                pred_instance = outs[ind]['instances'].to('cpu')
                vis_pred = vis.draw_instance_predictions(pred_instance).get_image()
                cv2.imwrite(f'output/visualize/pred_test_{ind+show_iter*len(data)}.png', vis_pred[:,:,::-1])    
            show_iter += 1
            if show_iter>=5:
                break        
    pass

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = build_model(cfg)
        # for debugging, load fastercnn here
        samCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        do_test(cfg, model)
       
    do_train(cfg)

if __name__ == '__main__':
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    