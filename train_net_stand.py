#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import datetime
import detectron2.utils.comm as comm
from detectron2.checkpoint import PeriodicCheckpointer,DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader 
from detectron2.data.build import build_detection_train_loader
from detic.modeling.clip import clip
import numpy as np
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from torch.cuda.amp import GradScaler
from detic.data.custom_dataset_mapper import SamDatasetMapper
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.custom_checkpointer import samCheckpointer
from detic.config import add_rsprompter_config
from detectron2.utils.logger import setup_logger
from detic.custom_solver import build_sam_optimizer
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator
from detic.evaluation.custom_lvis_eval import CustomLVISEvaluator,LVISEvaluatorFixedAP
import wandb
import torch.nn as nn
from detic.prompt_engineering import get_prompt_templates
from detic import constants
import pickle
import os
import sys
import numpy as np

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        #####
        mapper = SamDatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, is_train=False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        #####
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" :
            if cfg.TEST.FIXED_AP:
                evaluator = LVISEvaluatorFixedAP(dataset_name, cfg, True, output_folder)
            else:
                evaluator = CustomLVISEvaluator(dataset_name, cfg, True, output_folder)

        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
        if cfg.SOLVER.AMP.ENABLED:
            with torch.cuda.amp.autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    # also set requires_grad for module
    optimizer = build_sam_optimizer(cfg, model, logger)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, 
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR+f'/{TIMESTAMP}'),
        ]
        if comm.is_main_process()
        else []
    )
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    #####
    mapper = None if cfg.INPUT.CUSTOM_AUG == 'default' \
            else SamDatasetMapper(
                cfg, True, augmentations=build_custom_augmentation(cfg, True))
    #####
    
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    if cfg.SOLVER.AMP.ENABLED:
        scaler = GradScaler()
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            if cfg.SOLVER.AMP.ENABLED:
                with torch.cuda.amp.autocast():
                    loss_dict = model(data)
                    losses = sum(loss_dict.values())
            try: assert torch.isfinite(losses).all()
            except AssertionError:
                print("*"*50)
                print('loss is infinite')
                losses = torch.tensor(0., requires_grad=True, device=losses.device)

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.SOLVER.AMP.ENABLED:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
                if comm.is_main_process() and cfg.WANDB:
                    loss_dict_reduced['lr'] = optimizer.param_groups[0]["lr"]
                    loss_dict_reduced['iteration'] = iteration
                    loss_dict_reduced['total_loss'] = losses_reduced
                    wandb.log(loss_dict_reduced)
            periodic_checkpointer.step(iteration)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rsprompter_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d

def freeze_module(x):
    """
    """
    for p in x.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(x)
    return x

@torch.no_grad()
def get_custom_text_feat(clip_name, clip_model, class_names):
    def extract_mean_emb(text):
        tokens = clip.tokenize(text).cuda()
        if len(text) > 10000:
            text_features = torch.cat([
                clip_model.encode_text(text[:len(text) // 2]),
                clip_model.encode_text(text[len(text) // 2:])],
                dim=0)
        else:
            text_features = clip_model.encode_text(tokens)
        
        text_features = torch.mean(text_features, 0, keepdims=True)
        return text_features[0]

    templates = get_prompt_templates()
    clss_embeddings = []
    for clss in class_names:
        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
        clss_embeddings.append(extract_mean_emb(txts))
    background_embedding, _ = np.load(f'./datasets/{clip_name.replace("RN", "r")}_bg_empty_embed.npy', allow_pickle=True)
    clss_embeddings.append(torch.tensor(background_embedding).squeeze().to(clss_embeddings[0].device))
    text_emb = torch.stack(clss_embeddings, dim=0)
    return text_emb

def main(args):
    cfg = setup(args)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    if comm.is_main_process() and cfg.WANDB:
        wandb.init(project='SamDetector', name=TIMESTAMP, config=cfg)

    model = build_model(cfg)
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.TYPE)
    if not args.eval_only:
        model.train()
    clip_model = freeze_module(clip_model)
    model.clip = clip_model
    ##############
    # text_feats = get_custom_text_feat(cfg.MODEL.BACKBONE.TYPE, clip_model, constants.COCO_INSTANCE_CLASSES )
    # text_emb = np.load('datasets/coco/embeddings/resnet_50/coco_embed.npy')
    # text_emb= torch.tensor(text_emb).to(text_feats.device)
    # # import ipdb; ipdb.set_trace()
    # sys.exit()
    if 'coco' in cfg.DATASETS.TRAIN[0]:
    # if torch.all(text_feats == save_text.to(text_feats.device)):
    #     logger.info('text feats are the same')
    # else:
    #     logger.info('text feats are different')
    #     logger.info(torch.where(text_feats != save_text))

        text_feats = get_custom_text_feat(cfg.MODEL.BACKBONE.TYPE, clip_model, constants.COCO_SEEN_CLS if not args.eval_only else constants.COCO_INSTANCE_CLASSES)
        with open('datasets/coco/coco_cls_seen.pkl' if not args.eval_only else 'datasets/coco/coco_cls.pkl', 'rb') as f:
            save_text = pickle.load(f)
        if torch.all(text_feats == save_text.to(text_feats.device)):
            logger.info('text feats are the same')
        else:
            logger.info('text feats are different')
            logger.info(torch.where(text_feats != save_text))
            with open('datasets/coco/coco_cls_seen.pkl' if not args.eval_only else 'datasets/coco/coco_cls.pkl', 'wb') as f:
                pickle.dump(text_feats, f)
    elif 'lvis' in cfg.DATASETS.TRAIN[0]:
        import ipdb; ipdb.set_trace()
        thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        
        text_feats = get_custom_text_feat(cfg.MODEL.BACKBONE.TYPE, thing_classes)
        with open('datasets/lvis/lvis_base_cls.pkl' if not args.eval_only else 'datasets/lvis/lvis_cls.pkl', 'rb') as f:
            save_text = pickle.load(f)
        if torch.all(text_feats == save_text.to(text_feats.device)):
            logger.info('text feats are the same')
        else:
            logger.info('text feats are different')
            logger.info(torch.where(text_feats != save_text))
            with open('datasets/lvis/lvis_base_cls.pkl' if not args.eval_only else 'datasets/lvis/lvis_cls.pkl', 'wb') as f:
                pickle.dump(text_feats, f)
    #################
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )
    
    do_train(cfg, model, resume=args.resume)
    return None

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    