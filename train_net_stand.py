#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import datetime

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, dataset_mapper

from detic.data.build import custom_build_detection_test_loader, custom_build_detection_train_loader
from detectron2.data.build import get_detection_dataset_dicts, build_detection_test_loader, build_detection_train_loader, _train_loader_from_config

from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
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

from detic.data.custom_dataset_mapper import SamDatasetMapper
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.custom_checkpointer import samCheckpointer
from detic.config import add_rsprompter_config
from detectron2.utils.logger import setup_logger
from detic.custom_solver import build_sam_optimizer
import wandb
logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        # data_loader = build_detection_test_loader(cfg, dataset_name)
        #####
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else SamDatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = custom_build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        #####
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" :
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            # if dataset_name == 'coco_generalized_zeroshot_val':
            #     # Additionally plot mAP for 'seen classes' and 'unseen classes'
            #     evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            # else:
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
        
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        # also set requires_grad for module
        optimizer = build_sam_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)

    scheduler = build_lr_scheduler(cfg, optimizer)

    # checkpointer = DetectionCheckpointer(
    #     model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    # )
    ######
    checkpointer = samCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    ######
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    ###### set prompter params False
    # for key, params in model.named_parameters():
    #     if 'prompter' in key:
    #         params.requires_grad = False

    ######
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    # writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
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

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
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
        # if comm.is_main_process() and cfg.WANDB:
        #     wandb.finish()


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
    )  # if you don't like any of the default setup, write your own setup code
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg


def main(args):
    cfg = setup(args)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    if comm.is_main_process() and cfg.WANDB:
        wandb.init(project='SamDetector', name=TIMESTAMP, config=cfg)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        samCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        for key, params in model.named_parameters():
            params.requires_grad = False
        return do_test(cfg, model)

    ##### freeze prompter params
    for key, params in model.sam.prompt_encoder.named_parameters():
        params.requires_grad = False
    ########

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    