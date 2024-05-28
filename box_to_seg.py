#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, print_csv_format
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import launch
import sys
def create_instances(args, predictions, image_size, file_name=None, image_id=None):
    """
    It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
    """
    ret = Instances(image_size)
    # this score is the box score?
    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(args, predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    insdict = {}
    insdict['image_id'] = image_id
    insdict['file_name'] = file_name
    insdict['height'] = image_size[0]
    insdict['width'] = image_size[1]
    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return insdict,ret

def create_output_instances(mask_preds, image_size, box_instances, index):
    ret = {}
    input_preds = box_instances[index]
    ins = Instances(image_size, **input_preds.get_fields())
    ins.set('pred_masks', mask_preds) 
    ret['instances'] = ins
    return ret

def main(args,dicts,pred_by_image):
    os.makedirs(args.output, exist_ok=True)
    evalutor = COCOEvaluator(dataset_name=args.dataset,tasks=['segm'],distributed=True, output_dir=args.output)
    evalutor.reset()
    model = sam_model_registry['vit_t'](checkpoint='mobile_sam.pt')
    model = model.cuda()

    model.eval()
    predictor = SamPredictor(model)
    with torch.no_grad():
        for dic in tqdm.tqdm(dicts):
            img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
            basename = os.path.basename(dic["file_name"])
            predictions , box_instances= create_instances(args, pred_by_image[dic["image_id"]], img.shape[:2], basename, dic["image_id"])
            boxes = box_instances.pred_boxes.tensor
            predictor.set_image(img,image_format='RGB')
            predictor.features = predictor.features[0]
            # predictor.features = torch.repeat_interleave(predictor.features[0], boxes.shape[0], dim=0)
            bz = len(boxes)
            for index in range(bz):
                masks, iou_predictions, low_res_masks=predictor.predict(
                                        point_coords=None, 
                                        point_labels=None, 
                                        box=boxes[index].unsqueeze(0).numpy(),
                                        mask_input=None,
                                        multimask_output=False,
                                        return_logits=True)
                output_instance = create_output_instances(masks, img.shape[:2], box_instances, index)
                evalutor.process([predictions],[output_instance])
    results = evalutor.evaluate()
    logger.info("Evaluation results for {} in csv format:".format(args.dataset))
    print_csv_format(results)

def dataset_id_map(args,ds_id):
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        return ds_id - 1
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))


if __name__ == "__main__":
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf_threshold", default=0.02, type=float, help="confidence threshold")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")

    args = parser.parse_args()

    logger = setup_logger(output=args.output)

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))

    launch(main,
           args.num_gpus,
           num_machines=1,
           machine_rank=0,
           dist_url=args.dist_url,
           args=(args, dicts,pred_by_image))