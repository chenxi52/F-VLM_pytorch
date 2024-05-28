#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

categories_seen = [
    {'id': 1, 'name': 'person'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'motorcycle'},
    {'id': 7, 'name': 'train'},
    {'id': 8, 'name': 'truck'},
    {'id': 9, 'name': 'boat'},
    {'id': 15, 'name': 'bench'},
    {'id': 16, 'name': 'bird'},
    {'id': 19, 'name': 'horse'},
    {'id': 20, 'name': 'sheep'},
    {'id': 23, 'name': 'bear'},
    {'id': 24, 'name': 'zebra'},
    {'id': 25, 'name': 'giraffe'},
    {'id': 27, 'name': 'backpack'},
    {'id': 31, 'name': 'handbag'},
    {'id': 33, 'name': 'suitcase'},
    {'id': 34, 'name': 'frisbee'},
    {'id': 35, 'name': 'skis'},
    {'id': 38, 'name': 'kite'},
    {'id': 42, 'name': 'surfboard'},
    {'id': 44, 'name': 'bottle'},
    {'id': 48, 'name': 'fork'},
    {'id': 50, 'name': 'spoon'},
    {'id': 51, 'name': 'bowl'},
    {'id': 52, 'name': 'banana'},
    {'id': 53, 'name': 'apple'},
    {'id': 54, 'name': 'sandwich'},
    {'id': 55, 'name': 'orange'},
    {'id': 56, 'name': 'broccoli'},
    {'id': 57, 'name': 'carrot'},
    {'id': 59, 'name': 'pizza'},
    {'id': 60, 'name': 'donut'},
    {'id': 62, 'name': 'chair'},
    {'id': 65, 'name': 'bed'},
    {'id': 70, 'name': 'toilet'},
    {'id': 72, 'name': 'tv'},
    {'id': 73, 'name': 'laptop'},
    {'id': 74, 'name': 'mouse'},
    {'id': 75, 'name': 'remote'},
    {'id': 78, 'name': 'microwave'},
    {'id': 79, 'name': 'oven'},
    {'id': 80, 'name': 'toaster'},
    {'id': 82, 'name': 'refrigerator'},
    {'id': 84, 'name': 'book'},
    {'id': 85, 'name': 'clock'},
    {'id': 86, 'name': 'vase'},
    {'id': 90, 'name': 'toothbrush'},
]

categories_unseen = [
    {'id': 5, 'name': 'airplane'},
    {'id': 6, 'name': 'bus'},
    {'id': 17, 'name': 'cat'},
    {'id': 18, 'name': 'dog'},
    {'id': 21, 'name': 'cow'},
    {'id': 22, 'name': 'elephant'},
    {'id': 28, 'name': 'umbrella'},
    {'id': 32, 'name': 'tie'},
    {'id': 36, 'name': 'snowboard'},
    {'id': 41, 'name': 'skateboard'},
    {'id': 47, 'name': 'cup'},
    {'id': 49, 'name': 'knife'},
    {'id': 61, 'name': 'cake'},
    {'id': 63, 'name': 'couch'},
    {'id': 76, 'name': 'keyboard'},
    {'id': 81, 'name': 'sink'},
    {'id': 87, 'name': 'scissors'},
]

categories_unseen_id = [x['id'] for x in categories_unseen]
categories_seen_id = [x['id'] for x in categories_seen]


def _get_metadata(cat):
    if cat == 'all':
        return _get_builtin_metadata('coco')
    elif cat == 'seen':
        id_to_name = {x['id']: x['name'] for x in categories_seen}
    else:
        assert cat == 'unseen'
        id_to_name = {x['id']: x['name'] for x in categories_unseen}

    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_COCO = {
    "coco_zeroshot_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2.json", 'seen'),
    "coco_zeroshot_val": ("coco/val2017", "coco/zero-shot/instances_val2017_unseen_2.json", 'unseen'),
    "coco_not_zeroshot_val": ("coco/val2017", "coco/zero-shot/instances_val2017_seen_2.json", 'seen'),
    "coco_generalized_zeroshot_val": ("coco/val2017", "coco/zero-shot/instances_val2017_all_2_oriorder.json", 'all'),
    "coco_zeroshot_train_oriorder": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2_oriorder.json", 'all'),
    
}
_root = os.getenv("DETECTRON2_DATASETS", "datasets")


for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_COCO.items():
    register_coco_instances(
        key,
        _get_metadata(cat),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )


def create_instances(predictions, image_size):
    ret = Instances(image_size)
    # this score is the box score?
    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_generalized_zeroshot_val")
    parser.add_argument("--conf-threshold", default=0, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    num = 0
    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
        num+=1
        if num>50:
            break
