from detectron2.evaluation.lvis_evaluation import _evaluate_box_proposals,LVISEvaluator,_evaluate_predictions_on_lvis
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
import pickle
import os
import copy
from collections import defaultdict
import json
import detectron2.utils.comm as comm
from detectron2.utils.logger import create_small_table
class CustomLVISEvaluator(LVISEvaluator):
    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions. AP100,300,1000
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].objectness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 300, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._lvis_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res


class LVISEvaluatorFixedAP(LVISEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, topk=10000):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self.topk = topk
        self.by_cat = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to("cpu")

            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            boxes = instances.pred_boxes.tensor.tolist()
            masks = instances.pred_masks.tensor.tolist()
            for score, cls, box in zip(scores, classes, boxes):
                self.by_cat[cls].append({
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": box,
                    "score": score,
                    "segmentation": masks
                })

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            all_cats = comm.gather(self.by_cat, dst=0)
            if comm.is_main_process():
                for cats in all_cats:
                    for cat, anns in cats.items():
                        self.by_cat[cat].extend(anns)
        else:
            all_cats = [self.by_cat]

        if not comm.is_main_process():
            return

        # Keep top k predictions for each category
        for cat, anns in self.by_cat.items():
            anns.sort(key=lambda ann: ann["score"], reverse=True)
            self.by_cat[cat] = anns[:self.topk]

        # Convert to LVIS results format
        lvis_results = []
        for cat_anns in self.by_cat.values():
            lvis_results.extend(cat_anns)

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        # Call the standard LVIS evaluation
        for task in sorted(["bbox", "segm"]):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                class_names=self._metadata.get("thing_classes"),
            )
            self._results[task] = res
        return copy.deepcopy(self._results)
