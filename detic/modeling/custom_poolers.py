from typing import List, Union
from detectron2.modeling.poolers import ROIPooler
from torch import Tensor
from detectron2.structures import Boxes
import torch
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing
from detectron2.modeling.poolers import _create_zeros, assign_boxes_to_levels, convert_boxes_to_pooler_format
from detectron2.layers import nonzero_tuple


class customRoiPooler(ROIPooler):
    def forward(self, x: List[Tensor], box_lists: List[Boxes]) -> Tensor:
        """
        Returns:
            output:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
            Rois_index: the roi index to image in this batch
        """
        num_level_assignments = len(self.level_poolers)

        if not is_fx_tracing():
            torch._assert(
                isinstance(x, list) and isinstance(box_lists, list),
                "Arguments to pooler must be lists",
            )
        assert_fx_safe(
            len(x) == num_level_assignments,
            "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
                num_level_assignments, len(x)
            ),
        )
        assert_fx_safe(
            len(box_lists) == x[0].size(0),
            "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
                x[0].size(0), len(box_lists)
            ),
        )
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # pooler_fmt_boxes_level: 该层的所有 boxes
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            # x[level]: level 层该 batch所有的 x
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))
            # 每层分别经过 pooler 并按照 index 放入 output
        return output, pooler_fmt_boxes[:,0]