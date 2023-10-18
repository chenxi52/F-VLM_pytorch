from typing import List, Union
from detectron2.modeling.poolers import ROIPooler
from torch import Tensor
from detectron2.structures import Boxes
from detectron2.modeling.poolers import convert_boxes_to_pooler_format

class customRoiPooler(ROIPooler):
    def forward(self, x: List[Tensor], box_lists: List[Boxes]) -> Tensor:
        """
        Returns:
            output:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
            Rois_index: the roi index to image in this batch
        """
        output = super().forward(x, box_lists)
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        import ipdb;ipdb.set_trace()
        return output, pooler_fmt_boxes[:,0]