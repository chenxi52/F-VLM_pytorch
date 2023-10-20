# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Part of the code is from https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/data/transforms.py 
# Modified by Xingyi Zhou
# The original code is under Apache-2.0 License
import numpy as np
import sys
import torch
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image

import detectron2.data.transforms as T
from detectron2.data.transforms import ResizeTransform
from .custom_transform import EfficientDetResizeCropTransform
from typing import Tuple
__all__ = [
    "EfficientDetResizeCrop",
]
# Augmentation return transforms
class EfficientDetResizeCrop(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, size, scale, interp=Image.BILINEAR
    ):
        """
        """
        super().__init__()
        self.target_size = (size, size)
        self.scale = scale
        self.interp = interp

    def get_transform(self, img):
        # Select a random scale factor.
        scale_factor = np.random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]
        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.shape[1], img.shape[0]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * np.random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * np.random.uniform(0, 1))
        return EfficientDetResizeCropTransform(
            scaled_h, scaled_w, offset_y, offset_x, img_scale, self.target_size, self.interp)


class ResizeLongestSizeFlip(T.Augmentation):
    @torch.jit.unused
    def __init__(self, longest_length,interp=Image.BILINEAR) -> None:
        super().__init__()
        self._init(locals())

    @torch.jit.unused
    def get_transform(self,img):
        width, height = img.shape[1], img.shape[0]
        scaled_h, scaled_w = self.get_output_shape(height, width, self.longest_length)
        return ResizeTransform(height, width, scaled_h, scaled_w, self.interp)
    
    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

# class PadAug(T.Augmentation):
#     def __init__(self, target_size) -> None:
#         super().__init__()
#         self.target_size = target_size

#     def get_transform(self, img) -> Transform:
#         width, height = img.shape[1], img.shape[0]
#         return PadTransform(height, width, self.target_size)

# class HFlipMaskAug(T.RandomFlip):
#     def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
#         super().__init__(prob, horizontal=horizontal, vertical=vertical)
    
#     def get_transform(self, image):
#         h, w = image.shape[:2]
#         do = self._rand_range() < self.prob
#         if do:
#             return HFlipTransform(w)
#         else:
#             return NoOpTransform()
            
