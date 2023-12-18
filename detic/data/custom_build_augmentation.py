# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.data import transforms as T
from detic.data.transforms.custom_augmentation_impl import DivideBy255, Normalize
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop, ResizeLongestSize
from detectron2.config import LazyCall as L
def build_custom_augmentation(cfg, is_train, scale=None, size=None, \
    min_size=None, max_size=None):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge':
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN if min_size is None else min_size
            max_size = cfg.INPUT.MAX_SIZE_TRAIN if max_size is None else max_size
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    elif cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
        if is_train:
            scale = cfg.INPUT.SCALE_RANGE if scale is None else scale
            size = cfg.INPUT.TRAIN_SIZE if size is None else size
        else:
            scale = (1, 1)
            size = cfg.INPUT.TEST_SIZE
        augmentation = [EfficientDetResizeCrop(size, scale)]
    elif cfg.INPUT.CUSTOM_AUG == 'ResizeLongestSize':
        if is_train:
            size = cfg.INPUT.TRAIN_SIZE
        else:
            size = cfg.INPUT.TEST_SIZE 
        augmentation = [ResizeLongestSize(longest_length = size)]
        if is_train:
            augmentation.append(T.RandomFlip(prob=0.5))
    elif cfg.INPUT.CUSTOM_AUG == 'ResizeFlip':
        if is_train:
            size = cfg.INPUT.TRAIN_SIZE
        else:
            size = cfg.INPUT.TEST_SIZE
        augmentation = [T.Resize((size, size))]
        if is_train:
            augmentation.append(T.RandomFlip(prob=0.5))
    elif cfg.INPUT.CUSTOM_AUG == 'ResizeLongLSJ':
        size = cfg.INPUT.TRAIN_SIZE
        #先 normalize 再 padding
        if is_train:
            # resizeScale: 从给定的target-height width, 进行 scale, 取最接近 scale[i]* length[i] 的值
            # resizeScale可能获得的 scale 超出原图大小
            # fixedSizeCrop: padding wih val
            augmentation = [
                            T.ResizeScale(
                            min_scale=0.1, max_scale=2.0, target_height=size, target_width=size
                            ),
                            DivideBy255(),
                            Normalize(mean=([0.48145466, 0.4578275, 0.40821073]), std=([0.26862954, 0.26130258, 0.27577711])),
                            T.FixedSizeCrop(crop_size=(size, size), pad_value=0, seg_pad_value=0),
                            T.RandomFlip(horizontal=True),
                            ]
            
        else:
            augmentation =[
                            T.ResizeScale(
                            min_scale=1., max_scale=1., target_height=size, target_width=size
                            ),
                            DivideBy255(),
                            Normalize(mean=([0.48145466, 0.4578275, 0.40821073]), std=([0.26862954, 0.26130258, 0.27577711])),
                            T.FixedSizeCrop(crop_size=(size, size), pad_value=0, seg_pad_value=0),
                        ]

    else:
        assert 0, cfg.INPUT.CUSTOM_AUG
    return augmentation


build_custom_transform_gen = build_custom_augmentation
"""
Alias for backward-compatibility.
"""
