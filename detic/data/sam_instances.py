from typing import Any, Tuple, Union, List
from detectron2.structures.instances import Instances
import torch
import itertools


class samInstances(Instances):
    def __init__(self, ori_image_size: Tuple[int,int], image_size: Tuple[int, int], **kwargs: Any):
        super().__init__(image_size, **kwargs)
        self._ori_image_size = ori_image_size

    @property
    def ori_image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._ori_image_size
    
    def to(self, *args: Any, **kwargs: Any) -> "samInstances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = samInstances(self._ori_image_size, self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret
    
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "samInstances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = samInstances(self._ori_image_size, self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret
    
    @staticmethod
    def cat(instance_lists: List["samInstances"]) -> "samInstances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, samInstances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        ori_image_size = instance_lists[0].ori_image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = samInstances(ori_image_size, image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret
    
    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "ori_image_height={}, ".format(self._ori_image_size[0])
        s += "ori_image_width={}, ".format(self._ori_image_size[1])
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s
    __repr__ = __str__