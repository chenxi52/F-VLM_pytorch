from detectron2.modeling.roi_heads import MaskRCNNConvUpsampleHead, ROI_MASK_HEAD_REGISTRY, BaseMaskRCNNHead
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat
from ..layers.custom_batchnorm import get_norm
from torch import nn
import fvcore.nn.weight_init as weight_init

@ROI_MASK_HEAD_REGISTRY.register()
class CustomMaskRCNNConvUpsampleHead(MaskRCNNConvUpsampleHead):
    '''
    add norm after deconv
    '''
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        BaseMaskRCNNHead.__init__(self, **kwargs)
        nn.Sequential.__init__(self)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim, momentum=0.997, epsilon=1e-4),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        self.norm = get_norm(conv_norm, conv_dims[-1], momentum=0.997, epsilon=1e-4)
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)
