import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
from model import ShuffleV2Block, ShuffleNetV2
from torchvision.models.quantization.utils import _replace_relu

class QuantizableShuffleV2Block(ShuffleV2Block):
    def __init__(self, *args, **kwargs):
        super(QuantizableShuffleV2Block, self).__init__(*args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return self.cat.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return self.cat.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        else:
            raise ValueError("use stride 1 or 2, current stride {}".format(self.stride))
    
    def fuse_model(self):
        quant.fuse_modules(self.branch_main, [['0', '1', '2'],
                                              ['3', '4'],
                                              ['5', '6', '7']], inplace=True)
        if self.stride == 2:
            quant.fuse_modules(self.branch_proj, [['0', '1'],
                                                  ['2', '3', '4']], inplace=True)


class QuantizableShuffleNetV2(ShuffleNetV2):
    def __init__(self, *args, **kwargs):
        super(QuantizableShuffleNetV2, self).__init__( *args, **kwargs, block = QuantizableShuffleV2Block)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == "2.0x":
            x = self.dropout(x)
        x = x.reshape(-1, self.stage_out_channels[-1])
        x = self.dequant(x)
        x = self.classifier(x)
        return x

    def fuse_model(self):
        quant.fuse_modules(self.first_conv, ['0', '1', '2'], inplace=True)
        quant.fuse_modules(self.conv_last, ['0', '1', '2'], inplace=True)
        for m in self.modules():
            if isinstance(m, QuantizableShuffleV2Block):
                m.fuse_model()
        _replace_relu(self)


def shufflenet_v2_x2_0(num_classes=1000):
    return QuantizableShuffleNetV2(num_classes=num_classes, model_size="2.0x")


def shufflenet_v2_x1_5(num_classes=1000):
    return QuantizableShuffleNetV2(num_classes=num_classes, model_size="1.5x")


def shufflenet_v2_x1_0(num_classes=1000):
    return QuantizableShuffleNetV2(num_classes=num_classes, model_size="1.0x")


def shufflenet_v2_x0_5(num_classes=1000):
    return QuantizableShuffleNetV2(num_classes=num_classes, model_size="0.5x")
