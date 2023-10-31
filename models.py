import math

import torch.nn as nn
from torchvision import models

from util.unet import UNet


class RegNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.regnet = models.regnet_y_32gf(
            weights="RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1"
        )
        
        self.regnet.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.regnet.fc = nn.Linear(3712, 6)

        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        logit = self.regnet(input_batch)

        return logit, self.head_softmax(logit)
    

class ResNet18Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=6, bias=True)  # 出力チャネル数を1000->6

        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        logit = self.resnet18(input_batch)

        return logit, self.head_softmax(logit)


# UNet for image denoise
class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(num_features=1)
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        m.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output