import math

import torch
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
    

class DoubleConv(nn.Module):
    """DoubleConv is a basic building block of the encoder and decoder components.
    Consists of two convolutional layers followed by a ReLU activation function.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class NConv(nn.Module):
    """DoubleConv is a basic building block of the encoder and decoder components.
    Consists of two convolutional layers followed by a ReLU activation function.
    """
    def __init__(self, in_channels, out_channels, n=3):
        super().__init__()
        n_conv = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        for i in range(1, n):
            n_conv.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        self.n_conv = nn.Sequential(*n_conv)

    def forward(self, x):
        x = self.n_conv(x)
        return x


class Down(nn.Module):
    """Downscaling.
    Consists of two consecutive DoubleConv blocks followed by a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, n=2):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            NConv(in_channels, out_channels, n)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling.
    Performed using transposed convolution and concatenation of feature maps from the corresponding "Down" operation.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, n=2):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = NConv(in_channels, out_channels, n=n)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = NConv(in_channels, out_channels, n=n)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input tensor shape: (batch_size, channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, n=2):
        super(UNet, self).__init__()
        self.inc = (NConv(n_channels, 64, n=n))
        self.down1 = Down(64, 128, n=n)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256 // factor, n=n)

        self.up1 = Up(256, 128 // factor, bilinear, n=n)
        self.up2 = Up(128, 64 // factor, bilinear, n=n)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)

        return x