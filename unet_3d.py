import torch
import torch.nn as nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DoubleConv, self).__init__()

        # to define intermediate channels
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2

        # to define neural layers
        layers = [
            # first convolution
            nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # second convolution
            nn.Conv3d(in_channels=channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]

        # to insert batch normalization layer
        if batch_normal:
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(-1, nn.BatchNorm3d(out_channels))

        # to construct serializer
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=out_channels, batch_normal=batch_normal),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
        super(UpSampling, self).__init__()
        # to up sampling
        if bilinear:
            # 采用双线性插值的方法进行上采样
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # 采用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        # to perform convolution
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, batch_normal=batch_normal)

    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)
        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        # 进行卷积
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        self.inputs = DoubleConv(in_channels=self.in_channels, out_channels=64, batch_normal=self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)

        self.outputs = LastConv(64, num_classes)

    def forward(self, x):
        # down section
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        # up section
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)
        return x


if __name__ == '__main__':
    data = torch.randn(1, 3, 256, 256, 128)
    print(data.shape)
    model = UNet3D(in_channels=3, bilinear=False)
    print(model)
    out = model(data)
    print(out.shape)
