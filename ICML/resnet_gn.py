from collections import OrderedDict
from torch import nn
import torch


def conv2d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
    )


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1, dropout_p=0.2):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        self.dropout1 = nn.Dropout(p=dropout_p, inplace=True)

        self.conv2 = conv2d(out_channels, out_channels)
        self.bn2 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class TwoDResNetGN(nn.Module):
    def __init__(self, in_channels: int, n_outputs: int, n_blocks: int = 4, bn_momentum: float = 0.05, n_basefilters: int = 32, dropout_p: float=0.2,
                 no_pooling: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            n_basefilters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,)
        self.bn1 = nn.GroupNorm(num_groups=n_basefilters, num_channels=n_basefilters)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        blocks = [
            ("block1", ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, dropout_p=dropout_p))
        ]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            blocks.append(
                (f"block{i+2}", ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=2, dropout_p=dropout_p))
            )
            n_filters *= 2

        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            self.feature_size = self.avg_pool(self.blocks(self.pool1(self.conv1(dummy)))).view(-1).shape[0]
        
        self.fc = nn.Linear(self.feature_size, n_outputs, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.blocks(out)
        out = self.avg_pool(out).view(out.size(0), -1)
        return self.fc(out), out
