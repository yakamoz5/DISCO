from collections import OrderedDict
from torch import nn
import torch
from torchvision.models import resnet18,  resnet50


def conv2d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    padding = 1 if kernel_size != 1 else 0
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
        # self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)()

    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1, dropout_p: float = 0.2):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, stride=stride)
        # self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout_p, inplace=True)

        self.conv2 = conv2d(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TwoDResNet(nn.Module):
    def __init__(self, in_channels: int, n_outputs: int, n_blocks: int = 4,
                 bn_momentum: float = 0.05, n_basefilters: int = 32,
                 dropout_p: float = 0.2, no_pooling: bool = False):
        """
        This ResNet variant no longer uses adaptive pooling.
        Instead, after the convolutional blocks the feature map is flattened.
        The fc layer’s input dimension is computed automatically for a fixed input size of 32x32.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            n_basefilters,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # self.bn1 = nn.BatchNorm2d(n_basefilters, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2) if not no_pooling else nn.Identity()

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        # Build the residual blocks.
        blocks = [("block1", ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum))]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            blocks.append(
                (f"block{i+2}",
                 ResBlock(n_filters, 2 * n_filters,
                          bn_momentum=bn_momentum, stride=2, dropout_p=dropout_p))
            )
            n_filters *= 2
        self.blocks = nn.Sequential(OrderedDict(blocks))
        
        # Instead of adaptive pooling, we now use an identity and then flatten.
        self.gap = nn.Identity()
        self.flatten = nn.Flatten()

        # Determine the flattened feature dimension by doing a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            feat = self._forward_features(dummy)
            flatten_dim = feat.view(1, -1).size(1)
            self.feature_size = flatten_dim
        self.fc = nn.Linear(flatten_dim, n_outputs, bias=True)
    
    def _forward_features(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.blocks(x)
        x = self.gap(x)  # This is Identity now.
        return x

    def forward(self, x):
        features = self._forward_features(x)
        flat_features = self.flatten(features)
        return self.fc(flat_features), flat_features


class ConvNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_outputs=1, n_basefilters: int = 2, depth: int = 3, activation=nn.Tanh):
        """
        A simple convolutional encoder that uses strided convolutions with padding.
        Each layer uses a kernel size of 3, padding=1 and a stride of 2, halving the spatial dimensions.
        The number of channels doubles at each layer.
        The flattened feature dimension is computed automatically using a dummy forward pass.
        """
        super().__init__()
        layers = []
        current_in = in_channels
        for i in range(depth):
            out_channels = n_basefilters * (2 ** i)
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(activation())
            current_in = out_channels
        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

        # Compute the flattened dimension automatically using a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            self.feature_size = self.encoder(dummy).size(1)

        self.fc = nn.Linear(self.feature_size, n_outputs, bias=True)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features), features


class Resnet18(nn.Module):
    def __init__(self, n_outputs: int = 1, pretrained: bool = True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = resnet.fc.in_features
        self.fc = resnet.fc
    def forward(self, x):
        features = self.encoder(x)
        flat_features = features.view(features.size(0), -1)
        return self.fc(flat_features), flat_features


class Resnet50(nn.Module):
    def __init__(self, n_outputs: int = 1, pretrained: bool = True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = resnet.fc.in_features
        self.fc = resnet.fc
    def forward(self, x):
        features = self.encoder(x)
        flat_features = features.view(features.size(0), -1)
        return self.fc(flat_features), flat_features
