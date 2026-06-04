"""Various modules used in the decoder of the model.

Adapted from https://github.com/jinlinyi/PerspectiveFields
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from splatfactory import get_logger

logger = get_logger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class ConvModule(nn.Module):
    """Replacement for mmcv.cnn.ConvModule to avoid mmcv dependency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        use_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activate(x)


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, unit2only=False, upsample=True):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()
        self.upsample = upsample

        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass."""
        output = xs[0]

        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)

        return output
