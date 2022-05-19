"""Residual blocks

Code taken from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

from msilib.schema import Shortcut
from turtle import forward
from typing import Callable, Optional

from torch import Tensor
import torch.nn as nn
from torch.nn import Identity

from segwork.models.layers.base import conv3x3

class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, 
                block:nn.Module,
                shortcut:Callable[...,nn.Module] = None,
                **kwargs) -> None:
        super(ResidualBlock, self).__init__()
        self.block = block
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x:Tensor):
        skip_connection = x
        x = self.block(x)
        out = x + skip_connection if self.shortcut is None \
            else self.shortcut(skip_connection) + x
        return self.relu(out)