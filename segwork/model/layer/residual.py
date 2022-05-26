"""Residual blocks

Code taken from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

from msilib.schema import Shortcut
from turtle import forward
from typing import Callable, Optional

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Identity


class ResidualBlock(nn.Module):
    """Flexible residual block """
    def __init__(self, 
                block:nn.Module,
                shortcut:Callable[...,nn.Module] = None,
                skip_fn: Callable = torch.add,
                activation = nn.ReLU,
                **kwargs) -> None:
        super(ResidualBlock, self).__init__()
        self.block = block
        self.shortcut = shortcut
        self.activation = activation(inplace=True)
        self.skip_fn = skip_fn

    def forward(self, x:Tensor):
        skip_connection = x
        x = self.block(x)
        if self.shortcut:
            out = self.skip_fn(self.shortcut(skip_connection), x)
        else:
            out = self.skip_fn(skip_connection, x) 
        return self.activation(out)