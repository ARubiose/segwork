"""Residual blocks

Code taken from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

from msilib.schema import Shortcut
from turtle import forward
import typing

from torch import Tensor
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Flexible residual block """

    _register_name = 'Residual block'

    _default_params = {
        'in_channels' : 64,
        'out_channels' : 64,
        'kernel_size' : 3
    }

    def __init__(self, 
                block: nn.Module,
                block_args: typing.Iterable = {},
                shortcut:typing.Callable[...,nn.Module] = None,
                skip_fn: typing.Callable = torch.add,
                activation = nn.ReLU,
                layers: int = 1,
                **kwargs) -> None:
        super(ResidualBlock, self).__init__()

        if isinstance(block_args, typing.Dict):
            block_args = [block_args for _ in range(layers)]

        # Repeat last args
        if len(block_args) < layers:
            block_args.append([block_args[-1] for _ in range(layers - len(block_args))])
        
        blocks = []
        for idx in range(layers):
            blocks.append(block(**block_args[idx]))
        self.blocks = nn.Sequential(*blocks)

        self.shortcut = shortcut
        self.activation = activation(inplace=True)
        self.skip_fn = skip_fn
        self.layers = layers

    def forward(self, x:Tensor):
        skip_connection = x
        x = self.blocks(x)
        if self.shortcut:
            out = self.skip_fn(self.shortcut(skip_connection), x)
        else:
            out = self.skip_fn(skip_connection, x) 
        return self.activation(out)