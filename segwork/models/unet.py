
from typing import OrderedDict

import torch.nn as nn
import torch

from segwork.models.blocks.base import ConvBnAct

class UNet(nn.Module):
    """UNet base arch"""
    def __init__(self, encoder:nn.Module, decoder:nn.Module)-> None:
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.decoder(features)

class UNetBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, **kwargs):
        self.blocks = nn.Sequential(OrderedDict([
            ('CBA1',ConvBnAct(in_channels, out_channels, kernel_size=kernel_size, **kwargs)),
            ('CBA2',ConvBnAct(in_channels, out_channels, kernel_size=kernel_size, **kwargs)),
            ]))