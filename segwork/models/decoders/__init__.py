from turtle import forward
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.layers import create_conv2d, get_act_layer

from segwork.models.blocks.base import ConvBnAct
from segwork.models.blocks.upsample import Interpolate2d

def get_decoder_block():
    pass

class DecoderBlock(nn.Module):
    """Basic decoder block upsample"""
    def __init__(self,
                in_channels:int,
                out_channels:Union[int, tuple],
                layers: int = 3,
                size: Optional[Union[int, Tuple[int, int]]] = None,
                scale_factor: Optional[Union[float, Tuple[float, float]]] = 2,
                mode: str = 'nearest',
                align_corners: bool = False,
                kernel_size:int = 3,
                norm_layer=nn.BatchNorm2d,
                act_layer=get_act_layer('silu')
                ) -> None:
        super().__init__()
        self.upsample = Interpolate2d(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        if (isinstance(out_channels, int)):
            out_channels = [out_channels for _ in range(layers)]

        channels = [in_channels, *out_channels]
        layers = []
        for i in range(len(out_channels)):
            layers.append(
                ConvBnAct(channels[i], channels[i+1], kernel_size=kernel_size, 
                norm_layer=norm_layer, act_layer=act_layer))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x:torch.tensor):
        x = self.upsample(x)
        out = self.blocks(x)
        return out


