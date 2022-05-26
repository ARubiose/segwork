from typing import Tuple
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from segwork.model.layer.base import ConvBnAct
from segwork.registry import modules

@modules.register
class PyramidPooling(nn.Module):
    """Pyramid Pooling Module
    
    'Pyramid Scene Parsing Network' -->  https://arxiv.org/pdf/1612.01105v2.pdf
    """
    _register_name = 'PyramidPooling'

    def __init__(self,
        in_channels:int, 
        reduction_channels:int,
        bins: typing.Tuple[int],
        upsample_mode:str = 'bilinear',
        align_corners: bool = True,
        **kwargs
        ):
        super().__init__()

        self.features = []
        for bin in enumerate(bins):
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    ConvBnAct(in_channels, reduction_channels, kernel_size=1, **kwargs),
                )
            )
        self.features = nn.ModuleList(self.features)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        """Forward pass. 
        
        x.size() = BxCxHxW"""
        input_size = x[-2].size()
        out = [x]
        for feature in self.features:
            feature_bin = feature(x)
            out.append(F.interpolate(feature_bin, input_size, mode=self.upsample_mode, align_corners=self.align_corners))
        out = torch.cat(out, 1)