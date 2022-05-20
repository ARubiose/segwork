from ctypes import Union
from typing import Tuple

import torch
import torch.nn as nn

class PPM(nn.Module):
    """Pyramid pooling module 
    
    'Pyramid Scene Parsing Network' -->  https://arxiv.org/pdf/1612.01105v2.pdf

    The pyramid pooling module fuses features under `len(bins)` different pyramid scales.
    Args:
        in_dim
"""
    def __init__(self, 
            in_channels:int, 
            out_scale_channels: Union[int, Tuple[int]], 
            bins: Tuple[int]):
        super(PPM, self).__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_scale_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_scale_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)