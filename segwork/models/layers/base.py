from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1, **kwargs) -> nn.Conv2d:
    """3x3 convolution; Default: output stride = 1.0"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        **kwargs
    )

def conv1x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1, **kwargs) -> nn.Conv2d:
    """1x3 convolution; Default: output stride = 1.0"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(1,3),
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        **kwargs
    )

def conv3x1(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1, **kwargs) -> nn.Conv2d:
    """1x3 convolution; Default: output stride = 1.0"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(3,1),
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        **kwargs
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1, **kwargs) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=stride, 
        bias=False,
        **kwargs)

def ConvBnAct():
    """ Layer Conv + BN + Act"""

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
        
        # Sanity check - Raise errors
        self._sanity_check()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                # Average pooling works better than max pooling in all settings
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, out_scale_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_scale_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def _sanity_check(self, out_dim, bins) -> bool:
        pass #TODO

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)