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

class ConvBnAct(nn.Module):
    """ Block: Conv + BatchNormalization + activation 
    
    https://arxiv.org/pdf/1502.03167.pdf
    """

    _register_name = 'ConvBnAct'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU, inplace:bool = True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if activation is None else activation(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x