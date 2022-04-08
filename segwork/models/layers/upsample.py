"""Upsample module"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class Interpolate2d(nn.Module):
    """Down/up samples the input to either the given ``size`` or the given ``scale_factor``

    The algorithm used for interpolation is determined by mode. 
    The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.
    
    https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    Mode mode='nearest-exact' matches Scikit-Image and PIL nearest neighbours interpolation algorithms and fixes known 
    issues with mode='nearest'. This mode is introduced to keep backward compatibility.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

        Args:
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
                output spatial sizes
            scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
                multiplier for spatial size.
            mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
                ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
                Default: ``'nearest'``
            align_corners (bool, optional): if ``True``, the corner pixels of the input
                and output tensors are aligned, and thus preserving the values at
                those pixels. This only has effect when :attr:`mode` is
                ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    def __init__(
            self,
            size: Optional[Union[int, Tuple[int, int]]] = None,
            scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
            mode: str = 'nearest-exact',
            align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode.startswith('nearest') or mode=="area" else align_corners

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward method for interpolate module

        Args:
            input (torch.Tensor): The input tensor. If scale_factor is a tuple, its length has to match input.dim()

        Returns:
            torch.Tensor: Down/upsampled input
        """
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False, **kwargs)

# Needed if more tahn 1 type of transposed convolution
def create_convtrans2d(
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple] = 2,
        stride: Union[int, Tuple] = 2,
        padding: Union[int, Tuple] = 1,
        **kwargs) -> nn.Module:
    """ Factory method for transposed convolution building blocks
    
    Default: 2x input_size 
    
    TODO variants"""
    out_channels = out_channels or in_channels
    return nn.ConvTranspose2d(
            in_channels, 
            out_channels,  
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding, 
            **kwargs)

class DeConvBnAct(nn.Module):
    """Vision block: Transposed Conv + BatchNormalization + activation

    """

    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                scale_factor: Optional[int] = None,
                kernel_size: int = 2,
                stride: int = 2,
                norm_layer: Union[str, nn.Module] = nn.BatchNorm2d,
                act_layer: Union[str, nn.Module] = nn.ReLU,
                **kwargs) -> None:
        super().__init__()

        if scale_factor:
            kernel_size = scale_factor
            stride = scale_factor
            
        self.deconv = create_convtrans2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    @property
    def in_channels(self) -> int:
        return self.deconv.in_channels

    @property
    def out_channels(self) -> int:
        return self.deconv.out_channels

    def forward(self, x):
        x = self.deconv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x