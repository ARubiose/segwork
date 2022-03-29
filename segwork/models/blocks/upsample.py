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

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

        Args:
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
                output spatial sizes
            scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
                multiplier for spatial size. Has to match input size if it is a tuple.
            mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
                ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
                Default: ``'nearest'``
            align_corners (bool, optional): if ``True``, the corner pixels of the input
                and output tensors are aligned, and thus preserving the values at
                those pixels. This only has effect when :attr:`mode` is
                ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:

        super(Interpolate2d, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode.startswith('nearest') else align_corners

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False, **kwargs)