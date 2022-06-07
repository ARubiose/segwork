import typing
import PIL

import torch

COLOR_CHANNELS = 3
GRAYSCALE = 1

class ColorMasktoIndexMask:
    """Class to transform RGB color mask to index numpy masks with the specified color mapping"""

    def __init__(self, colors:typing.MutableMapping):
        self._colors = colors
    
    def __call__(self, label:torch.Tensor, dtype:torch.dtype = None):
        assert  isinstance(label, torch.Tensor), f'Label argument must be of type {torch.Tensor}. Got {label.__class__.__name__}'
        
        if dtype and dtype != label.dtype:
            label = label.type(dtype)
        dtype = label.dtype

        mask = torch.zeros(label.size()[-2], label.size()[-1], dtype= dtype)

        for color in self._colors:
            idx = (label == torch.tensor(color, dtype= dtype).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(-3) == 3)          
            mask[validx] = torch.tensor(self._colors[color], dtype= dtype)

        return mask

class IndexMasktoColorMask:
    """Class to transform index numpy masks to RGB/GreyScalewith the specified color mapping.
    
    Args:
    """

    def __init__(self, colors:typing.MutableMapping):
        self._colors = colors

    def __call__(self, label:torch.Tensor, channels:int = COLOR_CHANNELS, dtype = torch.uint8):
        assert  isinstance(label, torch.Tensor), f'Label argument must be of type {torch.Tensor}. Got {label.__class__.__name__}'
        
        if dtype and dtype != label.dtype:
            label = label.type(dtype)
        dtype = label.dtype

        mask = torch.zeros(channels, label.size()[-2], label.size()[-1], dtype= dtype)

        for color, idx in self._colors.items():
            validx = (label == torch.tensor(idx, dtype=dtype))  
            color = torch.tensor(color, dtype=dtype).unsqueeze(1)     
            mask[:,validx] = color

        return mask

