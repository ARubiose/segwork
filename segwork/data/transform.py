import os
from pathlib import Path
import typing

import torch
import torchvision
import numpy as np
from tqdm import tqdm

from segwork.data.dataset import SegmentationDataset\
    
__all__ = ['ColorMasktoIndexMask', 'IndexMasktoColorMask', 'generate_numpy_files' ]

COLOR_CHANNELS = 3
GRAYSCALE = 1

class ColorMasktoIndexMask(object):
    """Class to transform RGB color mask to index numpy masks with the specified color mapping"""

    def __init__(self, colors:typing.MutableMapping, dtype:torch.dtype = None):
        self._colors = colors
        self._dtype = dtype
    
    def __call__(self, label:torch.Tensor):
        assert  isinstance(label, torch.Tensor), f'Label argument must be of type {torch.Tensor}. Got {label.__class__.__name__}'
        
        if self._dtype and self._dtype != label.dtype:
            label = label.type(self._dtype)
        dtype = label.dtype

        mask = torch.zeros(label.size()[-2], label.size()[-1], dtype= dtype)

        for color in self._colors:
            idx = (label == torch.tensor(color, dtype= dtype).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(-3) == 3)          
            mask[validx] = torch.tensor(self._colors[color], dtype= dtype)

        return mask

    # TODO
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class IndexMasktoColorMask(object):
    """Class to transform index numpy masks to RGB/GreyScalewith the specified color mapping.
    
    Args:
    """

    def __init__(self, colors:typing.MutableMapping, dtype:torch.dtype = torch.uint8):
        self._colors = colors
        self._dtype = dtype

    def __call__(self, label:torch.Tensor, channels:int = COLOR_CHANNELS, dtype = torch.uint8):
        assert  isinstance(label, torch.Tensor), f'Label argument must be of type {torch.Tensor}. Got {label.__class__.__name__}'
         
        if self._dtype and self._dtype != label.dtype:
            label = label.type(self._dtype)
        dtype = label.dtype

        mask = torch.zeros(channels, label.size()[-2], label.size()[-1], dtype= dtype)

        for color, idx in self._colors.items():
            validx = (label == torch.tensor(idx, dtype=dtype))  
            color = torch.tensor(color, dtype=dtype).unsqueeze(1)     
            mask[:,validx] = color

        return mask

    # TODO
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

ColorMap = typing.MutableMapping[typing.Tuple[int,int,int], int]

def generate_numpy_files(
    path:typing.Union[str, Path],
    dataset:SegmentationDataset, 
    color_map:ColorMap,
    index_name:bool = True, 
    ):
        """Generate numpy files containing segmentation masks from PIL images

        :param path: Output path for the numpy files.
        :type path: :class:`str` or :class:`pathlib.Path`
        :param dataset: Dataset with labels as color images. It must implement the method :meth:`load_label(idx)` to retriv
        
        """
        # Create directory
        os.makedirs(path, exist_ok=True)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            ColorMasktoIndexMask(colors=color_map)
        ])


        for idx in tqdm(range(len(dataset))):

            # Path
            file_name = f'{idx:03d}.npy' if index_name else f'{os.path.basename(dataset.annotations[idx])}.npy'
            dir_name = os.path.join(path, 'label_numpy')
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            path_name = os.path.join(dir_name, file_name)

            # Transformation
            label = dataset.load_label(idx)
            mask = transform(label)
            
            # Save tensor
            np.save(path_name, mask.numpy())