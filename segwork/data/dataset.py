"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/vision/stable/io.html#image
"""

from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_image



class WeightCalculator:
    pass


class SegmentationDataset(VisionDataset):
    """Common interface to describe segmentation datasets
    Args:
        VisionDataset (_type_): _description_
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        download_fn: Callable = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        # split
    ):
        super().__init__(root, transforms, transform, target_transform)
    
    def get_image(self, index:int):
        pass

    def get_mask(self, index:int):
        pass

    def classes(self):
        pass
    
    def num_classes(self):
        pass

    @abstractmethod
    def mask_colors(self):
        pass

    @property
    def classes(self):
        pass

    def compute_class_weights(self, calculator:WeightCalculator):
        pass


