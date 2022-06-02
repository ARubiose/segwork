"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/vision/stable/io.html#image
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

# from segwork.data.balance import WeightCalculator

class SegmentationDataset(VisionDataset):
    """Common interface to describe segmentation datasets
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)

    def load_image(self, idx:int):
        """Returns image"""
        pass

    def load_mask(self, idx):
        """Returns mask"""
    
    def classes(self):
        pass
    
    def images(self):
        """Return a list with the names of the files containing the images"""
        pass

    def annotations(self):
        """Returns a list with the names of the files containing the ground truth masks"""
        pass

    @abstractmethod
    def num_classes(self):
        pass

    @abstractmethod
    def mask_colors(self):
        pass

    @property
    def classes(self):
        pass

    def capture_colors(self):
        pass



    # def compute_class_weights(self, calculator: WeightCalculator, *args, **kwargs):
    #     return calculator.calculate(self, *args, **kwargs)
        


