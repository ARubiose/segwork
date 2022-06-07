"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/vision/stable/io.html#image
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Optional, Tuple

from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from tqdm import tqdm

from segwork.data.balance import WeightCalculator

class SegmentationDataset(VisionDataset):
    """Common interface to describe segmentation datasets for supervised training

    Args:
        - (str) root: 
    """
    HEIGHT: int
    WIDTH: int

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        split:str = 'train'
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.split = split

    def __getitem__(self, idx:int):
        assert len(self.images) == len(self.annotations), \
            f'Different number of images and labels. Images: {len(self.images)} Labels:{self.labels}'
        
        image = self.load_image(idx)
        label = self.load_label(idx)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
                
        return (image, label)

    def __len__(self):
        """Returns the number of datapoints"""
        assert len(self.images) == len(self.annotations), \
            f'Different number of images and labels. Images: {len(self.images)} Labels:{self.labels}'
        return len(self.images)

    @abstractmethod
    def load_image(self, idx:int):
        """Returns a :py:class:`PIL.Image.Image` object for the specified image idx """
        raise NotImplementedError

    @abstractmethod
    def load_label(self, idx):
        """Returns a :py:class:`PIL.Image.Image` object for the specified label idx"""
        raise NotImplementedError

    @abstractmethod
    def images(self):
        """Returns a list of paths to the files containing the images"""
        raise NotImplementedError

    @abstractmethod
    def annotations(self):
        """Returns a list of paths to the files containing the ground truth masks"""
        raise NotImplementedError

    @abstractmethod
    def mask_colors(self):
        """Returns a mapping object of the class index and class colors"""
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes"""
        raise NotImplementedError

    @property
    def classes(self):
        """Returns a list of the classes"""
        raise NotImplementedError

    @property
    def num_data_points(self):
        """Returns the number of datapoints in the dataset"""
        assert len(self.images) == len(self.annotations), \
        f'Number of images ({len(self.images)}) must be equal to number of labels ({len(self.annotations)})'

        return len(self.images)

    def compute_class_weights(self, calculator:WeightCalculator, *args, **kwargs):
        """Basic method to calculate dataset weights"""

        for idx in tqdm(range(self.num_data_points)):
            label = self.load_numpy_label(idx)
            calculator.update(label)

        weights = calculator.calculate(*args, **kwargs)
        return weights
        


