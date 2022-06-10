"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/vision/stable/io.html#image

Compatible with torchvision
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import typing


import torchvision
from tqdm import tqdm
import numpy as np

from segwork.data.balance import PixelCalculator
from segwork.data.augmentations import ColorMasktoIndexMask

class SegmentationDataset(torchvision.datasets.VisionDataset):
    """Common interface to describe segmentation datasets for supervised training

    Args:
        - (str) root: 
    """
    HEIGHT: int
    WIDTH: int

    def __init__(
        self,
        root: str,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        transforms: typing.Optional[typing.Callable] = None,
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

    def compute_class_weights(self, calculator:PixelCalculator, *args, **kwargs):
        """Basic method to calculate dataset weights.
        
        Make sure your calculator input and your dataset output are compatible through the target_transform attribute"""

        for idx in tqdm(range(self.num_data_points)):
            label = self.load_weight_label(idx)
            calculator.update(label)

        weights = calculator.compute(*args, **kwargs)
        return weights

    @abstractmethod
    def load_weight_label(self, idx):
        """Load label to be used by the calculator"""
        raise NotImplementedError


def generate_numpy_files(self, 
    path:typing.Union[str, Path],
    dataset:SegmentationDataset, 
    color_map:typing.MutableMapping[typing.Tuple[int,int,int], int],
    index_name:bool = True
    ):
        """Generate numpy files containing segmentation masks
        
        """

        if not os.path.exists(path):
            # logger.warning(f'{path} not found. Creating directory...')
            pass

        assert os.path.isdir(path)
        transform = torchvision.transforms.Compose([
            ColorMasktoIndexMask(colors=color_map),
            torchvision.transforms.PILToTensor()
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



    
        


