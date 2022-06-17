"""https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/vision/stable/io.html#image

Compatible with torchvision
"""

import abc
import os
from pathlib import Path
import pathlib
import typing


import torchvision
from tqdm import tqdm
import numpy as np

from segwork.data.balance import WeightAlgorithm
from segwork.data.augmentations import ColorMasktoIndexMask

ColorMap = typing.MutableMapping[typing.Tuple[int,int,int], int]

class SegmentationDataset(torchvision.datasets.VisionDataset, metaclass=abc.ABCMeta):
    """Common interface to describe segmentation datasets for supervised training


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

    @abc.abstractmethod
    def load_image(self, idx:int):
        """Returns a :py:class:`PIL.Image.Image` object for the specified image idx """
        raise NotImplementedError

    @abc.abstractmethod
    def load_label(self, idx):
        """Returns a :py:class:`PIL.Image.Image` object for the specified label idx"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def images(self):
        """Returns a list of paths to the files containing the images"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def annotations(self):
        """Returns a list of paths to the files containing the ground truth masks"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mask_colors(self):
        """Returns a mapping object of the class index and class colors"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def classes(self):
        """Returns a list of the classes"""
        raise NotImplementedError

    @property
    def num_classes(self):
        """Returns the number of classes"""
        return len(self.classes)

    @property
    def num_data_points(self):
        """Returns the number of datapoints in the dataset"""
        assert len(self.images) == len(self.annotations), \
        f'Number of images ({len(self.images)}) must be equal to number of labels ({len(self.annotations)})'

        return len(self.images)

    def compute_class_weights(self, 
        weight_algorithm:WeightAlgorithm, 
        path: typing.Union[str, pathlib.Path] = None, *args, **kwargs):
        """Basic method to calculate dataset weights.
        
        Make sure your calculator input and your dataset output are compatible through the target_transform attribute"""

        if path:
            weight_algorithm.pixel_counter.load_counters(path)
        else:
            for idx in tqdm(range(self.num_data_points)):
                label = self.load_weight_label(idx)
                weight_algorithm.update(label)

        weights = weight_algorithm.compute(*args, **kwargs)
        return weights

    @abc.abstractmethod
    def load_weight_label(self, idx):
        """Load label to be used by the calculator"""
        raise NotImplementedError


def generate_numpy_files(self, 
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




    
        


