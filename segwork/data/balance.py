from abc import ABC, abstractmethod
import logging
import os
import pathlib

import typing
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

class WeightCalculator(ABC):
    """Base class to count pixels in an image
    
    Args:
    """
    def __init__(self, 
        num_classes:int,
        path:typing.Union[str, Path] = None,
        ignore_index: typing.Tuple[int, ...] = None):
        self._num_classes = num_classes
        self._path = path
        self._ignore_index = ignore_index if ignore_index or ignore_index == (0,) else tuple()
        self._initialize_pixel_count()
        self._initialize_class_count()

    @property
    def pixel_count(self):
        """Returns the data structure with the pixel count
        
        The pixel count represents the total number of pixel of i"""
        return self._pixel_count

    @property
    def class_count(self):
        """Returns the data structure with the class count.
        
        The class count represents the total number pixel in images where *i* is present"""
        return self._class_count

    @property
    def weights(self):
        """Returns the weights"""
        return self._weights

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def ignore_index(self) -> typing.Tuple[int,...]:
        return self._ignore_index

    @num_classes.setter
    def num_classes(self, n: int):
        assert not self.is_empty(), \
            f'Values are non zero, to change the number of classes reset the calculator'
        self.num_classes = n
        self._initialize_pixel_count()

    @abstractmethod
    def load_weights(self, *args, **kwargs):
        """Update object status en returns the loaded weights
        
        Raise:
            FileNotFoundError
        """
        raise NotImplementedError

    @abstractmethod
    def save_weights(self, *args, **kwargs):
        """Saves weights in a file.
        
        """
        raise NotImplementedError

    @abstractmethod
    def reset_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _initialize_pixel_count(self):
        raise NotImplementedError

    @abstractmethod
    def _initialize_class_count(self):
        raise NotImplementedError

    @abstractmethod
    def is_empty(self):
        """Return whether the calculator is empty or not"""
        raise NotImplementedError

    @abstractmethod
    def update(self, label):
        """Updates the state of the calculator
        
        It will update the number of pixels per class"""
        raise NotImplementedError

    def __repr__(self):
        return f'Calculator with \n\t{self.num_classes} classes. Igonre_index = {self.ignore_index}\
        \n\tPixel count:{self.pixel_count}. Class count:{self.class_count}'

class NumpyCalculator(WeightCalculator):
    """Base class for class weight calculator with numpy"""

    def __init__(self, dtype: np.dtype = None, *args, **kwargs):
        self._dtype = dtype if dtype else np.int64
        super().__init__(*args, **kwargs)
        self.reset_weights()
 
    def _initialize_pixel_count(self) -> np.ndarray:
        self._pixel_count = np.zeros(self.num_classes, dtype=self._dtype)
        return self.pixel_count

    def _initialize_class_count(self) -> np.ndarray:
        self._class_count = np.zeros(self.num_classes, dtype=self._dtype)
        return self._class_count

    def is_empty(self) -> bool:
        """Return whether the calculator is empty or not"""
        return not self.pixel_count.any()

    def update(self, label, ignore_index:typing.Tuple[int, ...] = None):
        """Updates the total count of pixels per class"""

        if ignore_index:
            self._ignore_index = ignore_index

        ignore_index_weights = np.array([0 if index in self._ignore_index else 1 for index in range(self.num_classes)])

        h, w = label.shape

        label_count = np.bincount(label.astype(self._dtype).flatten(), minlength=self.num_classes)
        label_count *= ignore_index_weights
        self._pixel_count = np.add(self.pixel_count, label_count)

        class_count = label_count > 0
        self._class_count = np.add(self.class_count, class_count * h * w)
        return label_count, class_count

    def load_weights(self, path:typing.Union[str, Path] = None, *args, **kwargs):
        
        if path:
            self._path = path

        try:
            with open(self._path, 'rb') as f:
                self._weights = np.load(f, *args, **kwargs)
        except Exception as e:
            logger.error(f'Weights could not be loaded. {e}')
            raise e

        return True 

    def save_weights(self, path:typing.Union[str, Path] = None, unique:bool=False, *args, **kwargs):
        
        if path:
            self._path = path

        if os.path.exists(self._path):
            if unique:
                raise FileExistsError
            logger.warning('Weight file {self._path} already exists, replacing weights. Change the unique attr to prevent it.')
            os.remove(self._path)

        self._save_weights()
        
    def _save_weights(self, *args, **kwargs):
        try:
            with open(self._path, 'wb') as f:
                np.save( f, self._weights *args, **kwargs)

        except Exception as e:
            logger.error(f'Weights could not be saved. {e}')
            raise e

        return True

    def reset_weights(self, *args, **kwargs):
        self._weights = np.zeros(self.num_classes, dtype=self._dtype)
        return self._weights

    def calculate(self):
        """Calculate the weights. Overried this"""
        raise NotImplementedError


class NumpyLinearWeight(NumpyCalculator):
    """Weight calculator base on Median frecuency"""

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        self._weights = np.exp(self.pixel_count)/sum(np.exp(self.pixel_count))
        return self._weights

class NumpyMedianFrequencyWeight(NumpyCalculator):
    """Weight calculator base on Median frecuency"""

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        frequency = self.pixel_count / self.class_count
        self._weights = np.median(frequency) / frequency 
        return self._weights

class LogarithmicWeight(NumpyCalculator):
    """Weight calculator based on logarithm """

    def __init__(self, c:float = 1.02, *args, **kwargs):
        self._c = c
        super().__init__(*args, **kwargs)

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        logits = self.pixel_count / np.sum(self.pixel_count)
        self._weights = 1 / np.log(self._c + logits) 
        return self._weights


class PytorchCalculator(WeightCalculator):
    """Base class for class weight calculator with numpy"""
    
    def __init__(self, dtype: np.dtype = None, *args, **kwargs):
        self._dtype = dtype if dtype else torch.uint64
        super().__init__(*args, **kwargs)

    def _initialize_pixel_count(self):
        self._pixel_count = torch.zeros(self.num_classes, dtype=self._dtype)
        return self.pixel_count

    def _initialize_class_count(self):
        self._class_count = torch.zeros(self.num_classes, dtype=self._dtype)
        return self._class_count

    def is_empty(self) -> bool:
        """Return whether the calculator is empty or not"""
        return not torch.any(self.pixel_count)

    def update(self, label):
        """Updates the total count of pixels per class"""
        
        label_count = torch.bincount(torch.flatten(label), minlength=self.num_classes)
        self._pixel_count = torch.add(self.pixel_count, label_count)

        class_count = label_count > 0
        self._class_count = torch.add(self.class_count, class_count)
        return label_count, class_count

    def load_weights(self, *args, **kwargs):
        self._weights = torch.load(self._path, *args, **kwargs)
        return self._weights

    def save_weights(self, *args, **kwargs):
        torch.save( self._weights, *args, **kwargs)

    def reset_weights(self, *args, **kwargs):
        self._weights = torch.zeros(self.num_classes, dtype=self._dtype)
        return self._weights

    def calculate(self):
        """Calculate the weights. Overried this"""
        raise NotImplementedError


class PytorchLinearWeight(PytorchCalculator):
    """Weight calculator base on Median frecuency"""

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        self._weights = torch.exp(self.pixel_count)/torch.sum(torch.exp(self.pixel_count))
        return self._weights

class PytorchMedianFrequencyWeight(PytorchCalculator):
    """Weight calculator base on Median frecuency"""

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        frequency = self.pixel_count / self.class_count
        self._weights = torch.median(frequency) / frequency 
        return self._weights

class PytorchLogarithmicWeight(PytorchCalculator):
    """Wight calculator based on logarithm """

    def __init__(self, c:float = 1.02, *args, **kwargs):
        self._c = c
        super().__init__(*args, **kwargs)

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        logits = self.pixel_count / torch.sum(self.pixel_count)
        self._weights = 1 / torch.log(self._c + logits) 
        return self.weights