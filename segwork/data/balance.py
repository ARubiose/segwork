import abc
import logging
import os
import pathlib

import typing
import pathlib

import numpy as np

_logger = logging.getLogger(__name__)

class PixelCounter(abc.ABC):
    """Base class to count pixels in an image
    
    Args:
        num_classes (int): Number of classes of the dataset.
    """
    def __init__(self, num_classes:int):
        self._num_classes = num_classes
        self._initialize_counters()

    @property
    def pixel_count(self):
        """Returns the data structure with the pixel count
        
        The pixel count represents the total number of pixel of class `c`"""
        return self._pixel_count

    @property
    def class_count(self):
        """Returns the data structure with the class count.
        
        The class count represents the total number of pixel in images where class `c` is present"""
        return self._class_count

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, n: int):
        assert isinstance(n, int), f'n  must be an int. Got {n.__class__.__name__}'
        if not self.is_empty():
            _logger.warning(f'Counters are not empty, resetting counters with new number of classes: {n}')
        self.num_classes = n
        self.reset_counters()

    @abc.abstractmethod
    def save_counters(self, path:typing.Union[pathlib.Path, str], *args, **kwargs):
        """Saves counters in a file.
        
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _initialize_counters(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def load_counters(self, path:typing.Union[pathlib.Path, str], *args, **kwargs):
        """Update object status en returns the loaded counters
        
        Raise:
            FileNotFoundError
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_empty(self):
        """Return whether the calculator is empty or not"""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, label):
        """Updates the state of the pixel counters
        
        It will update the number of pixels per class"""
        raise NotImplementedError

    def __repr__(self):
        return f'Counter of pixels with {self.num_classes} classes.\n'\
        f'Pixel count:\n{self.pixel_count}.\nClass count:\n{self.class_count}'

class NumpyPixelCounter(PixelCounter):
    """
    Implementation of PixelCounter with numpy.

    Args:
        dtyp (numpy.dtype) : dtype of the ndarrays to be used for the operations.
    
    """

    def __init__(self, dtype: np.dtype = None, *args, **kwargs):
        self._dtype = dtype if dtype else np.float32
        super().__init__(*args, **kwargs)
 
    def _initialize_counters(self) -> np.ndarray:
        self._pixel_count = np.zeros(self.num_classes, dtype=self._dtype)
        self._class_count = np.zeros(self.num_classes, dtype=self._dtype)
        return self.pixel_count, self._class_count

    def is_empty(self) -> bool:
        """Return whether the calculator is empty or not"""
        return not self.pixel_count.any()

    def update(self, label: np.ndarray):
        """Updates the total count of pixels per class"""
        h, w = label.shape

        label_count = np.bincount(label.astype(self._dtype).flatten(), minlength=self.num_classes)
        self._pixel_count = np.add(self.pixel_count, label_count)

        class_count = label_count > 0
        self._class_count = np.add(self.class_count, class_count * h * w)

        return self.pixel_count, self.class_count

    def load_counters(self, path:typing.Union[str, pathlib.Path], *args, **kwargs):
        try:
            with open(path, 'rb') as f:
                counters = np.load(f, *args, **kwargs)
                self._pixel_count = counters['pixel_count']
                self._class_count = counters['class_count']
                _logger.info(f'Pixel counts loaded from {path}')
        except Exception as e:
            _logger.error(f'Weights could not be loaded. {e}')
            raise e

        return True 

    def save_counters(self, path:typing.Union[str, pathlib.Path], exist_ok:bool=True, *args, **kwargs):

        if os.path.exists(path):
            if not exist_ok:
                raise FileExistsError
            _logger.warning(f'Weight file {path} already exists, replacing file. Pass exist_ok=False attr to prevent it.')

        self._save_counters(path)
        
    def _save_counters(self, path, *args, **kwargs):
        try:
            with open(path, 'wb') as f:
                np.savez( f, pixel_count=self.pixel_count, class_count=self.class_count, *args, **kwargs)
        except Exception as e:
            _logger.error(f'Weights could not be saved. {e}')
            raise e

        return True

    def reset_counters(self, *args, **kwargs):
        self._initialize_counters()

class WeightAlgorithm(abc.ABC):
    """
    Abstract base class to represent algortihm that calculate class weights
    
    Args:
        pixel_counter (PixelCounter): PixelCounter object use to calculate the weights
    """
    def __init__( self, pixel_counter:PixelCounter):
        self._pixel_counter = pixel_counter

    @property
    def pixel_counter(self):
        return self._pixel_counter

    @pixel_counter.setter
    def pixel_counter(self, pixel_counter: PixelCounter):
        assert isinstance(pixel_counter, PixelCounter), f'pixel_counter must be an object of a subclass of PixelCounter'
        self._pixel_counter = pixel_counter

    @property
    def weights(self):
        if self._weights:
            return self._weights
        self.compute()

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """Compute weights"""
        raise NotImplementedError

    def update(self, label, *args, **kwargs):
        self.pixel_counter.update(label, *args, **kwargs)


class LinearWeight(WeightAlgorithm):
    """Linear weight algorithm for calculation of weights"""

    def compute(self, *args, **kwargs):
        """Compute method to calculate weights based on a linear relation"""
        weights = self.pixel_counter.pixel_count
        return weights

class NumpyMedianFrequencyWeight(WeightAlgorithm):
    """Median frequency algorithm for calculation of weights
    
    https://arxiv.org/pdf/1411.4734.pdf"""

    def compute(self, *args, **kwargs):
        """Compute method to calculate weights based on Median frequency"""
        frequency = np.divide(self.pixel_counter.pixel_count, self.pixel_counter.class_count)
        weights = np.divide(np.median(frequency), frequency) 
        self._weights = weights
        return weights

class LogarithmicWeight(WeightAlgorithm):
    """Logarithmic algorithm for calculation of weights"""

    def __init__(self, c:float = 1.02, *args, **kwargs):
        self._c = c
        super().__init__(*args, **kwargs)

    def calculate(self, *args, **kwargs):
        """Compute method to calculate weights based on logarithms"""
        logits = np.divide(self.pixel_count, np.sum(self.pixel_count))
        weights = np.divide(1, np.log(self._c + logits))
        self._weights = weights
        return weights