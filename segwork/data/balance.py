from abc import ABC, abstractmethod
import typing
import os
import pathlib

import numpy as np

from segwork.data.dataset import SegmentationDataset

class WeightCalculator(ABC):
    """Base class for class weight calculator"""
    def __init__(self, 
        path: typing.Union[str, pathlib.Path], 
        dataset: SegmentationDataset,
        weight_loader:typing.Callable = np.load):
        self._path = path
        self._weight_loader = weight_loader
        self._dataset = dataset
        self.frequency_dict = {}

    def load_weights(self, *args, **kwargs):
        if os.path.exists(self._path):
            return self._weight_loader(self._path)
        return self.calculate(*args, **kwargs)

    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        pass

class MedianFrequencyWeight(WeightCalculator):
    """Weight calculator base on Median frecuency"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate(self, *args, **kwargs):
        """Calculate weights based on Median frequency"""
        for idx , item in self._dataset:
            label = self._dataset.load_mask(idx)
            



class LogarithmicWeight(WeightCalculator):
    """Wight claculater based on logarithm """
    pass


