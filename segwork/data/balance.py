from abc import ABC, abstractmethod


class WeightCalculator(ABC):
    """Base class for class weight calculator"""
    
    @abstractmethod
    def calculate(self, *args, **kwrags):
        raise NotImplementedError

class MedianFrequencyWeight(WeightCalculator):
    """Weight calculator base on weight"""
    def calculate(self, dataset, *args, **kwrags):
        pass
