from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from zmq import device


@dataclass
class ConfusionMatrix:
    """Confusion matrix to calculate metrics based on type I and type II errors for binary and multiclass segmentation."""

    def __init__(self, size:int):
        self._size = size
        self._single_matrix = torch.zeros(size, size)
        self._accumulated_matrix = torch.zeros(size, size)
    
    @property
    def matrix(self, accumulated:bool) -> Tuple:
        """Protected matrix to keep integrity"""
        return torch.clone(self._matrix)

    def update(self, 
        ground_truth :torch.Tensor, 
        mask:torch.Tensor,
        is_probability_class:bool = True,
        **kwargs) -> None:
        """Update matrix
        
        Args:
            (Tensor) ground_truth : Ground truth label. Size Bx1xHxW
            (Tensor) mask : Output of the segmentation model. Size Bx1xHxW or BxCxHxW
            (Bool) is_probability_class : Wether the mask is a segmentation mask or probability class maps

        """
        # Check tensor allocation
        assert ground_truth.device == mask.device, \
            f'Ground truth label and mask must be in the same device.'\
                'Ground truth is on {ground_truth.device} and mask is on {mask.device}'

        # Argmax if needed
        if is_probability_class:
            mask = torch.argmax(mask, 1)

        mask = mask.view(-1) 
        ground_truth = ground_truth.view(-1) 

    def reset(self):
        self._matrix = torch.zeros(self._size, self._size)

@dataclass
class Metric(ABC):

    cm : ConfusionMatrix

    @abstractmethod
    def calculate(self):
        pass

    def calculate_mean(self):
        pass

