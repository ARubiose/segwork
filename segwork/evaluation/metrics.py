from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ConfusionMatrix:
    """Confusion matix to calculate metrics based on type I and type II errors"""

    TP : int = 0
    TN : int = 0
    FP : int = 0
    FN : int = 0


    def update(self, 
        ground_truth :torch.Tensor, 
        mask:torch.Tensor, 
        transform: Callable,
        transform_mask: Callable,
        **kwargs) -> None:
        """Update matrix
        
        Args:
            (Tensor) ground_truth : Ground truth label. Size Bx1xHxW
            (Tensor) mask : Output of the segmentation model. Size BxCxHxW

        """

        
        pass

    def accuracy(self):
        pass

    def iou(self):
        pass