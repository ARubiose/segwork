from dataclasses import dataclass

import torch


@dataclass
class ConfusionMatrix:
    """Confusion matix to calculate metrics based on type I and type II errors"""

    TP : int = 0
    TN : int = 0
    FP : int = 0
    FN : int = 0

    def update(self, ground_truth :torch.Tensor, mask:torch.Tensor, argmax:bool = False, **kwargs) -> None:
        
        pass

    def accuracy(self):
        pass

    def iou(self):
        pass