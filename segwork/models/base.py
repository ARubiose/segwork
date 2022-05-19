from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import torch

import torch.nn as nn

from segwork.models.backbones.base import SegmentationBackbone
from segwork.models.registry import RegisteredModel

@dataclass
class BaseSegmenter(RegisteredModel):
    """Base segmentation model

    """
    feature_encoder:SegmentationBackbone
    segmentation_head:nn.Module
    segmentation_neck:nn.Module = torch.nn.Identity()

    def forward(self, data) -> torch.Tensor:
        features = self.feature_encoder(data)
        score_map = self.segmentation_neck(features)
        return self.segmentation_head(score_map)

    @abstractmethod
    def initialize(self) -> None:
        """Initialize model weights"""
        pass

def create_segmentation_model(name:str)-> nn.Module:
    """Factory method for off-the-shelf segmentation models

    Args:
        name (str): _description_
        ...

    Returns:
        nn.Module: _description_
    """
    pass