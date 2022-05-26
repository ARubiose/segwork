from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import torch

import torch.nn as nn

from segwork.model.backbone.base import SegmentationBackbone



class BaseSegmenter(nn.Module):
    """Base segmentation model

        To provide flexibility ":py:class:`nn.Module`" hooks are provided at 
        the beggining :py:attr:`initial_block`, and at the end :py:attr:`segmentation_head` 
    """

    def __init__(self, 
        feature_encoder:SegmentationBackbone,
        segmentation_head:nn.Module , 
        aux_classifier : nn.Module = None) -> None:

        super(BaseSegmenter, self).__init__()
        self.feature_encoder = feature_encoder
        self.segmentation_head = segmentation_head
        
        self.aux_classifier = aux_classifier

    
    def forward(self, input) -> torch.Tensor:
        """Default forward method of segmentation model"""

        features = self.feature_encoder(input)

        if self.aux_classifier:
            return self.aux_classifier(features['out'])

        prediction = self.segmentation_head(features)

        return prediction

    @abstractmethod
    def initialize(self) -> None:
        """Initialize model weights"""
        pass

    def __repr__(self):
        return f'Model name: {self._registry_name}'