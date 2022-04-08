from dataclasses import dataclass
import logging
from typing import Tuple

import torch.nn as nn
import timm
from timm.models.features import FeatureInfo as _FeatureInfo

from segwork.models.backbones.utils import get_timm_backbone, parse_model_name

logger = logging.getLogger(__name__)

#FIXME use a registry module for backbones
BACKBONES_SOURCE = {
    '':'Own backbone', # Registry as entrypoint?
    'timm':get_timm_backbone
}

def create_backbone(name:str, **kwargs) -> nn.Module:
    """Factory method for off-the-self backbones

    Args:
        name (str): _description_

    Returns:
        nn.Module: _description_
    """
    assert isinstance(name, str), f'Name must be a str. got {type(name)}'

    (source_name, model_name) = parse_model_name(name)

    return BACKBONES_SOURCE[source_name](model_name, **kwargs)

@dataclass
class FeaturesInfo(_FeatureInfo):
    """Extension for FeaturesInfo from timm.models.features """
    pass

# Experimentation
import functools

# Abstract or mixin / decorator?
def SegmentationBackbone(cls):

    @property
    def features_info(self) -> FeaturesInfo:
        return f'Some features'

    """Common interface to describe backbones

    Following timm encoders for features_only feature
    Attributes:
        in_channels
        out_channels
        output_stride(reduction) (list[int]):
        output_size
        pretrained (Imagenet)
        out_indices/depth
        return_layers
        FeaturesInfo
            num_channels
            reduction
            module
    Methods
        Freeze
    """
    pass




