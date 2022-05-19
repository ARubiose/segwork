from dataclasses import dataclass
import logging
from typing import Tuple

import torch.nn as nn
import timm
from timm.models.features import FeatureInfo as _FeatureInfo

from segwork.models.backbones.utils import get_timm_backbone, parse_model_name

logger = logging.getLogger(__name__)

# Mixin
class SegmentationBackbone:
    """Common interface to describe backbones"""

    @property
    def output_stride(self):
        """Output stride of the network"""
        pass

    @property
    def depth(self):
        """Depth or number of stages of the backbone"""
        pass




