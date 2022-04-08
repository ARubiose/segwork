from typing import Tuple, Union
import torch.nn as nn

from segwork.models.layers.base import create_upconv2d

# Useful?
class SegmentationDecoder:
    """Common interface to describe decoders
    
    Adapt for Multi-Scale inference
"""
    pass


