from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from timm.models import layers as timm_layers

from segwork.models.backbones.base import SegmentationBackbone
from segwork.models.decoders.base import SegmentationDecoder

# Classifier head -> from timm.models.layers.classifier import ClassifierHead

class SegmentationHead(nn.Module):
    """Basic segmentation head
    
    #TODO Add output size"""
    def __init__(self, 
                in_channels:int, 
                num_classes:int,
                kernel_size:int = 3,
                upsample:Callable = None,
                act:Callable = None) -> None:
        super(SegmentationHead, self).__init__()
        self.conv = timm_layers.create_conv2d(in_channels, num_classes, kernel_size)
        self.upsample = upsample
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.upsample:
            x = self.upsample(x)
        if self.act:
            x = self.act(x)
        return x

class EncoderDecoder(nn.Module):
    """Base skeleton of encoder decoder segmentation models

    Mostly all models follow the encoder-decoder architecture

    Inspirated in: _SimpleSegmentationModel from Torchvision
    All optional but backbone to be able to train backbone with aux_classifier

    Args:
        backbone (nn.Module): _description_
        decoder (nn.Module): _description_
        segmentation_head (nn.Module): _description_
        aux_classifier (nn.Module): _description_
    """
    def __init__(self, 
                encoder: Union[SegmentationBackbone, nn.Module] = None,
                decoder: Union[SegmentationDecoder, nn.Module] = None,
                segmentation_head: Optional[nn.Module] = None, # Support multiple heads?
                aux_classifier: Optional[nn.Module] = None) -> None:        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = segmentation_head
        self.aux_classifier = aux_classifier

    def forward(self, x):
        
        # Features is a list of features at different scales.
        features = self.encoder(x)

        if self.aux_classifier is not None:
            x = features[-1]
            x = self.aux_classifier(x)
            return x
        
        out = self.decoder(features)
        out = self.segmentation_head(out)
        return out

    @property
    def features_info(self):
        return self.encoder.features_info

def create_segmentation_model(name:str)-> nn.Module:
    """Factory method for off-the-shelf segmentation models

    Args:
        name (str): _description_
        ...

    Returns:
        nn.Module: _description_
    """
    pass