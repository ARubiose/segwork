from typing import Tuple, Union
import torch.nn as nn

def create_upconv2d(in_channels:int, 
                    out_channels:int, 
                    kernel_size:Union[int, Tuple] = 2,
                    stride:Union[int, Tuple] = 2,
                    **kwargs) -> nn.Module:
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, **kwargs)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels:int, classes:int) -> None:
        super(SegmentationHead, self).__init__()

    def forward(self, x):
        pass

class SegmentationHead(nn.Module):

    def __init__(self, in_channels:int, classes:int) -> None:
        super(SegmentationHead, self).__init__()

    def forward(self, x):
        pass