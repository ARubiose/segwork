import os
from PIL.Image import Image

import torchvision
import numpy as np

import segwork.data as segdata

class VOCSegmentationWrapper(torchvision.datasets.VOCSegmentation, segdata.SegmentationDataset):
    """Pascal VOC wrapper"""

    def load_image(self, idx:int):
        return Image.open(self.images[idx]).convert("RGB")

    def load_label(self, idx:int):
        return Image.open(self.targets[idx]).convert("RGB")

    @property
    def mask_colors(self):
        NotImplementedError

    @property
    def mask_colors_index(self):
        return { key : idx for idx, key in enumerate(self.mask_colors)}

    @property
    def num_classes(self):
        return len(self.mask_colors)

    @property
    def classes(self):
        return list(self.mask_colors.values())

    def load_numpy_label(self, idx:int, *args, **kwargs):
        """Return a :py:class:`numpy.ndarray` with the label for the specified idx"""
        NotImplementedError
