from abc import abstractmethod
import typing

import torch
import torch.nn as nn

class BackboneMixin:
    """Common interface to describe backbones / Feature extractors"""

    @property
    @abstractmethod
    def stage_width(self) -> typing.Tuple[int, ...]:
        """Get width of the output of stages"""
        pass

    @property
    @abstractmethod
    def stage_reduction(self) -> typing.Tuple[int, ...]:
        """Get resolution reduction of the output of stages"""
        pass

    @property
    @abstractmethod
    def output_stride(self) -> typing.Tuple[int, ...]:
        """Output stride of the backbone"""
        pass

    @property
    def depth(self):
        """Depth or number of stages of the backbone"""
        return len(self.stage_width)

    def load_weights(self, weights, key:str = None, *args):
        """Load weights"""
        if isinstance(weights, typing.Dict):
            self._load_weights_from_dict(weights)
        else:
            self._load_weights_from_path(weights, key)

    def _load_weights_from_dict(self, weights):
        """Load weights from dict"""
        self.load_state_dict(weights)

    def _load_weights_from_path(self, path, key:str = None):
        """Load weights from path"""
        checkpoint = torch.load(path)
        if key:
            self.load_state_dict(checkpoint[key])
        else:   
            self.load_state_dict(checkpoint)




