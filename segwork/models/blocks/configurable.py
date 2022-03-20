from abc import ABC, abstractmethod
from typing import Callable, Optional
from torch import Tensor

import torch.nn as nn
from omegaconf import DictConfig

class ConfigurableBlock(ABC):
    """Wrapper Abstract class for configurable modules"""

    def __init__(self, cfg:DictConfig, block:nn.Module, *args, **kwargs):
        self.cfg = cfg
        self._validate_configuration() 
        self.block = block(**self.cfg)

    def __call__(self, *args, **kwargs):
        return self.block.forward(*args, **kwargs)

    @abstractmethod
    def _validate_configuration(self):
        pass