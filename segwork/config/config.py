"""Base config file using YACS

Types of hyper-parameters
    * Dataset 
    * Model
    * Optimizer
    * Learning rate policy
    * Augmentation and regularization 
    * Misc
"""

import logging
import os
from pathlib import Path
from typing import Union
from yacs.config import CfgNode as ConfigurationNode

# Default logger
logger = logging.getLogger('Configuration')

class ConfigManager():
    """Base class for config manager"""

    BASE_CFG: ConfigurationNode

    def __init__(self):
        pass

# YACS base config file
_C = ConfigurationNode()

# Logger
_C.LOGGER = ConfigurationNode()
_C.LOGGER.FORMAT = ''


# Data augmentation hyper-parameters
_C.AUGMENTATION = ConfigurationNode()

# Model backbone configuration
_C.BACKBONE = ConfigurationNode()
_C.BACKBONE.NAME = 'RegisteredBackbone'


# Head model configuration
_C.HEAD = ConfigurationNode()

# Training configuration
_C.TRAINING = ConfigurationNode()

# Validation configuration - Metrics
_C.VALIDATION = ConfigurationNode()

def _get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def get_experiment_cfg( path:Union[str, Path], set_logger:bool = True):
    """Takes a path an returns an experiment configuration"""

    # Set logger
    if set_logger:
        logger = logging.getLogger(os.path.basename(path))

    base_cfg = _get_cfg_defaults()

    if os.path.exists(path):
        base_cfg.merge_from_file(path)
    else:
        logger.error(f'Configuration file {path} not found. Using default configuration')

    return base_cfg

