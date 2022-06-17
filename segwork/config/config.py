"""Base config file using YACS

Types of hyper-parameters
    * Dataset 
    * Model
    * Optimizer
    * Learning rate policy
    * Augmentation and regularization 
    * Misc
"""

from pathlib import Path
from typing import Union
import yaml

from yacs.config import CfgNode as ConfigurationNode

# YACS base config file
_C = ConfigurationNode() 
_C.NAME = "Default configuration name"

_C.DATASET = ConfigurationNode()
_C.DATASET.KEY = "drone"
_C.DATASET.KEY_VAL = "drone"

_C.DATALOADER = ConfigurationNode()
_C.DATALOADER.BATCH_SIZE = 2
_C.DATALOADER.VAL_BATCH_SIZE = 2
_C.DATALOADER.WORKERS = 0

_C.DATASET.TRANSFORM = ConfigurationNode()
_C.DATASET.TRANSFORM.TRANSFORM = 'transformationA' 
_C.DATASET.TRANSFORM.TRANSFORM_ARGS = ConfigurationNode()
_C.DATASET.TRANSFORM.TRANSFORM_ARGS.height = 256
_C.DATASET.TRANSFORM.TRANSFORM_ARGS.width = 512
_C.DATASET.TRANSFORM.TARGET_TRANSFORM = 'transformationA'
_C.DATASET.TRANSFORM.TARGET_TRANSFORM_ARGS = ConfigurationNode()
_C.DATASET.TRANSFORM.TARGET_TRANSFORM_ARGS.height = 256
_C.DATASET.TRANSFORM.TARGET_TRANSFORM_ARGS.width = 512 

_C.MODEL = ConfigurationNode()
_C.MODEL.KEY = "unet"
_C.MODEL.ARGS = ConfigurationNode()
_C.MODEL.ARGS.encoder_name = "resnet34"
_C.MODEL.ARGS.classes = 24
_C.MODEL.DEVICE = "cuda"


_C.OPTIM = ConfigurationNode()
_C.OPTIM.KEY = 'sgd'
_C.OPTIM.ARGS = ConfigurationNode()
_C.OPTIM.ARGS.lr = 0.1
_C.OPTIM.ARGS.momentum = 0.9

_C.OPTIM.LOSS = ConfigurationNode()
_C.OPTIM.LOSS.KEY = 'crossentropyloss'
_C.OPTIM.LOSS.ARGS = ConfigurationNode()

_C.LOGGER = ConfigurationNode()
_C.LOGGER.LOG_INTERVAL = 10

_C.TRAIN = ConfigurationNode()
_C.TRAIN.MAX_EPOCHS = 100

_C.VALIDATION = ConfigurationNode()
_C.VALIDATION.METRICS_KEYS = ['loss']
_C.VALIDATION.ARGS = ConfigurationNode()
_C.VALIDATION.ARGS.loss = ConfigurationNode()


def _get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def get_experiment_cfg(path:Union[str, Path], freeze: bool = False):
    """Takes a path an returns a CfgNode object"""

    base_cfg = _get_cfg_defaults()
    base_cfg.merge_from_file(path)

    if freeze:
        base_cfg.freeze()

    return base_cfg

def save_experiment_cfg(cfg: ConfigurationNode, path:Union[str, Path]):
    """Saves de experiment configuration from the specified configuration node to the specified path"""
    with open(path, 'w') as f:
        yaml.dump(cfg, f)

