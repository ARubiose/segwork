import logging
from typing import Tuple

import torch.nn as nn
import timm

MODEL_NAME_FORMAT = 'source/model'

def get_timm_backbone(name:str, **kwargs) -> nn.Module:
    try:
        backbone = timm.create_model(model_name=name, features_only=True, **kwargs)
    except:
        raise NameError(f'{name} not found.')
    return backbone

def parse_model_name(name:str) -> Tuple[str, str]:
    """Parse fully qualified name for model to (source, model_name)"""
    name_tokens = name.split('/')

    # Own encoder
    if len(name_tokens) == 1:
        return ('', name_tokens)

    # Vendor encoder
    if len(name_tokens) == 2:
        return (name_tokens[0], name_tokens[1])

    raise NameError('Model name should follow format {MODEL_NAME_FORMAT}')