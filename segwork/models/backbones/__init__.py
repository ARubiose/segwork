import logging
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

def get_timm_backbone(name:str, **kwargs) -> nn.Module:
    try:
        backbone = timm.create_model(model_name=name, features_only=True, **kwargs)
    except:
        logger.error(f'Backbone not available: {name}')
        
    return backbone