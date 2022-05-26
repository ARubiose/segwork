from segwork.model.backbone.base import BackboneMixin

import torch.nn as nn


def get_timm_backbone( name:str, **kwargs ):
    """Adapter for timm backbones"""

    try:
        import timm
        return timm.create_model( model_name = name, features_only = True, **kwargs)

    except ImportError:
        raise ImportError('Timm is not installed')

    pass

class TimmBackbone(nn.Module, BackboneMixin):

    def __init__(self, name:str):
        self._name = name

    

    