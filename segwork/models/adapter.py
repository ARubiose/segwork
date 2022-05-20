
from ast import Import
import logging


# Logger config
logger = logging.getLogger('Segwork.adapter')

def get_timm_backbone( name:str, **kwargs ):
    """Adapter for timm backbones"""

    try:
        import timm
        return timm.create_model( model_name = name, features_only = True, **kwargs)

    except ImportError:
        raise ImportError('Timm is not installed')

    pass