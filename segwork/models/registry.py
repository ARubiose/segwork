from abc import  abstractmethod
import logging
import torch.nn as nn

# Logger
logger = logging.getLogger('registry')

class BaseRegistry(type):
    """Meta class that defines core behaviour: Created classes are added to the registry"""

    def __new__(cls, name, bases, attrs, **kwargs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        return new_cls

    @classmethod
    @property
    @abstractmethod
    def REGISTRY(cls):
        raise NotImplementedError

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)

    @classmethod
    def add_register(cls, cls_entry):
        cls.REGISTRY.update({cls_entry.__name__:cls_entry})

# Model registry
class ModelRegistry(BaseRegistry):
    """Registry for models"""
    REGISTRY = {}

class RegisteredModel(nn.Module,  metaclass = ModelRegistry):
    """"""
    pass

def register_model(cls):
    """Register model in the registry"""
    if not isinstance(cls, nn.Module):
        logger.warning(f'Cannot register class {cls.__name__}. cls must be a subclass of nn.Module')
    ModelRegistry.add_register(cls)
    return cls

# Backbone registry
class BackboneRegistry(BaseRegistry):
    """Registry for backbones"""
    REGISTRY = {}

class RegisteredBackbone(nn.Module,  metaclass = BackboneRegistry):
    """"""
    pass

# Model registry
class ModelRegistry(BaseRegistry):
    """Registry for models"""
    REGISTRY = {}

class RegisteredModel(nn.Module,  metaclass = ModelRegistry):
    """"""
    pass

# Module registry
class ModuleRegistry(BaseRegistry):
    """Registry for modules"""
    REGISTRY = {}

class RegisteredModule(nn.Module,  metaclass = ModuleRegistry):
    """"""
    pass



def get_model( name: str , *args, **kwargs) -> nn.Module:
    """Get model from the registry"""
    return ModelRegistry.get_registry()[name](*args, **kwargs)

def get_backbone( name: str , *args, **kwargs) -> nn.Module:
    """Get module from the registry"""
    return BackboneRegistry.get_registry()[name](*args, **kwargs)
