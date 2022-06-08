"""
Classes for the implementation of a registry of components.

Code adapted from https://github.com/todofixthis/class-registry/blob/master/class_registry/registry.py
"""
from abc import ABC, abstractmethod
import collections
import inspect
from pathlib import Path
import typing
import copy

# TODO Logging for warning on not habing installed smp

__all__ = ['ConfigurableRegistry', 'backbones_reg', 'moduls_reg']

ItemType = typing.TypeVar('ItemType')

class Registry(collections.abc.Mapping, ABC):
    """Base class for a class registry"""

    def __iter__(self):
        """Returns a generator for iterating over registry items."""
        return self.keys()

    def __getitem__(self, key: typing.Hashable):
        """Get class associated with key.

        Args:
            key (typing.Hashable): Lookup key.
        """
        return self.get_item(key)

    @abstractmethod
    def __len__(self):
        """Get the number of registered keys.

        Returns:
            Number of registered keys.
        """
        raise NotImplementedError

    def __contains__(self, key: typing.Hashable):
        """Get wether the specified key is registered or not.

        Args:
            key (typing.Hashable): Lookup key.

        Returns
            True if key is in the register or False if not.

        """
        try:
            self.get(key)
        except KeyError:
            return False
        else:
            return True

    def keys(self) -> typing.Iterable[typing.Hashable]:
        """Get registered keys

        Returns
            A generator for iterating over registered keys.
        """
        for item in self.items():
            yield item[0]

    def items(self) -> typing.Iterable[typing.Tuple[typing.Hashable, ItemType]]:
        """Get registered classes and their asociated keys.
        
        Returns
            A generator for iterating over registered pairs (key, class).
        """
        raise NotImplementedError

    def values(self) -> typing.Iterable[ItemType]:
        """Get registered classes.

        Returns
            A generator for iterating over registered classes.
        """
        for item in self.items():
            yield item[1]

    def get(self, key: typing.Hashable) -> ItemType:
        """Get the item matching the key.
        
        Args:
            key (typing.Hashable): Lookup key.
            args: Positional arguments passed to the class constructor.
            kwargs: Keyword arguments passed to the class consturctor.

        Returns:
            Instance of the item matching the key
        """
        return self.get_item(key)

    @abstractmethod
    def get_item(self, key: typing.Hashable) -> ItemType:
        """Returns registry item with specified key"""
        raise NotImplementedError


class MutableRegistry(Registry, collections.abc.MutableMapping, ABC):
    """Extends :py:class:`Registry` with methods to modify the registered items.
    
    Support for registering classes through decorators."""

    def __init__(
            self,
            unique: bool = False,
    ) -> None:
        self.unique = unique

    def __delitem__(self, key: typing.Hashable):
        """Wrapper for :py:meth:`_unregister`. Support to dict like syntax.

        Args:
            key (typing.Hashable): Lookup key.
        """
        self._unregister(key)

    def __setitem__(self, key: typing.Hashable, cls: type):
        """Wrapper for :py:meth:`_register`. Support to dict like syntax.

        Args:
            key (typing.Hashable): Lookup key.
        """
        self._register(key, cls)

    @abstractmethod
    def _register(self, key: typing.Hashable, class_: type) -> None:
        """Registers a class with the registry."""

        raise NotImplementedError(f'Not implemented in {type(self).__name__}.')

    @abstractmethod
    def _unregister(self, key: typing.Hashable) -> type:
        """Unregisters the class at the specified key."""

        raise NotImplementedError(f'Not implemented in {type(self).__name__}.')

    def unregister(self, key: typing.Any) -> type:
        """Unregisters the class with the specified key.

        Args:
            key: Lookup key.

        Returns:
            The class that was unregistered.

        Raise:
            :py:class:`KeyError` if the key is not registered.

        """
        return self._unregister(key)



class ClassRegistry(MutableRegistry):
    """Base Class registry with register and unregister functions implemented with
    
    Maintains a registry of classes and provides a generic factory for
    instantiating them. Useful for modular components. 
    """

    def __init__(
            self,                
            unique: bool = False,
            attr_name: typing.Optional[str] = None,
    ) -> None:
        """
        Args:
            attr_name: If provided, :py:meth:`register` will automatically detect
            the key to use when registering new classes.
            unique: Determines what happens when two classes are registered with
            the same key:
            - ``True``: The second class will replace the first one.
            - ``False``: A ``ValueError`` will be raised.
        """
        super().__init__(unique)
        self.attr_name = attr_name
        self._registry = dict()

    def __len__(self):
        """
        Get the number of registered classes.

        Returns:
            Number of registered classes
        """
        return len(self._registry)

    def __repr__(self):
        return f'{type(self).__name__}(attr_name={self.attr_name}, unique={self.unique})\n\
            Number of registered classes: {self.__len__()} \n\
            Registered classes: {list(self.keys())}'
        
    def get_item(self, key: typing.Hashable, *args, **kwargs):
        """Implementation of get item."""
        return self._registry.get(key)
        
    def get_class(self, key):
        """
        Get the class associated with the specified key.  
        
        Value is not formatted. Value is directly the class without params. Same as get_item in this class

        Returns:
            Class associated with the specified key.

        Raise:
            :py:class:`KeyError` if the key is not registered.
        """
        return self._registry[key]

    def get_instance(self, key, *args, **kwargs):
        cls = self.get_class(key)
        return cls(*args, **kwargs)

    def items(self) -> typing.Iterable[typing.Tuple[typing.Hashable, str]]:
        """
        Iterates over all registered classes, in the order they were
        added.
        """
        return self._registry.items()

    def _register(self, key: typing.Hashable, cls: type) -> None:
        """
        Registers a class with the registry.
        """
        self._validate_key(key)
        self._registry[key] = cls

    def _unregister(self, key: typing.Hashable) -> type:
        """
        Unregisters the class at the specified key.

        Returns:
            Unregistered class

        Raise
            :py:class:KeyError if key is not registered
        """
        return self._registry.pop(key)

    def key_exists(self, key: typing.Hashable) -> bool:
        return key in self._registry

    def register(self, key: typing.Union[type, typing.Hashable]):
        """Decorator that registers a class with the registry.
        
        Args:
            key: The registry key to use for the registered class.
            Optional if the registry's :py:attr:`attr_name` is set.

        Raise:
            :py:class:`ValueError` if ket is not provided.

        """

        if inspect.isclass(key):
            if self.attr_name:
                self._register(getattr(key, self.attr_name), key)
                return key
            else:
                raise ValueError('Registry key is required.')

        def _decorator(cls):
            self._register(key, cls)
            return cls

        return _decorator

class ConfigurableRegistry(ClassRegistry):
    """Extension of Mutable registry that includes default settings and subfields.
    
    Support custom keys within the registry (2 levels). This wrapper aims to solve the insecurity
    """
    def __init__(self, 
        class_key:str, 
        attr_args: typing.MutableMapping = '_default_args',
        attr_kwargs: typing.MutableMapping = '_default_kwargs',
        additional_args = [],
        initial_registry: typing.Optional[typing.MutableMapping] = None,
        register_hook :typing.Callable = None,
        **kwargs):
        """Constructor for ConfigurableRegistry for
        
        Args:
            -   """
        super(ConfigurableRegistry, self).__init__(**kwargs)
        try:
             import segmentation_models_pytorch as smp
        except:
            raise ImportError('Error importing Segmentation models package. Package must be installed.')

        assert isinstance(additional_args, typing.List), f'Additional args must be a list. Got {additional_args.__class__.__name__}'
        
        self._class_key = class_key
        self._attr_args = attr_args
        self._attr_kwargs = attr_kwargs
        self._additional_args = additional_args
        self._registry = copy.deepcopy(initial_registry) if initial_registry else dict()
        self._register_hook = register_hook

    def _register(self, key: typing.Hashable, item:typing.Union[type, typing.MutableMapping]):
        """Register item.
        
        It could be a object of type :py:class:`type` or :py:class:`Mutablemapping`"""
 
        if isinstance(item, type):
            item = {
                key : { 
                    self._class_key : item,
                    self._attr_args : getattr(item, self._attr_args, list()),
                    self._attr_kwargs : getattr(item, self._attr_kwargs, dict()),
                    **self._get_attrs(item, *self._additional_args)
                }
            }

        self._registry.update(item)
        
        if self._register_hook:
            self._register_hook(item)

    def get_class(self, key: typing.Hashable):
        """Get class for the specified key.
        
        Raise:
            KeyError"""
        return self._registry[key].get(self._class_key)

    def get_instance(self, key: typing.Hashable, *args, **kwargs):
        """Get instance of class with default params
        
        Raise:
            KeyError"""
        
        cls_args = self.get_field(key, self._attr_args, list())
        cls_kwargs = self.get_field(key, self._attr_kwargs, dict())
        cls_kwargs.update(**kwargs)
        return super().get_instance(key, *cls_args, **cls_kwargs)

    def get_field(self, key: typing.Hashable, subkey: typing.Hashable, default: typing.Optional[typing.Any] = None):
        """Get subfield for the specified key
        
        Raise:
            KeyError"""
        return self._registry[key].get(subkey, default)

    def _get_attrs(self, cls:type, *args):
        """Returns a dictionary for the args and class specified

        If `py:attr:` does not exist in , an empty dictionary is return as value for the attribute
        
        Raise:
            KeyError"""

        return {subkey: getattr(cls, subkey, None) for subkey in args}

    def set_field(self, key: typing.Hashable, subkey: typing.Hashable, value: typing.Any, update:bool = True):
        """Set field of registry"""
        self._registry[key].update( subkey = value )

    def add_additional_args(self, attr_name:str):
        """Add attribute name to be stored in the registry"""
        self._additional_args.append(attr_name)

# REGISTRIES INITIALIZATION
try:
    import segmentation_models_pytorch as smp
    _initial_backbone_registry = smp.encoders.encoders
    _initial_model_registry = dict()

    unet_registry = dict(
        model = smp.unet.Unet,
        params = {
            'encoder_name':  "resnet34",
            'encoder_depth':  5,
            'encoder_weights':"imagenet",
            'decoder_use_batchnorm':True,
            'decoder_channels': (256, 128, 64, 32, 16),
            'decoder_attention_type': None,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )
    _initial_model_registry['unet'] = unet_registry
except Exception as e:
    # loggin.warning(f'segmentation_pytorch not installed.')
    print(e)
    _initial_backbone_registry = dict()
    _initial_model_registry = dict()

backbones_reg = ConfigurableRegistry(
    class_key = 'encoder',                      # Key to the nn.module class
    initial_registry = _initial_backbone_registry,       # Initial registry. Default: None
    attr_name = '_register_name',
    attr_args = 'params',
    additional_args= ['pretrained_settings'],
    register_hook= lambda item : smp.encoders.encoders.update(item)) # Retrocompatibility

models_reg = ConfigurableRegistry(
    class_key = 'model',                      # Key to the nn.module class
    initial_registry = _initial_model_registry,       # Initial registry. Default: None
    attr_name = '_register_name',
    attr_kwargs = 'params',
    additional_args= ['pretrained_settings'] )







