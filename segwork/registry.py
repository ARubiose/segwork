"""
Classes for the implementation of a registry of components and a basic factory method.
"""
import abc
import logging
import collections
import inspect
import typing
import copy

_logger = logging.getLogger(__name__)

__all__ = ['Registry', 'MutableRegistry', 'ClassRegistry', 'ConfigurableRegistry', 'backbones_reg', 'models_reg']

ItemType = typing.TypeVar('ItemType')

class Registry(collections.abc.Mapping, abc.ABC):
    """Base abstract class for a registry.
    """

    def __iter__(self):
        """Returns a generator for iterating over registry items."""
        return self.keys()

    def __getitem__(self, key: typing.Hashable):
        """Get class associated with key.

        Args:
            key (typing.Hashable): Lookup key.
        """
        return self.get_item(key)

    @abc.abstractmethod
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
        except KeyError as e:
            self.__missing__(e)
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def get_item(self, key: typing.Hashable) -> ItemType:
        """Returns registry item with specified key"""
        raise NotImplementedError

    def __missing__(self, e: KeyError):
        """Routine for missing keys"""
        raise KeyError(e)


class MutableRegistry(
    Registry, 
    collections.abc.MutableMapping,
    abc.ABC):
    """Extends :py:class:`Registry` with methods to add, modify and remove registered items.
    
    Supports registering classes through decorators.
    
    Args: 
        attr_name: If provided, :py:meth:`register` will automatically detect
        the key to use when registering new classes.
        unique: Determines what happens when two classes are registered with
        the same key:
        - ``True``: The second class will replace the first one.
        - ``False``: A ``ValueError`` will be raised"""

    def __init__(
            self,
            attr_name: typing.Optional[str] = '_register_name',   
            unique: bool = False
    ) -> None:
        self.attr_name = attr_name
        self.unique = unique

    def __delitem__(self, key: typing.Hashable):
        """Wrapper for :py:meth:`_unregister`. Support to dict like syntax.

        Args:
            key (typing.Hashable): Lookup key.
        """
        self._unregister(key)

    def __setitem__(self, key: typing.Hashable, value: ItemType):
        """Wrapper for :py:meth:`_register`. Support to dict like syntax.

        Args:
            key (typing.Hashable): Lookup key.
        """
        self._validate_register(key, value)
        self._register(key, value)

    def _register_class(self, key: typing.Hashable, cls: type):
        """Register item from class

        Args:
            key (typing.Hashable): Lookup key.
        """
        value = self._generate_value_from_cls(cls)
        self._validate_register(key, value)
        self._register(key, value)

    @abc.abstractmethod
    def _generate_value_from_cls(cls: type) -> ItemType:
        raise NotImplementedError

    def _validate_register(self, key:typing.Hashable, value:ItemType):
        """Validate register"""
        self._validate_key(key)
        self._validate_value(value)

    def _validate_key(self, key):
        """Validate key before register"""
        if self.unique and not self.is_unique(key):
            raise KeyError(f'Entry with key {key} already exists. Set the unique attribute to false to overried items.')

        if key in ['', None]:
            raise ValueError(f'Attempting to register class with empty registry key')

    def _validate_value(self, value:ItemType):
        """Validate value before register"""
        pass

    @abc.abstractmethod
    def is_unique(self, key:typing.Hashable):
        raise NotImplementedError(f'Not implemented in {type(self).__name__}.')

    @abc.abstractmethod
    def _register(self, key: typing.Hashable, value: ItemType) -> None:
        """Registers a class with the registry."""
        raise NotImplementedError(f'Not implemented in {type(self).__name__}.')

    @abc.abstractmethod
    def _unregister(self, key: typing.Hashable) -> type:
        """Unregisters the class at the specified key."""

        raise NotImplementedError(f'Not implemented in {type(self).__name__}.')

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
                self._register_class(getattr(key, self.attr_name), key)
                return key
            else:
                raise ValueError('Registry key is required.')

        def _decorator(cls):
            self._register_class(key, cls)
            return cls

        return _decorator

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
    """Description
    
    Maintains a registry of classes and provides a generic factory for
    instantiating them. Useful for modular components. 
    """

    def __init__(self, 
        register_hook:typing.Optional[typing.Callable] = None, 
        get_hook: typing.Optional[typing.Callable] = None,
        initial_registry: typing.MutableMapping = None, 
        *args, **kwargs) -> None:
        """Constructor method for :py:class:`ClassRegistry`
        """
        super().__init__(*args, **kwargs)
        self._register_hook = register_hook
        self._get_hook = get_hook
        self._registry = self._get_initial_registry(initial_registry) if initial_registry else dict()

    def __len__(self):
        """
        Get the number of registered classes.

        Returns:
            Number of registered classes
        """
        return len(self._registry)

    def is_unique(self, key: typing.Hashable) -> bool:
        """Returns whether a key exists or not in the registry"""
        return key not in self._registry  

    def __repr__(self):
        return f'{type(self).__name__}\n\tattr_name: {self.attr_name}\n\tunique: {self.unique}\n'\
            f'\tNumber of registered classes: {self.__len__()} \n'\
            f'\tRegistered classes: {list(self.keys())}'
        
    def get_item(self, key: typing.Hashable, *args, **kwargs):
        """Implementation of get item."""
        value = self._registry.get(key)
        if self._get_hook:
            value = self._get_hook(value)
        return value
        
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
        """Create instance of class associated with the specified key"""
        cls = self.get_class(key)
        return cls(*args, **kwargs)

    def _generate_value_from_cls(self, cls:type) -> ItemType:
        return cls

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

        if self._register_hook:
            self._register_hook(key, cls)

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

    def _validate_value(self, value:type):
        """Validate value before register class"""
        assert isinstance(value, type), f'Vaue must be of type {type}. Got {value.__class__.__name__}'
        
    def _get_initial_registry(self, registry: typing.MutableMapping) -> typing.Dict:
        """Initialize"""
        self._validate_initial_registry(registry)
        return copy.deepcopy(registry)

    def _validate_initial_registry(self, registry):
        if not all(inspect.isclass(value) for value in registry.values()):
            raise ValueError(f'Values in the registry must be type {type}')

class ConfigurableRegistry(ClassRegistry):
    """Subclass of :py:class:`ClassRegistry` that includes default settings and additional information of the registered class.
    
    """
    def __init__(self, 
        class_key:str, 
        attr_args: str = '_default_args',
        attr_kwargs: str = '_default_kwargs',
        additional_args = [],
        **kwargs):
        """Constructor for ConfigurableRegistry for
        
        Args:
            -   """
        assert isinstance(additional_args, typing.List), f'Additional args must be a list. Got {additional_args.__class__.__name__}'
        self._class_key = class_key
        self._attr_args = attr_args
        self._attr_kwargs = attr_kwargs
        self._additional_args = additional_args

        super(ConfigurableRegistry, self).__init__(**kwargs)

    def _generate_value_from_cls(self, cls):
        return {
                self._class_key : cls,
                self._attr_args : getattr(cls, self._attr_args, list()),
                self._attr_kwargs : getattr(cls, self._attr_kwargs, dict()),
                **{subkey: getattr(cls, subkey, None) for subkey in self._additional_args}
            }

    def _validate_value(self, value: typing.Dict):
        assert self._class_key in value, f'Value must have a key {self._class_key} containing a reference to the class.'
        # Warning if no args are store. 

    def get_class(self, key: typing.Hashable):
        """Get class for the specified key.
        
        Raise:
            KeyError"""
        return self._registry[key].get(self._class_key)

    def get_instance(self, key: typing.Hashable, *args, **kwargs):
        """Get instance of class with default params
        
        Raise:
            KeyError"""

        cls = self.get_class(key)
        cls_args = set([*self.get_subfield(key, self._attr_args, list()), *args])
        cls_kwargs = self.get_subfield(key, self._attr_kwargs, dict())
        cls_kwargs.update(**kwargs)
        return cls(*cls_args, **cls_kwargs)

    def get_subfield(self, key: typing.Hashable, subkey: typing.Hashable, default: typing.Optional[typing.Any] = None):
        """Get subfield for the specified key
        
        Raise:
            KeyError"""
        return self._registry[key].get(subkey, default)

    def set_field(self, key: typing.Hashable, subkey: typing.Hashable, value: typing.Any, update:bool = True):
        """Set field of registry"""
        self._registry[key].update( subkey = value )

    def add_additional_args(self, attr_name:str):
        """Add attribute name to be stored in the registry"""
        self._additional_args.append(attr_name)

    def _validate_initial_registry(self, registry: typing.MutableMapping):
        if not all(self._class_key in value for key, value in registry.items()):
            raise ValueError(f'Values in the registry must have a key {self.class_key} containing the classes')

    def __repr__(self):
        msg = f'\tClass key: {self._class_key}\n'\
        f'\tAttribute args: {self._attr_args}\n'\
        f'\tAttribute kwargs: {self._attr_kwargs}\n'\
        f'\tAdditional info from attributes: {self._additional_args}\n'
        return f'{super().__repr__()}\n' + msg

### Proposed registrys

try:
    import segmentation_models_pytorch as smp

    def smp_hook(key, value):
        """Safe inclusion of key and value into smp encoders repository"""
        smp.encoders.encoders[key] = value

    unet_registry = dict(
        model = smp.Unet,
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

    unetplusplus_registry = dict(
        model = smp.UnetPlusPlus,
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

    manet_registry = dict(
        model = smp.MAnet,
        params = {
            'encoder_name':  "resnet34",
            'encoder_depth':  5,
            'encoder_weights':"imagenet",
            'decoder_use_batchnorm':True,
            'decoder_channels': (256, 128, 64, 32, 16),
            'decoder_pab_channels': 64,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )

    linknet_registry = dict(
        model = smp.Linknet,
        params = {
            'encoder_name':  "resnet34",
            'encoder_depth':  5,
            'encoder_weights':"imagenet",
            'decoder_use_batchnorm':True,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )

    fpn_registry = dict(
        model = smp.FPN,
        params = {
            'encoder_name':  "resnet34",
            'encoder_depth':  5,
            'encoder_weights':"imagenet",
            # 'decoder_pyramid_channels':True,
            # 'decoder_segmentation_channels':True,
            # 'decoder_merge_policy':True,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }

    )

    psp_registry = dict(
        model = smp.PSPNet,
        params = {
            'encoder_name':  "resnet34",
            'encoder_depth':  5,
            'encoder_weights':"imagenet",
            # 'psp_out_channels':True,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )

    pan_registry = dict(
        model = smp.PAN,
        params = {
            'encoder_name': "resnet34",
            'encoder_depth': 5,
            'encoder_weights':"imagenet",
            'encoder_output_stride' : 16,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )

    deeplabv3_registry = dict(
        model = smp.DeepLabV3,
        params = {
            'encoder_name': "resnet34",
            'encoder_depth': 5,
            'encoder_weights':"imagenet",
            'decoder_channels' : 256,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )
    deeplabv3plus_registry = dict(
        model = smp.DeepLabV3Plus,
        params = {
            'encoder_name': "resnet34",
            'encoder_depth': 5,
            'encoder_weights':"imagenet",
            'decder_channels': 256,
            # 'encoder_output_stride' : 256,
            # 'decoder_atrous_rates' : 256,
            'in_channels':  3,
            'classes': 1,
            'activation': None,
            'aux_params':  None,
        }
    )

    _initial_model_registry = dict()
    _initial_model_registry['unet'] = unet_registry
    _initial_model_registry['unet++'] = unetplusplus_registry
    _initial_model_registry['manet'] = manet_registry
    _initial_model_registry['linknet'] = linknet_registry
    _initial_model_registry['fpn'] = fpn_registry
    _initial_model_registry['psp'] = psp_registry
    _initial_model_registry['pan'] = pan_registry
    _initial_model_registry['deeplabv3'] = deeplabv3_registry
    _initial_model_registry['deeplabv3plus'] = deeplabv3_registry

    _initial_backbone_registry = smp.encoders.encoders
    _register_hook = smp_hook # Retrocompatibility
    _default_kwargs = 'params'
except Exception as e:
    # loggin.warning(f'segmentation_pytorch not installed.')
    print(e)
    _initial_backbone_registry = dict()
    _initial_model_registry = dict()
    _register_hook = None
    _default_kwargs = '_default_kwargs'

backbones_reg = ConfigurableRegistry(
    class_key = 'encoder',                      # Key to the nn.module class
    initial_registry = _initial_backbone_registry,       # Initial registry. Default: None
    attr_kwargs = _default_kwargs,
    additional_args= ['pretrained_settings'],
    register_hook= _register_hook) 

models_reg = ConfigurableRegistry(
    class_key = 'model',                      # Key to the nn.module class
    initial_registry = _initial_model_registry )









