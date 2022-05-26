"""
Classes for the implementation of a registry of components.

Code adapted from https://github.com/todofixthis/class-registry/blob/master/class_registry/registry.py
"""
# Error for key not found

# Base class for registry

from abc import ABC, abstractmethod
import collections
import inspect
from pathlib import Path
import typing

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
        return self.get_class(key)

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

    def items(self) -> typing.Iterable[typing.Tuple[typing.Hashable, type]]:
        """Get registered classes and their asociated keys.
        
        Returns
            A generator for iterating over registered pairs (key, class).
        """
        raise NotImplementedError

    def values(self) -> typing.Iterable[type]:
        """Get registered classes.

        Returns
            A generator for iterating over registered classes.
        """
        for item in self.items():
            yield item[1]

    def get(self, key: typing.Hashable, *args, **kwargs):
        """Creates a new instance of the class matching the key.
        
        Args:
            key (typing.Hashable): Lookup key.
            args: Positional arguments passed to the class constructor.
            kwargs: Keyword arguments passed to the class consturctor.

        Returns:
            Instance of the class matching the key
        """
        return self.get_class(key)(*args, **kwargs)

    @abstractmethod
    def get_class(self, key: typing.Hashable):
        """Returns the class associated with the specified key

        Args:
            key (typing.Hashable): Lookup key

        Returns:
            Class associated with the specified key
        """
        raise NotImplementedError


class MutableRegistry(Registry, collections.abc.MutableMapping, ABC):
    """Extends :py:class:`Registry` with methods to modify the registered classes."""
    @abstractmethod
    def __init__(self, attr_name: typing.Optional[str] = None):
        """MutableMapping constructor
        Args:
            attr_name (typing.Optional[str]): Key to use from registered class attribute when registering a new class. Defaults to None.
        """
        super().__init__()
        self.attr_name = attr_name

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

    def register(self, key: typing.Union[type, typing.Hashable]):
        """Decorator that registers a class with the registry.
        
        Args:
            key: The registry key to use for the registered class.
            Optional if the registry's :py:attr:`attr_name` is set.

        Raise:
            :py:class:`ValueError` if ket is not provided.

        """
        # Key set from class.attr_name
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
    """Base Class registry with register and unregister functions implemented
    
    Maintains a registry of classes and provides a generic factory for
    instantiating them. Useful for modular components. 
    """

    def __init__(
            self,
            attr_name: typing.Optional[str] = None,
            unique: bool = False,
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
        super(ClassRegistry, self).__init__(attr_name)

        self.unique = unique

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
        

    def get_class(self, key):
        """
        Get the class associated with the specified key.

        Returns:
            Class associated with the specified key.

        Raise:
            :py:class:`KeyError` if the key is not registered.
        """
        return self._registry[key]


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
        if key in ['', None]:
            raise ValueError(f'Attempting to register class {cls} with empty registry key')

        if self.unique and (key in self._registry):
            raise KeyError(f'{cls} with key {key} is already registered.')

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

# Easy configurables components  
models = ClassRegistry(attr_name='_register_name')
backbones = ClassRegistry(attr_name='_register_name')
modules = ClassRegistry(attr_name='_register_name')
# Datasets
# Dataloaders

# Include timm backbones

class ThirdPartyClassRegistry(ClassRegistry):
    pass

class DefaultSettingClassRegistry(ClassRegistry):
    """Registry of classes with default settings"""

    def __init__(
        self,
        attr_name: typing.Optional[str] = None,
        unique: bool = False,
        config_path: typing.Union[str, Path] = 'config'
        ):
        super().__init__(attr_name=attr_name, unique=unique)

import pkg_resources

def print_compati():
    stream = pkg_resources.resource_stream(__name__, 'model/backbone/com_timm_backbones.txt')
    return [line.strip() for line in stream]


