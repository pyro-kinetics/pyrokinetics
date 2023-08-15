"""
Defines generic factory object.

Factory objects used throughout this project may inherit this class and add additional
features, such as filetype inference.
"""

from typing import Any, Generator, Tuple, Type


class Factory:
    """
    A variation on the generic 'factory' pattern, as defined in the classic 'Design
    Patterns' by the gang of four.

    Creates a mapping of keys to types. New types can be 'registered' to the factory
    via a key, and instances of those types can be created by the `create` method by
    passing that key. Multiple keys can be registered to the same type. Optionally, the
    factory can only return types derived from a given super class.

    Parameters
    ----------
    super_class
        Registered classes must be derived from this type.
    """

    def __init__(self, super_class: Type = object):
        self._super_class = super_class
        self._registered_types = {}

    def type(self, key: str) -> Type:
        """
        Returns type associated with a given key. Raises ``KeyError`` if there is no
        type registered with that key.
        """
        try:
            return self._registered_types[key]
        except KeyError:
            raise KeyError(
                f"'{key}' is not registered with {self._super_class.__name__} factory"
            )

    def create(self, key: str, *args, **kwargs) -> Any:
        """Create a new object of type ``key``, forwarding all arguments."""
        return self.type(key)(*args, **kwargs)

    def register(self, key: str, cls: Type) -> None:
        """
        Register a new type with the factory. ``cls`` must be derived from
        the ``super_class`` that was passed to the ``__init__`` method.
        """
        try:
            if issubclass(cls, self._super_class):
                self._registered_types[key] = cls
            else:
                raise ValueError(
                    f"Classes registered must subclass {self._super_class.__name__}"
                )
        except TypeError as e:
            raise TypeError("Only classes may be registered") from e
        except ValueError as e:
            raise TypeError(str(e))

    def __call__(self, key: str, *args, **kwargs) -> Any:
        """Alternative to `create`"""
        return self.create(key, *args, **kwargs)

    def __getitem__(self, key: str) -> Type:
        """Alternative to `type`"""
        return self.type(key)

    def __setitem__(self, key: str, cls: Type) -> None:
        """Alternative to `register`"""
        self.register(key, cls)

    def __contains__(self, key: str) -> None:
        """Check if key is registered by the factory"""
        return key in self._registered_types

    def __iter__(self) -> Generator[str, None, None]:
        """Iterate over registered keys"""
        return iter(self._registered_types)

    def items(self) -> Generator[Tuple[str, Type], None, None]:
        """Dict-like items iterator"""
        return self._registered_types.items()
