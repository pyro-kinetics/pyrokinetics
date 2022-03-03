"""
Defines generic Factory object.
Factory objects used throughout this project may inherit this class and add additional
features, such as filetype inference.
"""

from typing import Type
from collections import UserDict


class Factory(UserDict):
    """
    Given a key as a string, returns an object object derived from BaseClass, which
    is provided to __init__. By default, BaseClass is 'object', meaning the factory
    can produce any type.

    Factory behaves like a dict. Types may be registered by calling:
    `my_factory[type_key] = Type`
    where `type_key` is a string. If types take no arguments to their __init__
    functions, they may be created using:
    `my_type = my_factory[type_key]`
    Alternatively, they may be be created by calling the factory as a function:
    `my_type = my_factory(type_key, *args, **kwargs)`
    """

    def __init__(self, BaseClass: Type = object):
        super().__init__()
        self.BaseClass = BaseClass

    def get_type(self, key: str):
        """
        Returns type, but does not instantiate. Derived classes may override this to
        change behaviour for both __getitem__ and __call__ functions
        """
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(
                f"{self.__class__.__name__} has not registered the key {key}."
            )

    def __getitem__(self, key: str):
        """Gets type, then instantiates with no args passed"""
        return self.get_type(key)()

    def __call__(self, key: str, *args, **kwargs):
        """Gets type, then instantiates with args/kwargs"""
        return self.get_type(key)(*args, **kwargs)

    def __setitem__(self, key: str, value: Type):
        try:
            if issubclass(value, self.BaseClass):
                self.data[key] = value
            else:
                raise ValueError(
                    f"Classes registered to {self.__class__.__name__} must "
                    f"subclass {self.BaseClass.__name__}"
                )
        except TypeError as e:
            raise TypeError(
                f"Only classes may be registered to {self.__class__.__name__}"
            ) from e
        except ValueError as e:
            raise TypeError(str(e))
