"""KineticsReader

Defines abstract base class for KineticsReader objects, which are capable of
parsing various types of kinetics data.

Also defines a factory object that creates the required reader given the
kinetics type. This factory may instead infer the kinetics type from a file.
"""

from pathlib import Path
from typing import Union, Type, Dict
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from ..species import Species


class KineticsReader(ABC):
    @abstractmethod
    def read(self, filename: Union[str, Path], *args, **kwargs) -> Dict[str, Species]:
        """Read kinetics data, produce a dict of Species objects"""
        pass

    def verify(self, filename: Union[str, Path]) -> None:
        """Perform a series of checks on the file to ensure it is valid

        Does not return anything, but should raise exceptions if something goes
        wrong.

        By default, simply read the file. This should be avoided in cases where
        reading and processing the whole file takes a long time. Ideally, this
        function should make only a few quick metadata checks, and leave the
        actual processing to `read`. It is therefore recommended to shadow this
        function in subclasses.
        """
        self.read(filename)

    def __call__(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> Dict[str, Species]:
        return self.read(filename, *args, **kwargs)


class KineticsReaderFactory(MutableMapping):
    """
    Given a kinetics type as a string, returns a Reader object capable of parsing the
    relevant file type.

    Optionally, the user may instead simply supply a file name, and the file type
    will be automatically inferred.
    """

    def __init__(self):
        self.__dict = dict()

    def __getitem__(self, key: str) -> Union[KineticsReader, None]:
        # First, assume the given key is a kinetics type ("SCENE", "JETTO", etc)
        # Note that the values of self.__dict are class types. A new instance is
        # created for each call to __getitem__
        try:
            return self.__dict[key]()
        except KeyError as key_error:
            # If this fails, check to see if it's a valid filename
            filename = Path(key)
            if not filename.exists():
                raise KeyError(
                    f"{key} is not a valid kinetics type, nor is it the name of a "
                    f"kinetics input file."
                ) from key_error
            # Given it's a file name, try inferring the kinetics type
            try:
                return self.__dict[self._infer_kinetics_type(filename)]()
            except RuntimeError as infer_error:
                raise infer_error from key_error

    def _infer_kinetics_type(self, filename: Path) -> Union[str, None]:
        for kinetic_type, Reader in self.__dict.items():
            try:
                Reader().verify(filename)
                return kinetic_type
            except Exception:
                continue
        raise RuntimeError("Unable to infer kinetics file type")

    def __setitem__(self, key: str, value: Type[KineticsReader]):
        try:
            if issubclass(value, KineticsReader):
                value.kinetics_type = key  # tag the type with the key name
                self.__dict[key] = value
            else:
                raise ValueError(
                    "Classes registered to KineticsReaderFactory must subclass "
                    "KineticsReader"
                )
        except TypeError as e:
            raise TypeError(
                "Only classes may be registered to KineticsReaderFactory"
            ) from e
        except ValueError as e:
            raise TypeError(str(e))

    def __delitem__(self, key):
        self.__dict.pop(key)

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)


# Create global instance of reader factory
kinetics_readers = KineticsReaderFactory()
