"""readers.py

Defines utilities for creating the various 'reader' objects used throughout
Pyrokinetics.
"""

from pathlib import Path
from typing import Union, Type, Any
from .typing import PathLike
from abc import ABC, abstractmethod
from .factory import Factory


class Reader(ABC):
    @abstractmethod
    def read(self, filename: PathLike, *args, **kwargs) -> Any:
        """Read and process a file"""
        pass

    def verify(self, filename: PathLike) -> None:
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

    def __call__(self, filename: PathLike, *args, **kwargs) -> Any:
        return self.read(filename, *args, **kwargs)


def create_reader_factory(BaseReader=Reader, name: str = None):
    """Generates Factory which inherits UserDict and redefines the __getitem__
    and __setitem__ methods. The registered types should subclass BaseReader, which
    defaults to Reader.

    Factories behave similarly to dictionaries, and define a mapping between
    between strings and types. On performing a lookup, the factory will create a
    new object.

    Args:
        BaseReader (type): Parent class of readers created by this factory.
        name (str): Name of the factory created. If None, it is set to
            the name "{BaseReader.__name__}Factory".

    Returns:
        reader_factory (instance of ReaderFactory)
    """

    class ReaderFactory(Factory):
        """
        Given a key as a string, returns a Reader object derived from BaseReader.
        These objects should define the methods 'read' and 'verify' at a minimum.

        Optionally, the user may instead simply supply a file name, and the file type
        will be automatically inferred.
        """

        def __init__(self):
            super().__init__(BaseReader)

        def get_type(self, key: str) -> Union[Type[BaseReader], None]:
            # First, assume the given key is name registered to the factory
            # Note that the values of self.data are class types. A new instance is
            # created for each call to __getitem__
            try:
                return super().get_type(key)
            except KeyError as key_error:
                # Given it's a file name, try inferring the kinetics type
                try:
                    inferred_key = self._infer_file_type(key)
                    return super().get_type(inferred_key)
                except (FileNotFoundError, RuntimeError):
                    raise KeyError(
                        f"There is no {BaseReader.__name__} defined for type {key}, "
                        f"nor is {key} the name of a valid input file."
                    ) from key_error

        def _infer_file_type(self, filename: PathLike) -> Union[str, None]:
            # Check to see if it's a valid filename
            filename = Path(filename)
            if not filename.exists():
                raise FileNotFoundError(f"{filename} does not exist")
            for file_type, Reader in self.data.items():
                try:
                    Reader().verify(filename)
                    return file_type
                except Exception:
                    continue
            raise RuntimeError("Unable to infer file type")

        def __setitem__(self, key: str, value: Type[BaseReader]):
            super().__setitem__(key, value)
            self.get_type(key).file_type = key  # tag the type with its own key

    # Set BaseReader as a class-level attribute
    ReaderFactory.BaseReader = BaseReader

    # Rename class
    if name is None:
        name = f"{BaseReader.__name__}Factory"
    ReaderFactory.__name__ = name
    ReaderFactory.__qualname__ = name

    return ReaderFactory()
