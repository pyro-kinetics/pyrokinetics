r"""
Pyrokinetics handles many different file types generated from many different software
packages. This module contains utilities for simplifying the process of reading and
processing files. These utilities also make it possible to extend Pyrokinetics for
new file types without modifying the existing code.

For more information, see :ref:`sec-file-readers`.
"""

from __future__ import annotations

__all__ = [
    "AbstractFileReader",
    "FileReader",
    "ReadableFromFile",
]

from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Type

from .typing import PathLike
from .factory import Factory


class AbstractFileReader(ABC):
    """
    An abstract base class for classes that can read data from disk and create a
    Pyrokinetics object. Subclasses should define both a ``read_from_file`` method and
    a ``verify_file_type`` method.

    Subclasses should usually make use of :class:`FileReader`, as this additionally
    associates the class with its associated 'readable'. These classes are kept separate
    to handle the special case of ``GKInput``, as it is both the 'readable' class and
    the reader.
    """

    @abstractmethod
    def read_from_file(self, filename: PathLike, *args, **kwargs) -> Any:
        """
        Read and process the data from a file.

        Parameters
        ----------
        filename: PathLike
            The file to be read.
        *args
            Additional positional arguments used by the derived file reader.
        **kwargs
            Keyword arguments used by the derived file reader.

        Returns
        -------
        Any
            Derived classes may return any type of data from this function.

        Notes
        -----
        Rather than accepting ``*args`` and/or ``**kwargs``, it is recommended that
        derived classes should specify their keywords explicitly.
        """
        pass

    def verify_file_type(self, filename: PathLike) -> None:
        """
        Perform a series of checks on the file to ensure it is valid. Raises an
        exception if the file is of the wrong type. Exits normally if the file is valid.

        The default implementation simply reads the file, performs the usual processing,
        and discards the results. This is rarely the best way to verify a file type,
        so this should be overridden is most cases. In particular, the default
        implementation should not be used if:

        - Reading and processing the whole file is computationally expensive.
        - The read function depends upon keyword arguments.
        - The read function can read multiple related file types and further information
          is needed to differentiate between them. For example, multiple gyrokinetics
          codes use Fortran namelists as input files, so a specialised verify method
          is needed to check the names stored within to determine which code the input
          file belongs to.
        - An exception raised when reading from file should halt the program.

        Parameters
        ----------
        filename: PathLike
            The file to be read.
        """
        self.read_from_file(filename)

    def __call__(self, filename: PathLike, *args, **kwargs) -> Any:
        """Forwards calls to ``read_from_file``."""
        return self.read_from_file(filename, *args, **kwargs)


class FileReader(AbstractFileReader):
    """
    Builds upon :class:`AbstractFileReader`, but adds the required class keyword
    arguments ``file_type`` and ``reads``. These are used to register the file reader
    with their associated 'readable' class.
    """

    file_type: ClassVar[str]

    def __init_subclass__(cls, file_type: str, reads: Type[ReadableFromFile], **kwargs):
        """
        Sets the ``file_type`` class attribute on subclasses and registers them with
        the appropriate 'readable' factory.
        """
        super().__init_subclass__(**kwargs)
        cls.file_type = file_type
        reads._register(file_type, cls)


class FileReaderFactory(Factory):
    """
    Factory variant in which file type can be inferred from a path as well as a key.
    """

    def type(self, key: str) -> Type:
        """
        Returns type associated with a given key. If there is no type registered with
        that key, tries to infer the type by reading the file. This makes use of the
        :func:`~FileReader.verify_file_type` functions implemented for each
        registered reader.
        """
        try:
            return super().type(key)
        except KeyError:
            return super().type(self._infer_file_type(key))

    def _infer_file_type(self, filename: PathLike) -> str:
        """
        Check to see if ``filename`` is a valid file type for any registered file
        readers. Uses the ``verify_file_type`` function of each file reader.
        """
        err_msg = f"'{filename}' is neither a registered key nor a valid file."
        filename = Path(filename)
        if not filename.exists():
            raise KeyError(err_msg)
        for key, FileReader in self.items():
            with suppress(Exception):
                FileReader().verify_file_type(filename)
                return key
        raise KeyError(err_msg)


class ReadableFromFile:
    """
    Base class that adds the following functions to a class decorated with
    :func:`readable_from_file`:

    - ``from_file``: A classmethod that allows a instance of the readable class to be
      created from a file path.

    - ``supported_file_types``: Returns a list of all registered file types that can
      be used to instantiate the readable class.

    It also adds the following private objects:

    - ``_factory``: A :class:`.FileReaderFactory` that returns 'readable' subclasses.

    - ``_register``: A function used to register 'reader' classes with the readable,
      allowing those classes to be used when reading files from disc.
    """

    _factory: ClassVar[FileReaderFactory]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._factory = FileReaderFactory(super_class=FileReader)

    @classmethod
    def from_file(cls, path: PathLike, file_type: Optional[str] = None, **kwargs):
        """
        Read a file from disk, returning an instance of this class.

        Parameters
        ----------
        path: PathLike
            Location of the file on disk.
        file_type: Optional[str]
            String specifying the type of file. If unset, the file type will be
            inferred automatically. Specifying the file type may improve performance.
        **kwargs:
            Keyword arguments forwarded to the file reader.

        Raises
        ------
        ValueError
            If ``path`` does not refer to a valid file.
        RuntimeError
            If ``file_type`` is unset, and it is not possible to infer the file type
            automatically.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File {path} not found.")
        # Infer reader type from path if not provided with file_type
        reader = cls._factory(str(path) if file_type is None else file_type)
        return reader(path, **kwargs)

    @classmethod
    def supported_file_types(cls) -> List[str]:
        """
        Returns a list of all registered file types. These file types are readable by
        :func:`from_file`.
        """
        return [*cls._factory]

    @classmethod
    def _register(cls, file_type: str, Reader: Type[FileReader]) -> None:
        """
        Registers classes so that they're usable with :func:`from_file`.

        Parameters
        ----------
        file_type
            The registered name for the file reader class. This name is appended to the
            list returned by :func:`supported_file_types`. When building from a file
            using :func:`from_file`, the optional ``file_type`` argument will correspond
            to this name.
        Reader
            The class to register.
        """
        if not issubclass(Reader, FileReader):
            raise TypeError("Can only register subclasses of FileReader")
        if file_type in cls._factory:
            raise RuntimeError(
                f"File type {file_type} is already registered with {cls.__qualname__}"
            )
        cls._factory[file_type] = Reader