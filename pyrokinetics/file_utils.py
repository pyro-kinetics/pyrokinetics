r"""
Pyrokinetics handles many different file types generated from many different software
packages. This module contains utilities for simplifying the process of reading and
processing files. These utilities also make it possible to extend Pyrokinetics for
new file types without modifying the existing code.

For more information, see :ref:`sec-file-readers`.
"""

__all__ = [
    "AbstractFileReader",
    "ReadableFromFileMixin",
    "readable_from_file",
]

from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, List, Optional, Type

from .typing import PathLike
from .factory import Factory


class AbstractFileReader(ABC):
    """
    An abstract base class for classes that can read data from disk and create a
    Pyrokinetics object. Subclasses should define both a ``read_from_file`` method and
    a ``verify_file_type`` method.

    Subclasses should also be decorated with :meth:`ReadableFromFileMixin.reader`, as
    this enables the file reader to be used by :meth:`ReadableFromFileMixin.from_file`.
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


class _FileReaderFactory(Factory):
    """
    Factory variant in which file type can be inferred from a path as well as a key.
    """

    def type(self, key: str) -> Type:
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


class ReadableFromFileMixin:
    """
    Mixin class that adds the following functions to a class decorated with
    :func:`readable_from_file`:

    - ``from_file``: A classmethod that allows a instance of the readable class to be
      created from a file path.

    - ``supported_file_types``: Returns a list of all registered file types that can
      be used to instantiate the readable class.

    - ``reader``: A decorator used to register 'reader' classes with the readable,
      allowing those classes to be used when reading files from disc.
    """

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
        reader = cls._factory(path if file_type is None else file_type)
        return reader(path, **kwargs)

    @classmethod
    def supported_file_types(cls) -> List[str]:
        """
        Returns a list of all registered file types. These file types are readable by
        :func:`from_file`.
        """
        return [*cls._factory]

    @classmethod
    def reader(cls, key: str) -> Callable:
        """
        Decorator for classes that inherit `AbstractFileReader` and create instances of
        this class. Registers classes so that they're usable with :func:`from_file`.

        Parameters
        ----------
        key: str
            The registered name for the file reader class. This name is appended to the
            list returned by :func:`supported_file_types`. When building from a file
            using :func:`from_file`, the optional ``file_type`` argument will correspond
            to this name.

        Returns
        -------
        Callable
            The decorator function that registers the class with the factory.
        """

        def decorator(t: Type[AbstractFileReader]) -> Type[AbstractFileReader]:
            if not issubclass(t, AbstractFileReader):
                raise TypeError("Can only register subclasses of AbstractFileReader")
            cls._factory[key] = t
            t.file_type = key
            return t

        return decorator


def readable_from_file(cls) -> Any:
    """
    Decorator that marks a class as being generated from various possible file
    types. For example, :class:`Equilibrium` is readable from G-EQDSK and TRANSP files.

    If multiple related classes are readable from file, only the base class should be
    marked.

    Should be used alongside the mixin class `ReadableFromFileMixin`.
    """
    if not issubclass(cls, ReadableFromFileMixin):
        raise TypeError(
            "readable_from_file decorates classes that inherit ReadableFromFileMixin"
        )
    cls._factory = _FileReaderFactory(super_class=AbstractFileReader)
    return cls
