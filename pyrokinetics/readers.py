from pathlib import Path
from typing import Type, Any
from .typing import PathLike
from abc import ABC, abstractmethod
from .factory import Factory


class Reader(ABC):
    """
    An abstract base class for classes that can read data from disk and create a
    Pyrokinetics object. ``Reader`` classes should define both a ``read`` method and a
    ``verify`` method.

    - ``read`` simply reads a file and returns a Pyrokinetics object.
    - ``verify`` checks whether a file is of a particular type. It should raise an
      exception if it isn't, or do nothing otherwise. This is used for automatic
      filetype inference.
    """

    @abstractmethod
    def read(self, filename: PathLike, **kwargs) -> Any:
        """
        Read and process an arbitrary number of files.

        Parameters
        ----------
        filename: PathLike
            The file to be read.
        **kwargs
            Keyword arguments to be used by the reader. This may be used to pass in
            options to the reader, and may also be used to pass in auxiliary file names.

        Returns
        -------
        Any
            Derived classes may return any type of data from this function.

        Notes
        -----
        Rather than accepting ``**kwargs``, it is recommended that derived classes
        should specify their keywords explicitly.
        """
        pass

    def verify(self, filename: PathLike) -> None:
        """
        Perform a series of checks on the file to ensure it is valid. Does not return
        anything, but should raise exceptions if something goes wrong.

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

        Parameters
        ----------
        filename: PathLike
            The file to be read.
        """
        self.read(filename)

    def __call__(self, filename: PathLike, **kwargs) -> Any:
        """
        Forwards calls to ``read``.
        """
        return self.read(filename, **kwargs)


def create_reader_factory(BaseReader=Reader, name: str = None):
    """
    Generates Factory which inherits UserDict and redefines the __getitem__ and
    __setitem__ methods. The registered types should subclass BaseReader, which
    defaults to Reader.

    Factories behave similarly to dictionaries, and define a mapping between
    between strings and types. On performing a lookup, the factory will create a
    new object.

    Parameters
    ----------
    BaseReader:
        Parent class of readers created by this factory.
    name:
        Name of the factory created. If None, it is set to the name
        ``"<BaseReader name>Factory"``.

    Returns
    -------
    reader_factory: ReaderFactory
    """

    class ReaderFactory(Factory):
        """
        Given a key as a string, returns a Reader object derived from BaseReader.
        These objects should define the methods ``read`` and ``verify`` at a minimum.

        Optionally, the user may instead simply supply a file name, and the file type
        will be automatically inferred.
        """

        def __init__(self):
            super().__init__(BaseReader)

        def get_type(self, key: str) -> Type[BaseReader]:
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

        def _infer_file_type(self, filename: PathLike) -> str:
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
