"""
This modules contains utility functions for adding plugins to Pyrokinetics.
This is achieved using entry-points.

If you want to register your own :class:`~pyrokinetics.equilibrium.Equilibrium`
reader, it should inherit :class:`pyrokinetics.file_utils.FileReader`,
and its ``read_from_file()`` function should return an ``Equilibrium``. To add
this plugin to Pyrokinetics, you should add the following to your
``pyproject.toml`` file::

    [project.entry-points."pyrokinetics.equilibrium"]
    my_eq = "my_project.my_module:MyEqReader"

This will register the class ``MyEqReader``, and within Pyrokinetics the
equilibrium type will be ``"my_eq"``. Note that here, ``"pyrokinetics.equilibrium"`` is
an entry point group name, not a module. The group names for each Pyrokinetics
file reader are:

- ``"pyrokinetics.gkinput"``
- ``"pyrokinetics.gkoutput"``
- ``"pyrokinetics.equilibrium"``
- ``"pyrokinetics.kinetics"``

For more information, please see:

- `PyPA entry points specifications
  <https://packaging.python.org/en/latest/specifications/entry-points/>`_
- `Setuptools entry points tutorial
  <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_
"""

from importlib.metadata import entry_points
from textwrap import dedent
from typing import Type

from .file_utils import FileReader, ReadableFromFile

__all__ = ["register_file_reader_plugins"]


def register_file_reader_plugins(
    group_name: str, Readable: Type[ReadableFromFile]
) -> None:
    """
    Defines an entry point group for classes that implement the
    :class:`~pyrokinetics.file_utils.FileReader` and registers any
    user-defined plugins with that group.

    Parameters
    ----------
    group_name
        The name of the entry points group. This will be prepended with
        ``"pyrokinetics."`` if it is not already.
    Readable
        The type of class returned by the registered file readers.
    """
    # Ensure group name has the correct prefix
    prefix = "pyrokinetics."
    if group_name[: len(prefix)] != prefix:
        group_name = prefix + group_name

    # Get group, returning early if there are no plugins
    try:
        group = entry_points()[group_name]
    except KeyError:
        return

    # Register all plugins in the user's environment
    for entry in group:
        # Simply loading the entry point should register the plugin if it
        # inherits FileReader
        cls = entry.load()
        # Check that it is of the correct type
        if not issubclass(cls, FileReader):
            raise TypeError(
                f"Plugin class {cls.__qualname__} should subclass {cls.__qualname__}"
            )
        # Check that the registered file type matches the entry point name
        if entry.name not in Readable.supported_file_types():
            err_msg = dedent(
                f"""\
                Entry point name {entry.name} does not match any registered file types
                for the class {Readable.__qualname__}. Registered types include
                '{"', '".join(Readable.supported_file_types())}'.
                """
            )
            raise RuntimeError(err_msg)
