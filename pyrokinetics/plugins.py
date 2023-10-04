"""
This modules contains utility functions for adding plugins to Pyrokinetics.
This is achieved using entry-points.

If you want to register your own :class:`~pyrokinetics.equilibrium.Equilibrium`
reader, it should inherit :class:`pyrokinetics.file_utils.AbstractFileReader`,
and its ``read_from_file()`` function should return an ``Equilibrium``. To add
this plugin to Pyrokinetics, you should add the following to your
``pyproject.toml`` file::

    [project.entry-points."pyrokinetics.Equilibrium"]
    my_eq = "my_project.my_module:MyEqReader"

This will register the class ``MyEqReader``, and within Pyrokinetics the
equilibrium type will be ``"my_eq"``. Note that here, ``"pyrokinetics.Equilibrium"`` is 
an entry point group name, not a class. The group names for each Pyrokinetics
file reader are:

- ``"pyrokinetics.GKInput"``
- ``"pyrokinetics.GKOutput"``
- ``"pyrokinetics.Equilibrium"``
- ``"pyrokinetics.Kinetics"``

For more information, please see:

- `PyPA entry points specifications
  <https://packaging.python.org/en/latest/specifications/entry-points/>`_
- `Setuptools entry points tutorial
  <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_
"""

from importlib.metadata import entry_points
from typing import Type

from .file_utils import AbstractFileReader, ReadableFromFileMixin

__all__ = ["register_file_reader_plugins", "PluginError"]


class PluginError(TypeError):
    """
    Variant on ``TypeError`` to be thrown when plugins cannot be loaded.

    Parameters
    ----------
    plugin_cls
        The class the user is trying to register as a plugin.
    super_cls
        The Pyrokinetics super class that the plugin class inherits.
    """

    def __init__(self, plugin_cls: Type, super_cls: Type) -> None:
        super().__init__(
            f"{plugin_cls.__qualname__} must subclass {super_cls.__qualname__}"
        )


def register_file_reader_plugins(
    group_name: str, Readable: Type[ReadableFromFileMixin]
) -> None:
    """
    Defines an entry point group for classes that implement the
    :class:`~pyrokinetics.file_utils.AbstractFileReader` and registers any
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
        cls = entry.load()
        if not issubclass(cls, AbstractFileReader):
            raise PluginError(cls, AbstractFileReader)
        Readable.reader(entry.name)(cls)
