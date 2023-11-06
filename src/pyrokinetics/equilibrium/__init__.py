# isort: skip_file

from .equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
    supported_equilibrium_types,
)
from .flux_surface import FluxSurface

# Import each module to register the associated readers
from . import geqdsk  # noqa
from . import transp  # noqa
from . import gacode  # noqa

from ..plugins import register_file_reader_plugins

register_file_reader_plugins("equilibrium", Equilibrium)

__all__ = [
    "Equilibrium",
    "EquilibriumCOCOSWarning",
    "read_equilibrium",
    "supported_equilibrium_types",
    "FluxSurface",
]
