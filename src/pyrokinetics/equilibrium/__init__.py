from .equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
    supported_equilibrium_types,
)
from .flux_surface import FluxSurface

# Import each built-in reader to register them with Equilibrium
from . import geqdsk  # noqa
from . import transp  # noqa
from . import gacode  # noqa

# Register external plugins with Equilibrium
from ..plugins import register_file_reader_plugins

register_file_reader_plugins("equilibrium", Equilibrium)

__all__ = [
    "Equilibrium",
    "EquilibriumCOCOSWarning",
    "read_equilibrium",
    "supported_equilibrium_types",
    "FluxSurface",
]
