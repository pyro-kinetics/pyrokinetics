from ..plugins import register_file_reader_plugins
from . import geqdsk  # noqa
from . import transp  # noqa
from .equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
    supported_equilibrium_types,
)
from .flux_surface import FluxSurface

pygacode_found: bool
try:
    import pygacode

    pygacode_found = True
    del pygacode
except ImportError:
    pygacode_found = False

if pygacode_found:
    from . import gacode  # noqa

register_file_reader_plugins("equilibrium", Equilibrium)

__all__ = [
    "Equilibrium",
    "EquilibriumCOCOSWarning",
    "read_equilibrium",
    "supported_equilibrium_types",
    "FluxSurface",
]
