# Import EquilibriumReaders
from .EquilibriumReader import EquilibriumReader, equilibrium_readers
from .EquilibriumReaderTRANSP import EquilibriumReaderTRANSP
from .EquilibriumReaderGEQDSK import EquilibriumReaderGEQDSK

# Register each reader type with factory
equilibrium_readers["TRANSP"] = EquilibriumReaderTRANSP
equilibrium_readers["GEQDSK"] = EquilibriumReaderGEQDSK

# Import main Equilibrium class
from .Equilibrium import Equilibrium

from .equilibrium import (
    Equilibrium as EquilibriumNew,
    read_equilibrium,
    equilibrium_reader,
    supported_equilibrium_types,
)
from .flux_surface import FluxSurface
from . import geqdsk  # noqa
from . import transp  # noqa

__all__ = ["Equilibrium", "EquilibriumReader"]
