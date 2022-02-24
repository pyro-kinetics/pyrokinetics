# Import EquilibriumReaders
from .EquilibriumReader import EquilibriumReader, equilibrium_readers
from .EquilibriumReaderTRANSP import EquilibriumReaderTRANSP
from .EquilibriumReaderGEQDSK import EquilibriumReaderGEQDSK

# Register each reader type with factory
equilibrium_readers["TRANSP"] = EquilibriumReaderTRANSP
equilibrium_readers["GEQDSK"] = EquilibriumReaderGEQDSK

# Import main Equilibrium class
from .Equilibrium import Equilibrium  # noqa: E402

__all__ = ["Equilibrium", "EquilibriumReader"]
