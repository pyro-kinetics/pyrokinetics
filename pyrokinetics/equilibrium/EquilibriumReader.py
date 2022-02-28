"""EquilibriumReader

Defines abstract base class for EquilibriumReader objects, which are capable of
parsing information about tokamak geometry at equilibrium.

Defines also a factory for objects derived from EquilibriumReader
"""

from ..readers import Reader, create_reader_factory


class EquilibriumReader(Reader):
    pass


# Create global instance of reader factory
equilibrium_readers = create_reader_factory(BaseReader=EquilibriumReader)
