"""KineticsReader

Defines abstract base class for KineticsReader objects, which are capable of
parsing various types of kinetics data. Defines also a factory for objects
derived from KineticsReader
"""

from ..readers import Reader, create_reader_factory


class KineticsReader(Reader):
    pass


# Create global instance of reader factory
kinetics_readers = create_reader_factory(BaseReader=KineticsReader)
