# Import main Kinetics class
from .Kinetics import Kinetics

# Import KineticsReaders
from .KineticsReader import kinetics_readers
from .KineticsReaderSCENE import KineticsReaderSCENE
from .KineticsReaderJETTO import KineticsReaderJETTO
from .KineticsReaderTRANSP import KineticsReaderTRANSP

# Register each reader type with factory
kinetics_readers["SCENE"] = KineticsReaderSCENE
kinetics_readers["JETTO"] = KineticsReaderJETTO
kinetics_readers["TRANSP"] = KineticsReaderTRANSP

__all__ = ["Kinetics"]
