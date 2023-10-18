from .kinetics import Kinetics, read_kinetics, supported_kinetics_types

# Import each built-in reader to register them with Kinetics
from .scene import KineticsReaderSCENE  # noqa
from .jetto import KineticsReaderJETTO  # noqa
from .transp import KineticsReaderTRANSP  # noqa
from .pfile import KineticsReaderpFile  # noqa
from .gacode import KineticsReaderGACODE  # noqa

# Register external plugins with Kinetics
from ..plugins import register_file_reader_plugins

register_file_reader_plugins("kinetics", Kinetics)

__all__ = ["Kinetics", "read_kinetics", "supported_kinetics_types"]
