# isort: skip_file

from .kinetics import Kinetics, read_kinetics, supported_kinetics_types

# Import each module to register the associated readers
from .scene import KineticsReaderSCENE  # noqa
from .transp import KineticsReaderTRANSP  # noqa
from .pfile import KineticsReaderpFile  # noqa
from .jetto import KineticsReaderJETTO  # noqa
from .gacode import KineticsReaderGACODE  # noqa

from ..plugins import register_file_reader_plugins

register_file_reader_plugins("kinetics", Kinetics)

__all__ = ["Kinetics", "read_kinetics", "supported_kinetics_types"]
