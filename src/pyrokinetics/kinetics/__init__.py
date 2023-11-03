from ..plugins import register_file_reader_plugins
from .jetto import KineticsReaderJETTO  # noqa
from .kinetics import Kinetics, read_kinetics, supported_kinetics_types
from .pfile import KineticsReaderpFile  # noqa
from .scene import KineticsReaderSCENE  # noqa
from .transp import KineticsReaderTRANSP  # noqa

pygacode_found: bool
try:
    import pygacode

    pygacode_found = True
    del pygacode
except ImportError:
    pygacode_found = False

if pygacode_found:
    from .gacode import KineticsReaderGACODE  # noqa

register_file_reader_plugins("kinetics", Kinetics)

__all__ = ["Kinetics", "read_kinetics", "supported_kinetics_types"]
