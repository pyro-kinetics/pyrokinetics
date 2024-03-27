from platform import python_version_tuple

# Import built-in input and output readers to register them with GkInput and GkOutput.
from .cgyro import GKInputCGYRO, GKOutputReaderCGYRO  # noqa
from .gene import GKInputGENE, GKOutputReaderGENE  # noqa
from .gk_input import GKInput, read_gk_input, supported_gk_input_types
from .gk_output import GKOutput, read_gk_output, supported_gk_output_types
from .gs2 import GKInputGS2, GKOutputReaderGS2  # noqa
from .stella import GKInputSTELLA#, GKOutputReaderSTELLA  # noqa
from .tglf import GKInputTGLF, GKOutputReaderTGLF  # noqa

# Only import IDS if Python version is greater than 3.9
if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from .ids import GKOutputReaderIDS  # noqa

# Register external plugins
from ..plugins import register_file_reader_plugins

register_file_reader_plugins("gk_input", GKInput)
register_file_reader_plugins("gk_output", GKOutput)

__all__ = [
    "GKInput",
    "GKOutput",
    "read_gk_input",
    "read_gk_output",
    "supported_gk_input_types",
    "supported_gk_output_types",
]
