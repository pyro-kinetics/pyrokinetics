from platform import python_version_tuple

from .cgyro import GKInputCGYRO, GKOutputReaderCGYRO  # noqa
from .gene import GKInputGENE, GKOutputReaderGENE  # noqa
from .gkw import GKInputGKW, GKOutputReaderGKW  # noqa
from .gs2 import GKInputGS2, GKOutputReaderGS2  # noqa
from .tglf import GKInputTGLF, GKOutputReaderTGLF  # noqa
from .gk_input import GKInput, read_gk_input, supported_gk_input_types
from .gk_output import GKOutput, read_gk_output, supported_gk_output_types

# Only import IDS if Python version is greater than 3.9
if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from .ids import GKOutputReaderIDS  # noqa

__all__ = [
    "GKInput",
    "GKOutput",
    "read_gk_input",
    "read_gk_output",
    "supported_gk_input_types",
    "supported_gk_output_types",
]
