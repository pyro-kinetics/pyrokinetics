from platform import python_version_tuple

from .cgyro import GKInputCGYRO, GKOutputReaderCGYRO  # noqa
from .gene import GKInputGENE, GKOutputReaderGENE  # noqa
from .gs2 import GKInputGS2, GKOutputReaderGS2  # noqa
from .tglf import GKInputTGLF, GKOutputReaderTGLF  # noqa
from .gk_input import GKInput, gk_inputs
from .gk_output import GKOutput, supported_gk_output_types  # noqa

# Only import IDS if Python version is greater than 3.9
if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from .GKOutputReaderIDS import GKOutputReaderIDS  # noqa

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_inputs["TGLF"] = GKInputTGLF

__all__ = ["GKInput", "gk_inputs", "GKOutput", "supported_gk_output_types"]
