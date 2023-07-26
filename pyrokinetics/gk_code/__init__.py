from platform import python_version_tuple

from .GKInput import GKInput, gk_inputs
from .GKInputGS2 import GKInputGS2
from .GKInputCGYRO import GKInputCGYRO
from .GKInputGENE import GKInputGENE
from .GKInputTGLF import GKInputTGLF

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_inputs["TGLF"] = GKInputTGLF

from .gk_output import GKOutput, supported_gk_output_types  # noqa
from .GKOutputReaderGS2 import GKOutputReaderGS2  # noqa
from .GKOutputReaderCGYRO import GKOutputReaderCGYRO  # noqa
from .GKOutputReaderGENE import GKOutputReaderGENE  # noqa
from .GKOutputReaderTGLF import GKOutputReaderTGLF  # noqa

# Only import IDS if Python version is greater than 3.9
if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from .GKOutputReaderIDS import GKOutputReaderIDS  # noqa

__all__ = ["GKInput", "gk_inputs", "GKOutput", "supported_gk_output_types"]
