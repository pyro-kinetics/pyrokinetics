from .gk_input import GKInput, gk_inputs
from .gs2 import GKInputGS2
from .cgyro import GKInputCGYRO
from .gene import GKInputGENE
from .tglf import GKInputTGLF

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_inputs["TGLF"] = GKInputTGLF

from .gk_output import GKOutput, supported_gk_output_types  # noqa
from .GKOutputReaderGS2 import GKOutputReaderGS2  # noqa
from .GKOutputReaderCGYRO import GKOutputReaderCGYRO  # noqa
from .GKOutputReaderGENE import GKOutputReaderGENE  # noqa
from .GKOutputReaderTGLF import GKOutputReaderTGLF  # noqa

__all__ = ["GKInput", "gk_inputs", "GKOutput", "supported_gk_output_types"]
