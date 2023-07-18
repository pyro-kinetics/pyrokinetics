from .cgyro import GKInputCGYRO, GKOutputReaderCGYRO  # noqa
from .gene import GKInputGENE, GKOutputReaderGENE  # noqa
from .gs2 import GKInputGS2, GKOutputReaderGS2  # noqa
from .tglf import GKInputTGLF, GKOutputReaderTGLF  # noqa
from .gk_input import GKInput, gk_inputs
from .gk_output import GKOutput, supported_gk_output_types  # noqa

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_inputs["TGLF"] = GKInputTGLF

__all__ = ["GKInput", "gk_inputs", "GKOutput", "supported_gk_output_types"]
