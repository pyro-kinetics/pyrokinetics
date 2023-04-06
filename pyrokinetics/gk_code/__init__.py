from .GKInput import GKInput, gk_inputs
from .GKInputGS2 import GKInputGS2
from .GKInputCGYRO import GKInputCGYRO
from .GKInputGENE import GKInputGENE
from .GKInputTGLF import GKInputTGLF

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_inputs["TGLF"] = GKInputTGLF

from .gk_output import GKOutput  # noqa
from .GKOutputReaderGS2 import GKOutputReaderGS2  # noqa
