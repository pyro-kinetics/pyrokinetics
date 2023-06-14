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
from .GKOutputReaderIDS import GKOutputReaderIDS  # noqa

# from . import GKOutputReaderGS2
# from . import GKOutputReaderCGYRO
# from . import GKOutputReaderGENE
# from . import GKOutputReaderTGLF

__all__ = ["GKInput", "gk_inputs", "GKOutput", "supported_gk_output_types"]
# gk_output_readers["GS2"] = GKOutputReaderGS2
# gk_output_readers["CGYRO"] = GKOutputReaderCGYRO
# gk_output_readers["GENE"] = GKOutputReaderGENE
# gk_output_readers["TGLF"] = GKOutputReaderTGLF
