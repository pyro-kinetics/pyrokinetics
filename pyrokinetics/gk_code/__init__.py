# Import GKCode classes
from .GKCode import GKCode, gk_codes
from .GKCodeGS2 import GKCodeGS2
from .GKCodeGENE import GKCodeGENE
from .GKCodeCGYRO import GKCodeCGYRO

# Register each reader type with factory
gk_codes["GS2"] = GKCodeGS2
gk_codes["GENE"] = GKCodeGENE
gk_codes["CGYRO"] = GKCodeCGYRO

from .GKOutput import GKOutput

# Import refactored versions
from .GKInput import GKInput, gk_inputs
from .GKOutputReader import GKOutputReader, gk_output_readers
from .GKInputGS2 import GKInputGS2
from .GKInputCGYRO import GKInputCGYRO
from .GKInputGENE import GKInputGENE
from .GKOutputReaderGS2 import GKOutputReaderGS2

gk_inputs["GS2"] = GKInputGS2
gk_inputs["CGYRO"] = GKInputCGYRO
gk_inputs["GENE"] = GKInputGENE
gk_output_readers["GS2"] = GKOutputReaderGS2


__all__ = ["GKCode", "GKOutput", "GKInput", "GKOutputReader"]
