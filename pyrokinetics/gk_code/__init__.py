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

__all__ = ["GKCode", "GKOutput"]
