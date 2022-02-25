# Import GKCode classes
from .GKCode import GKCode, gk_codes
from .GS2 import GS2
from .GENE import GENE
from .CGYRO import CGYRO

# Register each reader type with factory
gk_codes["GS2"] = GS2
gk_codes["GENE"] = GS2
gk_codes["CGYRO"] = GS2

from .GKOutput import GKOutput

__all__ = ["GKCode", "GKOutput"]
