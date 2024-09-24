from .fourier_cgyro import LocalGeometryFourierCGYRO
from .fourier_gene import LocalGeometryFourierGENE
from .local_geometry import LocalGeometry, local_geometry_factory
from .metric import MetricTerms
from .miller import LocalGeometryMiller
from .miller_turnbull import LocalGeometryMillerTurnbull
from .mxh import LocalGeometryMXH

# Register LocalGeometry objects with factory
local_geometry_factory["MillerTurnbull"] = LocalGeometryMillerTurnbull
local_geometry_factory["Miller"] = LocalGeometryMiller
local_geometry_factory["FourierGENE"] = LocalGeometryFourierGENE
local_geometry_factory["MXH"] = LocalGeometryMXH
local_geometry_factory["FourierCGYRO"] = LocalGeometryFourierCGYRO

__all__ = [
    "LocalGeometry",
    "local_geometry_factory",
    "MetricTerms",
]
