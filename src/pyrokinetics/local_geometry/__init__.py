from .fourier_cgyro import LocalGeometryFourierCGYRO, default_fourier_cgyro_inputs
from .fourier_gene import LocalGeometryFourierGENE, default_fourier_gene_inputs
from .local_geometry import LocalGeometry, local_geometry_factory
from .metric import MetricTerms
from .miller import LocalGeometryMiller, default_miller_inputs
from .miller_turnbull import LocalGeometryMillerTurnbull, default_miller_turnbull_inputs
from .mxh import LocalGeometryMXH, default_mxh_inputs

# Register LocalGeometry objects with factory
local_geometry_factory["MillerTurnbull"] = LocalGeometryMillerTurnbull
local_geometry_factory["Miller"] = LocalGeometryMiller
local_geometry_factory["FourierGENE"] = LocalGeometryFourierGENE
local_geometry_factory["MXH"] = LocalGeometryMXH
local_geometry_factory["FourierCGYRO"] = LocalGeometryFourierCGYRO

__all__ = [
    "LocalGeometry",
    "local_geometry_factory",
    "default_miller_inputs",
    "default_miller_turnbull_inputs",
    "default_fourier_gene_inputs",
    "default_mxh_inputs",
    "default_fourier_cgyro_inputs",
    "MetricTerms",
]
