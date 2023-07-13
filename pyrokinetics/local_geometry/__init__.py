from .local_geometry import LocalGeometry, local_geometries
from .miller_turnbull import LocalGeometryMillerTurnbull, default_miller_turnbull_inputs
from .miller import LocalGeometryMiller, default_miller_inputs
from .fourier_gene import LocalGeometryFourierGENE, default_fourier_gene_inputs
from .fourier_cgyro import LocalGeometryFourierCGYRO, default_fourier_cgyro_inputs
from .mxh import LocalGeometryMXH, default_mxh_inputs
from .metric import MetricTerms

# Register LocalGeometry objects with factory
local_geometries["MillerTurnbull"] = LocalGeometryMillerTurnbull
local_geometries["Miller"] = LocalGeometryMiller
local_geometries["FourierGENE"] = LocalGeometryFourierGENE
local_geometries["MXH"] = LocalGeometryMXH
local_geometries["FourierCGYRO"] = LocalGeometryFourierCGYRO

__all__ = [
    "LocalGeometry",
    "default_miller_inputs",
    "default_miller_turnbull_inputs",
    "default_fourier_gene_inputs",
    "default_mxh_inputs",
    "default_fourier_cgyro_inputs",
    "MetricTerms",
]
