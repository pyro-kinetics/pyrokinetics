from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMillerTurnbull import LocalGeometryMillerTurnbull, default_miller_turnbull_inputs
from .LocalGeometryBasicMiller import LocalGeometryBasicMiller, default_basic_miller_inputs
from .LocalGeometryFourierGENE import (
    LocalGeometryFourierGENE,
    default_fourier_gene_inputs,
)
from .LocalGeometryFourierCGYRO import (
    LocalGeometryFourierCGYRO,
    default_fourier_cgyro_inputs,
)
from .LocalGeometryMXH import LocalGeometryMXH, default_mxh_inputs

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMillerTurnbull
local_geometries["BasicMiller"] = LocalGeometryBasicMiller
local_geometries["FourierGENE"] = LocalGeometryFourierGENE
local_geometries["MXH"] = LocalGeometryMXH
local_geometries["FourierCGYRO"] = LocalGeometryFourierCGYRO

__all__ = [
    "LocalGeometry",
    "default_basic_miller_inputs",
    "default_miller_turnbull_inputs",
    "default_fourier_gene_inputs",
    "default_mxh_inputs",
    "default_fourier_cgyro_inputs",
]
