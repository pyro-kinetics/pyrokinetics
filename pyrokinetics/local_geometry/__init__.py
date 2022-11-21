from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMiller import LocalGeometryMiller, default_miller_inputs
from .LocalGeometryFourierGENE import LocalGeometryFourierGENE, default_fourier_gene_inputs
from .LocalGeometryFourierCGYRO import (
    LocalGeometryFourierCGYRO,
    default_fourier_cgyro_inputs,
)
from .LocalGeometryMXH import LocalGeometryMXH, default_mxh_inputs

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMiller
local_geometries["FourierGENE"] = LocalGeometryFourierGENE
local_geometries["MXH"] = LocalGeometryMXH
local_geometries["FourierCGYRO"] = LocalGeometryFourierCGYRO

__all__ = [
    "LocalGeometry",
    "default_miller_inputs",
    "default_fourier_gene_inputs",
    "default_mxh_inputs",
    "default_fourier_cgyro_inputs",
]
