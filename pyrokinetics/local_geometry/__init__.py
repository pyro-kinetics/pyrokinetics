from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMiller import LocalGeometryMiller, default_miller_inputs
from .LocalGeometryFourier import LocalGeometryFourier, default_fourier_inputs

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMiller
local_geometries["Fourier"] = LocalGeometryFourier

__all__ = ["LocalGeometry", "default_miller_inputs", "default_fourier_inputs"]
