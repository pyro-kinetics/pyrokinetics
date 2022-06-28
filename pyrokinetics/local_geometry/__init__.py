from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMiller import LocalGeometryMiller, default_miller_inputs

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMiller

__all__ = ["LocalGeometry", "default_miller_inputs"]
