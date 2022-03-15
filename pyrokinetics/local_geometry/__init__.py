from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMiller import LocalGeometryMiller, get_default_miller_inputs

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMiller

__all__ = ["LocalGeometry", "get_default_miller_inputs"]
