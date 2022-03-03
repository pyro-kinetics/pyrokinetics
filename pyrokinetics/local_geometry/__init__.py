from .LocalGeometry import LocalGeometry, local_geometries
from .LocalGeometryMiller import LocalGeometryMiller

# Register LocalGeometry objects with factory
local_geometries["Miller"] = LocalGeometryMiller

__all__ = ["LocalGeometry"]
