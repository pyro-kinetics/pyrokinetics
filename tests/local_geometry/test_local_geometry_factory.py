from pyrokinetics.local_geometry import local_geometries, LocalGeometryMiller
import pytest


def test_miller():
    """Ensure a miller object is created successfully"""
    miller = local_geometries["Miller"]
    assert isinstance(miller, LocalGeometryMiller)


def test_non_local_geometry():
    """Ensure failure when getting a non-existent LocalGeometry"""
    with pytest.raises(KeyError):
        local_geometries["Hello world"]
