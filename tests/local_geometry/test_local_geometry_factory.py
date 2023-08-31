import pytest

from pyrokinetics.local_geometry import local_geometry_factory, LocalGeometryMiller


def test_miller():
    """Ensure a miller object is created successfully"""
    miller = local_geometry_factory("Miller")
    assert isinstance(miller, LocalGeometryMiller)


def test_non_local_geometry():
    """Ensure failure when getting a non-existent LocalGeometry"""
    with pytest.raises(KeyError):
        local_geometry_factory("Hello world")
