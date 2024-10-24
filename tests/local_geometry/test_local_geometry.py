import numpy as np

import pyrokinetics as pk


def test_normalise():
    """Test that a local geometry can be renormalised with simulation units."""
    pyro = pk.Pyro(gk_file=pk.gk_templates["GS2"])
    geometry = pyro.local_geometry
    norms = pyro.norms

    Rmaj = geometry.Rmaj
    a_minor = geometry.a_minor
    assert Rmaj.units == norms.units.lref_minor_radius
    assert a_minor.units == norms.units.lref_minor_radius

    # Convert to a different units standard
    # LocalGeometry.normalise() is an in-place operation
    geometry = geometry.normalise(norms.gene)
    assert np.isfinite(geometry.Rmaj.magnitude)
    assert np.isfinite(geometry.a_minor.magnitude)
    assert geometry.Rmaj.units == norms.units.lref_major_radius
    assert geometry.a_minor.units == norms.units.lref_major_radius
    assert (geometry.Rmaj / geometry.a_minor) == (Rmaj / a_minor)
