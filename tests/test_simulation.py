"""These tests are used to demonstrate experimental new features.

To call these tests in isolation, use::

    $ pytest -k test_simulation
"""

import pyrokinetics as pk
from pyrokinetics.gk_code import GKInputGS2

def test_simulation_convert_gk_code():
    # It is expected that in a future build, the following three lines would be
    # replaced by a simple call to ``read_gk_file(filename) -> LocalGKSimulation``.
    input = GKInputGS2()
    input.read_from_file(pk.gk_templates["GS2"])
    gs2 = input.get_simulation()
    
    assert gs2._norms.lref == gs2._norms.units.lref_minor_radius
    assert gs2._geometry.Rmaj.units == gs2._norms.lref
    assert gs2._geometry.a_minor.units == gs2._norms.lref
    aspect_gs2 = gs2._geometry.Rmaj / gs2._geometry.a_minor

    gene = gs2.to_gk_code("GENE")
    # Check that we have generated entirely new objects
    # N.B. Not checking uniqueness all the way down the stack here
    assert id(gene) != id(gs2)
    assert id(gene._norms) != id(gs2._norms)
    assert id(gene._geometry) != id(gs2._geometry)
    # Check that units are doing what we expect
    assert gene._norms.lref == gene._norms.units.lref_major_radius
    assert gene._norms.lref == gs2._norms.units.lref_major_radius
    assert gene._geometry.a_minor.units == gene._norms.lref
    assert gene._geometry.a_minor.units == gene._norms.lref
    aspect_gene = gene._geometry.Rmaj / gene._geometry.a_minor
    assert aspect_gs2 == aspect_gene
