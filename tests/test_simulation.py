"""These tests are used to demonstrate experimental new features.

To call these tests in isolation, use::

    $ pytest -k test_simulation
"""

import numpy as np

import pyrokinetics as pk
from pyrokinetics.gk_code import GKInputGS2
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.units import ureg


def test_simulation_to_gk_code():
    # It is expected that in a future build, the following three lines would be
    # replaced by a simple call to ``read_gk_file(filename) -> LocalGKSimulation``.
    input = GKInputGS2()
    input.read_from_file(pk.gk_templates["GS2"])
    gs2_sim = input.get_simulation()

    assert gs2_sim._norms.lref == gs2_sim._norms.units.lref_minor_radius
    assert gs2_sim._geometry.Rmaj.units == gs2_sim._norms.lref
    assert gs2_sim._geometry.a_minor.units == gs2_sim._norms.lref
    aspect_gs2 = gs2_sim._geometry.Rmaj / gs2_sim._geometry.a_minor

    gene_sim = gs2_sim.to_gk_code("GENE")
    # Check that we have generated entirely new objects
    # N.B. Not checking uniqueness all the way down the stack here
    assert id(gene_sim) != id(gs2_sim)
    assert id(gene_sim._norms) != id(gs2_sim._norms)
    assert id(gene_sim._geometry) != id(gs2_sim._geometry)
    assert id(gene_sim._numerics) != id(gs2_sim._numerics)
    # Check that units are doing what we expect
    # TODO check species
    assert gene_sim._norms.lref == gene_sim._norms.units.lref_major_radius
    assert gene_sim._norms.lref == gs2_sim._norms.units.lref_major_radius
    assert gene_sim._geometry.a_minor.units == gene_sim._norms.lref
    assert gene_sim._geometry.a_minor.units == gene_sim._norms.lref
    aspect_gene = gene_sim._geometry.Rmaj / gene_sim._geometry.a_minor
    assert aspect_gs2 == aspect_gene


def test_simulation_with_geometry():
    # It is expected that in a future build, the following three lines would be
    # replaced by a simple call to ``read_gk_file(filename) -> LocalGKSimulation``.
    input = GKInputGS2()
    name = "xpfhbx"  # Don't copy paste this for other tests, need it to be unique
    input.read_from_file(pk.gk_templates["GS2"])
    sim = input.get_simulation(name=name)

    # Ensure we're using simulation units
    assert sim._norms.lref == sim._norms.units.lref_minor_radius
    assert sim._geometry.Rmaj.units == sim._norms.lref
    assert sim._geometry.a_minor.units == sim._norms.lref
    assert sim._name == f"{name}000000"

    # Add new geometry with physical units
    eq = pk.read_equilibrium(pk.eq_templates["GEQDSK"], file_type="GEQDSK")
    geometry = LocalGeometryMiller()
    geometry.from_global_eq_no_normalise(eq, psi_n=0.5)
    assert geometry.Rmaj.units == ureg.meter
    assert geometry.a_minor.units == ureg.meter
    assert geometry.B0.units == ureg.tesla

    # Create new simulation using this geometry
    new_sim = sim.with_geometry(geometry)
    assert id(new_sim) != id(sim)
    assert id(new_sim._norms) != id(sim._norms)
    assert id(new_sim._geometry) != id(sim._geometry)
    assert id(new_sim._numerics) != id(sim._numerics)
    assert new_sim._name == f"{name}000001"
    assert new_sim._norms.lref == new_sim._norms.units.lref_minor_radius_xpfhbx000001
    assert new_sim._norms.bref == new_sim._norms.units.bref_B0_xpfhbx000001
    assert new_sim._geometry.Rmaj.units == new_sim._norms.lref
    assert new_sim._geometry.a_minor.units == new_sim._norms.lref
    assert new_sim._geometry.B0.units == new_sim._norms.bref
    assert np.isclose(new_sim._geometry.Rmaj.to(ureg.meter), geometry.Rmaj)
    assert np.isclose(new_sim._geometry.a_minor.to(ureg.meter), geometry.a_minor)
    assert np.isclose(new_sim._geometry.B0.to(ureg.tesla), geometry.B0)
