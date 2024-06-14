"""These tests are used to demonstrate experimental new features.

To call these tests in isolation, use::

    $ pytest -k test_simulation
"""

import numpy as np

import pyrokinetics as pk
from pyrokinetics.gk_code import GKInputGS2
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.units import ureg


def test_simulation_to_gk_code():
    # It is expected that in a future build, the following three lines would be
    # replaced by a simple call to ``read_gk_file(filename) -> LocalGKSimulation``.
    input = GKInputGS2()
    input.read_from_file(pk.gk_templates["GS2"])
    gs2_sim = input.get_simulation()

    # Check geometry params
    assert gs2_sim._norms.lref == gs2_sim._norms.units.lref_minor_radius
    assert gs2_sim._geometry.Rmaj.units == gs2_sim._norms.lref
    assert gs2_sim._geometry.a_minor.units == gs2_sim._norms.lref
    gs2_aspect = gs2_sim._geometry.Rmaj / gs2_sim._geometry.a_minor
    assert gs2_sim._norms.bref == gs2_sim._norms.units.bref_B0
    assert gs2_sim._geometry.B0.units == gs2_sim._norms.bref
    gs2_bratio = gs2_sim._geometry.bunit_over_b0

    # Check species params
    gs2_ion_species = gs2_sim._norms.ion_species
    gs2_electron = gs2_sim._species["electron"]
    gs2_ion = gs2_sim._species[gs2_ion_species]
    assert gs2_sim._norms.tref == gs2_sim._norms.units.tref_electron
    assert gs2_electron["temp"].units == gs2_sim._norms.tref
    assert gs2_ion["temp"].units == gs2_sim._norms.tref
    gs2_tratio = gs2_electron["temp"] / gs2_ion["temp"]
    assert gs2_sim._norms.nref == gs2_sim._norms.units.nref_electron
    assert gs2_electron["dens"].units == gs2_sim._norms.nref
    assert gs2_ion["dens"].units == gs2_sim._norms.nref
    gs2_nratio = gs2_electron["dens"] / gs2_ion["dens"]
    # We get physical units in mref, so the following doesn't hold
    # assert gs2_sim._norms.mref == gs2_sim._norms.units.mref_deuterium
    assert gs2_electron["mass"].units == gs2_sim._norms.mref
    assert gs2_ion["mass"].units == gs2_sim._norms.mref
    gs2_mratio = gs2_electron["mass"] / gs2_ion["mass"]

    # Convert to a new gyrokinetics convention
    gene_sim = gs2_sim.to_gk_code("GENE")

    # Check that we have generated entirely new objects
    # N.B. Not checking uniqueness all the way down the stack here
    assert id(gene_sim) != id(gs2_sim)
    assert id(gene_sim._norms) != id(gs2_sim._norms)
    assert id(gene_sim._geometry) != id(gs2_sim._geometry)
    assert id(gene_sim._species) != id(gs2_sim._species)
    assert id(gene_sim._numerics) != id(gs2_sim._numerics)

    # Check that units are doing what we expect
    assert gene_sim._norms.lref == gene_sim._norms.units.lref_major_radius
    assert gene_sim._norms.lref == gs2_sim._norms.units.lref_major_radius
    assert gene_sim._geometry.Rmaj.units == gene_sim._norms.lref
    assert gene_sim._geometry.a_minor.units == gene_sim._norms.lref
    gene_aspect = gene_sim._geometry.Rmaj / gene_sim._geometry.a_minor
    assert gs2_aspect == gene_aspect
    assert gene_sim._norms.bref == gs2_sim._norms.bref
    assert gene_sim._geometry.B0.units == gene_sim._norms.bref
    gene_bratio = gene_sim._geometry.bunit_over_b0
    assert gs2_bratio == gene_bratio

    gene_ion_species = gene_sim._norms.ion_species
    gene_electron = gene_sim._species["electron"]
    gene_ion = gene_sim._species[gene_ion_species]
    assert gene_sim._norms.tref == gs2_sim._norms.tref
    assert gene_electron["temp"].units == gene_sim._norms.tref
    assert gene_ion["temp"].units == gene_sim._norms.tref
    gene_tratio = gene_electron["temp"] / gene_ion["temp"]
    assert gs2_tratio == gene_tratio
    assert gene_sim._norms.nref == gs2_sim._norms.nref
    assert gene_electron["dens"].units == gene_sim._norms.nref
    assert gene_ion["dens"].units == gene_sim._norms.nref
    gene_nratio = gene_electron["dens"] / gene_ion["dens"]
    assert gs2_nratio == gene_nratio
    assert (1.0 * gene_sim._norms.mref).to(ureg.kg) == (1.0 * gs2_sim._norms.mref).to(
        ureg.kg
    )
    assert gene_electron["mass"].units == gene_sim._norms.mref
    assert gene_ion["mass"].units == gene_sim._norms.mref
    gene_mratio = gene_electron["mass"] / gene_ion["mass"]
    assert gs2_mratio == gene_mratio


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
    assert id(new_sim._species) != id(sim._species)
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


def test_simulation_with_species():
    # It is expected that in a future build, the following three lines would be
    # replaced by a simple call to ``read_gk_file(filename) -> LocalGKSimulation``.
    input = GKInputGS2()
    name = "xpfhbj"  # Don't copy paste this for other tests, need it to be unique
    input.read_from_file(pk.gk_templates["GS2"])
    sim = input.get_simulation(name=name)

    # Ensure we're using simulation units
    assert sim._norms.lref == sim._norms.units.lref_minor_radius
    assert sim._geometry.Rmaj.units == sim._norms.lref
    assert sim._geometry.a_minor.units == sim._norms.lref
    assert sim._name == f"{name}000000"

    # Need to read in physical equilibrium before we can read from kinetics
    eq = pk.read_equilibrium(pk.eq_templates["GEQDSK"], file_type="GEQDSK")
    geometry = LocalGeometryMiller()
    geometry.from_global_eq_no_normalise(eq, psi_n=0.5)
    assert geometry.Rmaj.units == ureg.meter
    assert geometry.a_minor.units == ureg.meter
    assert geometry.B0.units == ureg.tesla

    kinetics = pk.read_kinetics(pk.kinetics_templates["TRANSP"], file_type="TRANSP")
    species = LocalSpecies()
    species.from_kinetics_no_normalise(kinetics, psi_n=0.5)
    assert species["electron"]["dens"].units == ureg.meter**-3
    assert species["electron"]["temp"].units == ureg.eV

    new_sim = sim.with_geometry(geometry).with_species(
        species, ion_reference_species="deuterium"
    )
    assert id(new_sim) != id(sim)
    assert id(new_sim._norms) != id(sim._norms)
    assert id(new_sim._geometry) != id(sim._geometry)
    assert id(new_sim._species) != id(sim._species)
    assert id(new_sim._numerics) != id(sim._numerics)
    assert new_sim._name == f"{name}000002"  # 0000001 was made by with_geometry
    assert new_sim._norms.lref == new_sim._norms.units.lref_minor_radius_xpfhbj000002
    assert new_sim._norms.bref == new_sim._norms.units.bref_B0_xpfhbj000002
    assert new_sim._norms.tref == new_sim._norms.units.tref_electron_xpfhbj000002
    assert new_sim._norms.nref == new_sim._norms.units.nref_electron_xpfhbj000002
    assert new_sim._norms.mref == new_sim._norms.units.mref_deuterium_xpfhbj000002
