import pyrokinetics as pk
from pyrokinetics.normalisation import (
    ureg,
    SimulationNormalisation,
    _create_unit_registry,
    simulation_context,
)
from pyrokinetics.local_geometry import LocalGeometry
from pyrokinetics.kinetics import Kinetics

import numpy as np

import pytest


@pytest.fixture(autouse=True)
def reset_unit_registry():
    # This doesn't really work properly, because
    # SimulationNormalisation takes the module level variable as a
    # default argument, and we don't reset _that_.

    # Maybe we can instead remove created units from the registry?
    pk.normalisation.ureg = _create_unit_registry(simulation_context)


@pytest.fixture(scope="module")
def kinetics():
    # Currently no nice way to construct a Kinetics object _not_ from file
    return Kinetics(pk.template_dir / "jetto.cdf")


@pytest.fixture(scope="module")
def geometry():
    return LocalGeometry({"a_minor": 2.3, "B0": 1.2, "bunit_over_b0": 2, "Rmaj": 4.6})


def test_as_system_context_manager():
    quantity = 1 * ureg.metre

    with ureg.as_system("imperial"):
        assert quantity.to_base_units() == (1 * ureg.metre).to(ureg.yards)

    assert quantity.to_base_units() == quantity


def test_convert_velocities():
    velocity = 1 * ureg.vref_nrl

    assert velocity.to(ureg.vref_most_probable).m == np.sqrt(2)
    assert velocity.to(ureg.vref_most_probable).to(ureg.vref_nrl) == velocity


def test_convert_lengths():
    r_minor = 1 * ureg.lref_minor_radius
    r_major = 1 * ureg.lref_major_radius

    with ureg.context("simulation_context", aspect_ratio=2):
        assert r_minor.to(ureg.lref_major_radius) == 2.0 * ureg.lref_major_radius
        assert r_major.to(ureg.lref_minor_radius) == 0.5 * ureg.lref_minor_radius

        assert r_minor.to("lref_major_radius").to("lref_minor_radius") == r_minor


def test_switch_convention():
    norm = SimulationNormalisation("test")

    assert norm.lref == norm.pyrokinetics.lref

    norm.default_convention = "imas"

    assert norm.lref == norm.imas.lref


def test_set_bref(geometry):
    norm = SimulationNormalisation("test")
    norm.set_bref(geometry)

    q = 1 * norm.bref
    assert q.to("tesla") == 1.2 * ureg.tesla
    assert q.to(norm.cgyro.bref) == 0.5 * norm.cgyro.bref


def test_set_lref(geometry):
    norm = SimulationNormalisation("test")
    norm.set_lref(geometry)

    q = 1 * norm.lref
    assert q.to("m") == 2.3 * ureg.metres
    assert q.to(norm.gene.lref) == 0.5 * norm.gene.lref


def test_set_kinetic(kinetics):
    norm = SimulationNormalisation("test")
    norm.set_kinetic_references(kinetics, psi_n=0.5)

    assert np.isclose(1 * norm.tref, 87271046.22767112 * norm.units.kelvin)
    assert np.isclose(1 * norm.nref, 2.0855866269392273e20 / norm.units.metres**3)
    assert np.isclose(1 * norm.mref, 1.0004964957043108 * norm.units.deuterium_mass)


def test_normalisation_constructor(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    velocity = 1 * norm.vref
    velocity_gs2 = velocity.to(norm.gs2.vref)
    expected = (1 / np.sqrt(2)) * norm.gs2.vref
    assert np.isclose(velocity_gs2, expected)


def test_convert_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    frequency_gs2 = 1 * norm.gs2.vref / norm.gs2.lref

    frequency = frequency_gs2.to(norm)
    expected = np.sqrt(2) * norm.vref / norm.lref

    frequency_gene = frequency.to(norm.gene)
    expected_gene = 2 * np.sqrt(2) * norm.gene.vref / norm.gene.lref

    assert np.isclose(frequency, expected)
    assert np.isclose(frequency_gene, expected_gene)


def test_convert_simulation_to_physical(geometry, kinetics):
    """Convert directly to physical reference unit"""
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    length = 1 * norm.units.lref_minor_radius
    length_physical = length.to(norm.lref, norm.context)
    length_expected = 1 * norm.lref

    assert length_physical == length_expected


def test_convert_simulation_to_normalisation(geometry, kinetics):
    """Convert to physical reference unit using norm object"""
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    length = 1 * norm.units.lref_minor_radius

    length_physical = length.to(norm)
    length_expected = 1 * norm.lref
    assert length_physical == length_expected

    length_gene = length.to(norm.gene)
    length_gene_expected = 0.5 * norm.gene.lref
    assert length_gene == length_gene_expected
