import pyrokinetics as pk
from pyrokinetics.normalisation import (
    ureg,
    SimulationNormalisation,
    PyroNormalisationError,
)
from pyrokinetics.local_geometry import LocalGeometry
from pyrokinetics.kinetics import Kinetics
from pyrokinetics.templates import gk_gene_template, gk_cgyro_template, gk_gs2_template

import numpy as np

import pytest


@pytest.fixture(scope="module")
def kinetics():
    # Currently no nice way to construct a Kinetics object _not_ from file
    return Kinetics(pk.template_dir / "jetto.cdf")


@pytest.fixture(scope="module")
def geometry():
    return LocalGeometry({"a_minor": 2.3, "B0": 1.2, "bunit_over_b0": 2, "Rmaj": 4.6})


def test_as_system_context_manager():

    ureg.default_system = "mks"
    quantity = 1 * ureg.metre

    with ureg.as_system("imperial"):
        assert quantity.to_base_units() == (1 * ureg.metre).to(ureg.yards)

    assert quantity.to_base_units() == quantity


def test_convert_velocities():
    velocity = 1 * ureg.vref_nrl

    assert velocity.to(ureg.vref_most_probable).m == 1.0 / np.sqrt(2)
    assert velocity.to(ureg.vref_most_probable).to(ureg.vref_nrl) == velocity


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
    assert q.to(norm.gene.lref) == norm.gene.lref / 4.6


def test_set_kinetic(kinetics):
    norm = SimulationNormalisation("test")
    norm.set_kinetic_references(kinetics, psi_n=0.5)

    assert np.isclose(1 * norm.tref, 87271046.22767112 * norm.units.kelvin)
    assert np.isclose(1 * norm.nref, 2.0855866269392273e20 / norm.units.metres ** 3)
    assert np.isclose(1 * norm.mref, 1 * norm.units.deuterium_mass)


def test_normalisation_constructor(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    velocity = 1 * norm.vref
    velocity_gs2 = velocity.to(norm.gs2.vref)
    expected = (1 / np.sqrt(2)) * norm.gs2.vref
    assert np.isclose(velocity_gs2, expected)


def test_convert_bref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.bref_B0).to(norm) == 1 * norm.bref
    assert (1 * ureg.bref_Bunit).to(norm.cgyro) == 1 * norm.cgyro.bref


def test_convert_lref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.lref_minor_radius).to(norm) == 1 * norm.lref
    assert (1 * ureg.lref_major_radius).to(norm.gene) == 1 * norm.gene.lref
    assert (1 * ureg.lref_minor_radius).to(norm.gene) == norm.gene.lref / 4.6
    assert (1 * ureg.lref_major_radius).to(norm) == 4.6 * norm.lref


def test_convert_mref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.mref_deuterium).to(norm) == 1 * norm.mref


def test_convert_nref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.nref_electron).to(norm) == 1 * norm.nref


def test_convert_nref_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * norm.nref).to(norm.gs2).m == 1


def test_convert_tref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.tref_electron).to(norm) == 1 * norm.tref


def test_convert_vref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.vref_nrl).to(norm) == 1.0 * norm.vref
    # Has to go through vref_nrl, so not exact, loses like 1e-16
    assert np.isclose((1 * ureg.vref_most_probable).to(norm.gs2), 1.0 * norm.gs2.vref)


def test_convert_single_unit_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    length_gs2 = 1 * norm.gs2.lref
    length_gene = length_gs2.to(norm.gene)
    expected_gene = norm.gene.lref / 4.6
    assert np.isclose(length_gene, expected_gene)


def test_convert_mixed_units_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    frequency_gs2 = 1 * norm.gs2.vref / norm.gs2.lref

    frequency = frequency_gs2.to(norm)
    expected = np.sqrt(2) * norm.vref / norm.lref

    frequency_gene = frequency.to(norm.gene)
    expected_gene = 4.6 * np.sqrt(2) * norm.gene.vref / norm.gene.lref

    assert np.isclose(frequency, expected)
    assert np.isclose(frequency_gene, expected_gene)


def test_convert_single_units_simulation_to_physical(geometry, kinetics):
    """Convert directly to physical reference unit"""
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    length = 1 * norm.units.lref_minor_radius
    length_physical = length.to(norm.lref, norm.context)
    length_expected = 1 * norm.lref

    assert length_physical == length_expected


def test_convert_single_units_simulation_to_normalisation(geometry, kinetics):
    """Convert to physical reference unit using norm object"""
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    length = 1 * norm.units.lref_minor_radius

    length_physical = length.to(norm)
    length_expected = 1 * norm.lref
    assert length_physical == length_expected

    length_gene = length.to(norm.gene)
    length_gene_expected = norm.gene.lref / 4.6
    assert length_gene == length_gene_expected


def test_convert_mixed_simulation_units_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    frequency_gs2 = 1 * norm.units.vref_most_probable / norm.units.lref_minor_radius

    frequency = frequency_gs2.to(norm)
    expected = np.sqrt(2) * norm.vref / norm.lref

    frequency_gene = frequency.to(norm.gene)
    expected_gene = 4.6 * np.sqrt(2) * norm.gene.vref / norm.gene.lref

    assert np.isclose(frequency, expected)
    assert np.isclose(frequency_gene, expected_gene)


def test_convert_beta(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert norm.beta.to(norm.cgyro) == norm.cgyro.beta


def test_error_no_reference_value():
    norm = SimulationNormalisation("bad")
    with pytest.raises(PyroNormalisationError):
        (1 * norm.units.lref_minor_radius).to(norm.gene)


def test_convert_tref_between_norms(geometry, kinetics):
    """Test issue #132"""

    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * norm.tref).to(norm.gs2).m == 1


def test_gene_length_normalisation():

    pyro = pk.Pyro(gk_file=gk_gene_template)

    assert (
        pyro.local_species.electron.nu.units == ureg.vref_nrl / ureg.lref_minor_radius
    )
    assert pyro.norms.gene.beta_ref == ureg.beta_ref_ee_B0


def test_gs2_length_normalisation():

    pyro = pk.Pyro(gk_file=gk_gs2_template)

    assert (
        pyro.local_species.electron.nu.units
        == ureg.vref_most_probable / ureg.lref_minor_radius
    )
    assert pyro.norms.gs2.beta_ref == ureg.beta_ref_ee_B0


def test_cgyro_length_normalisation():

    pyro = pk.Pyro(gk_file=gk_cgyro_template)

    assert (
        pyro.local_species.electron.nu.units == ureg.vref_nrl / ureg.lref_minor_radius
    )
    assert pyro.norms.cgyro.beta_ref == ureg.beta_ref_ee_Bunit
