import pyrokinetics as pk
from pyrokinetics.normalisation import (
    ureg,
    SimulationNormalisation,
    PyroNormalisationError,
)
from pyrokinetics.local_geometry import LocalGeometry
from pyrokinetics.kinetics import read_kinetics
from pyrokinetics.templates import gk_gene_template, gk_cgyro_template, gk_gs2_template
from pyrokinetics.constants import electron_mass, deuterium_mass
from pyrokinetics.gk_code import GKInputGS2

import numpy as np

import pytest


@pytest.fixture(scope="module")
def kinetics():
    # Currently no nice way to construct a Kinetics object _not_ from file
    return read_kinetics(pk.template_dir / "jetto.jsp", "JETTO")


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

    base = 1 * norm.units.bref_B0
    assert base.to(norm) == q


def test_set_lref(geometry):
    norm = SimulationNormalisation("test")
    norm.set_lref(geometry)

    q = 1 * norm.lref
    assert q.to("m") == 2.3 * ureg.metres
    assert q.to(norm.gene.lref) == norm.gene.lref / 4.6

    base = 1 * norm.units.lref_minor_radius
    assert base.to(norm) == q


def test_set_kinetic(kinetics):
    norm = SimulationNormalisation("test")
    norm.set_kinetic_references(kinetics, psi_n=0.5)

    assert np.isclose(1 * norm.tref, 23774277.31113508 * norm.units.kelvin)
    assert np.isclose(1 * norm.nref, 3.98442302e19 / norm.units.metres**3)
    assert np.isclose(1 * norm.mref, 1 * norm.units.deuterium_mass)

    base_tref_electron = 1 * norm.units.tref_electron
    base_nref_electron = 1 * norm.units.nref_electron
    base_mref_deuterium = 1 * norm.units.mref_deuterium

    assert np.isclose(
        base_tref_electron.to(norm), 23774277.31113508 * norm.units.kelvin
    )
    assert np.isclose(base_nref_electron.to(norm), 3.98442302e19 / norm.units.metres**3)
    assert np.isclose(base_mref_deuterium.to(norm), 1 * norm.units.deuterium_mass)


def test_set_all_references():
    pyro = pk.Pyro(gk_file=gk_gs2_template)
    norm = SimulationNormalisation("test")

    reference_values = {
        "tref_electron": 1000.0 * norm.units.eV,
        "nref_electron": 1e19 * norm.units.meter**-3,
        "lref_minor_radius": 1.5 * norm.units.meter,
        "bref_B0": 2.0 * norm.units.tesla,
    }

    norm.set_all_references(pyro, **reference_values)

    assert np.isclose(1 * norm.tref, reference_values["tref_electron"])
    assert np.isclose(1 * norm.lref, reference_values["lref_minor_radius"])
    assert np.isclose(1 * norm.bref, reference_values["bref_B0"])

    base_tref_electron = 1 * norm.units.tref_electron
    base_nref_electron = 1 * norm.units.nref_electron
    base_lref_minor_radius = 1 * norm.units.lref_minor_radius
    base_bref_B0 = 1 * norm.units.bref_B0

    assert np.isclose(base_tref_electron.to(norm), reference_values["tref_electron"])
    assert np.isclose(
        base_lref_minor_radius.to(norm), reference_values["lref_minor_radius"]
    )
    assert np.isclose(base_bref_B0.to(norm), reference_values["bref_B0"])

    # Had to convert density to SI. Not sure why
    assert np.isclose(
        (1 * norm.nref).to("meter**-3"), reference_values["nref_electron"]
    )
    assert np.isclose(
        base_nref_electron.to(norm).to("meter**-3"), reference_values["nref_electron"]
    )


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
        pyro.local_species.electron.nu.to(pyro.norms.gs2).units
        == ureg.vref_most_probable / ureg.lref_minor_radius
    )
    assert pyro.norms.gs2.beta_ref == ureg.beta_ref_ee_B0


def test_cgyro_length_normalisation():
    pyro = pk.Pyro(gk_file=gk_cgyro_template)

    assert (
        pyro.local_species.electron.nu.units == ureg.vref_nrl / ureg.lref_minor_radius
    )
    assert pyro.norms.cgyro.beta_ref == ureg.beta_ref_ee_Bunit


def get_gs2_basic_dict(e_mass=0.0002724437107, electron_temp=1.0, electron_dens=1.0, Rmaj=3.0, Rgeo_Rmaj=1.0):

    d_mass = (deuterium_mass / electron_mass).m * e_mass
    c_mass = 12 * d_mass
    # species data
    data = {
        "species_knobs": {"nspec": 3},
        "species_parameters_1": {"type": "electron", "z": -1, "mass": e_mass, "temp": electron_temp, "dens": electron_dens},
        "species_parameters_2": {"type": "ion", "z": 1, "mass": d_mass, "temp": 2 * electron_temp, "dens": electron_dens * 5.0 / 6.0},
        "species_parameters_3": {"type": "ion", "z": 6, "mass": c_mass, "temp": 2 * electron_temp, "dens": electron_dens * 1.0 / 6.0},
        "theta_grid_parameters": {"rmaj": Rmaj, "r_geo": Rgeo_Rmaj * Rmaj},
        "theta_grid_eik_knobs": {"irho": 2},
    }

    return data


e_mass_opts = {"deuterium": 0.0002724437107, "hydrogen": 0.0005448874215, "electron": 1.0, "failure" :0.5}
e_temp_opts = {"electron": 1.0, "deuterium": 0.5, "failure": 2.0}
e_dens_opts = {"electron": 1.0, "deuterium": 6.0 / 5.0, "failure": 0.5}
rmaj_opts = {"major_radius": 1.0, "minor_radius": 3.0}
rgeo_rmaj_opts = {"B0": 1.0, "Bgeo": 1.1}


def test_non_standard_normalisation_mass():
    for spec, mass in e_mass_opts.items():
        gk_dict = get_gs2_basic_dict(e_mass=mass)

        gk_input = GKInputGS2()
        gk_input.read_dict(gk_dict)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._get_normalisation()
        elif spec == "deuterium":
            norm_dict = gk_input._get_normalisation()
            assert norm_dict == {}
        else:
            norm_dict = gk_input._get_normalisation()
            assert norm_dict["mref_species"] == spec

def test_non_standard_normalisation_temp():
    for spec, temp in e_temp_opts.items():
        gk_dict = get_gs2_basic_dict(electron_temp=temp)

        gk_input = GKInputGS2()
        gk_input.read_dict(gk_dict)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._get_normalisation()
        elif spec == "electron":
            norm_dict = gk_input._get_normalisation()
            assert norm_dict == {}
        else:
            norm_dict = gk_input._get_normalisation()
            assert norm_dict["tref_species"] == spec


def test_non_standard_normalisation_dens():
    for spec, dens in e_dens_opts.items():
        gk_dict = get_gs2_basic_dict(electron_dens=dens)

        gk_input = GKInputGS2()
        gk_input.read_dict(gk_dict)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._get_normalisation()
        elif spec == "electron":
            norm_dict = gk_input._get_normalisation()
            assert norm_dict == {}
        else:
            norm_dict = gk_input._get_normalisation()
            assert norm_dict["nref_species"] == spec


def test_non_standard_normalisation_length():
    for length, rmaj in rmaj_opts.items():
        gk_dict = get_gs2_basic_dict(Rmaj=rmaj)

        gk_input = GKInputGS2()
        gk_input.read_dict(gk_dict)

        norm_dict = gk_input._get_normalisation()
        if length == "minor_radius":
            assert norm_dict == {}
        else:
            assert norm_dict["lref"] == length


def test_non_standard_normalisation_b():
    for b_field, ratio in rgeo_rmaj_opts.items():
        gk_dict = get_gs2_basic_dict(Rgeo_Rmaj=ratio)

        gk_input = GKInputGS2()
        gk_input.read_dict(gk_dict)

        norm_dict = gk_input._get_normalisation()
        if b_field == "B0":
            assert norm_dict == {}
        else:
            assert norm_dict["bref"] == b_field