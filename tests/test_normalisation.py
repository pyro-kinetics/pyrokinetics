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
from pyrokinetics.gk_code import (
    GKInputGS2,
    GKInputCGYRO,
    GKInputGENE,
    GKInputTGLF,
    GKInputGKW,
    GKInputSTELLA,
)

import numpy as np

import pytest


@pytest.fixture(scope="module")
def kinetics():
    # Currently no nice way to construct a Kinetics object _not_ from file
    return read_kinetics(pk.template_dir / "jetto.jsp", "JETTO")


@pytest.fixture(scope="module")
def geometry():
    return LocalGeometry(
        {
            "a_minor": 2.3 * ureg.meter,
            "B0": 1.2 * ureg.tesla,
            "bunit_over_b0": 2 * ureg.dimensionless,
            "Rmaj": 4.6 * ureg.meter,
        }
    )


@pytest.fixture(scope="module")
def geometry_sim_units():
    return LocalGeometry(
        {
            "a_minor": 1.0 * ureg.lref_minor_radius,
            "B0": 1.0 * ureg.bref_B0,
            "bunit_over_b0": 1 * ureg.dimensionless,
            "Rmaj": 3.0 * ureg.lref_minor_radius,
        }
    )


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
    assert q.to(norm.gene.lref) == norm.gene.lref / 2.0

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


def test_set_all_references_overwrite():
    pyro = pk.Pyro(gk_file=gk_gs2_template)
    norm = SimulationNormalisation("test")

    reference_values = {
        "tref_electron": 1000.0 * norm.units.eV,
        "nref_electron": 1e19 * norm.units.meter**-3,
        "lref_minor_radius": 1.5 * norm.units.meter,
        "bref_B0": 2.0 * norm.units.tesla,
    }

    norm.set_all_references(pyro, **reference_values)

    # Change all references and set again
    reference_values = {k: 2.0 * v for k, v in reference_values.items()}
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

    assert (1 * ureg.bref_Bunit).to(norm.cgyro) == 1 * norm.cgyro.bref
    assert (1 * ureg.bref_B0).to(norm) == 1 * norm.bref
    assert (1 * ureg.bref_Bunit).to(norm.cgyro) == 1 * norm.cgyro.bref


def test_convert_lref_simulation_to_physical(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    assert (1 * ureg.lref_minor_radius).to(norm) == 1 * norm.lref
    assert (1 * ureg.lref_major_radius).to(norm.gene) == 1 * norm.gene.lref
    assert (1 * ureg.lref_minor_radius).to(norm.gene) == norm.gene.lref / 2.0
    assert (1 * ureg.lref_major_radius).to(norm) == 2.0 * norm.lref


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
    expected_gene = norm.gene.lref / 2.0
    assert np.isclose(length_gene, expected_gene)


def test_convert_mixed_units_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    frequency_gs2 = 1 * norm.gs2.vref / norm.gs2.lref

    frequency = frequency_gs2.to(norm)
    expected = np.sqrt(2) * norm.vref / norm.lref

    frequency_gene = frequency.to(norm.gene)
    expected_gene = 2.0 * np.sqrt(2) * norm.gene.vref / norm.gene.lref

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
    length_gene_expected = norm.gene.lref / 2.0
    assert length_gene == length_gene_expected


def test_convert_mixed_simulation_units_to_normalisation(geometry, kinetics):
    norm = SimulationNormalisation(
        "test", geometry=geometry, kinetics=kinetics, psi_n=0.5
    )

    frequency_gs2 = 1 * norm.units.vref_most_probable / norm.units.lref_minor_radius

    frequency = frequency_gs2.to(norm)
    expected = np.sqrt(2) * norm.vref / norm.lref

    frequency_gene = frequency.to(norm.gene)
    expected_gene = 2.0 * np.sqrt(2) * norm.gene.vref / norm.gene.lref

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


def get_basic_gk_input(
    e_mass=0.0002724437107,
    electron_temp=1.0,
    electron_dens=1.0,
    Rmaj=3.0,
    Rgeo_Rmaj=1.0,
    code=None,
):

    d_mass = (deuterium_mass / electron_mass).m * e_mass
    c_mass = 12 * d_mass

    if code == "GS2":
        dict = {
            "species_knobs": {"nspec": 3},
            "species_parameters_1": {
                "type": "electron",
                "z": -1,
                "mass": e_mass,
                "temp": electron_temp,
                "dens": electron_dens,
            },
            "species_parameters_2": {
                "type": "ion",
                "z": 1,
                "mass": d_mass,
                "temp": 2 * electron_temp,
                "dens": electron_dens * 5.0 / 6.0,
            },
            "species_parameters_3": {
                "type": "ion",
                "z": 6,
                "mass": c_mass,
                "temp": 2 * electron_temp,
                "dens": electron_dens * 1.0 / 6.0,
            },
            "theta_grid_parameters": {"rmaj": Rmaj, "r_geo": Rgeo_Rmaj * Rmaj},
            "theta_grid_eik_knobs": {"irho": 2},
        }
        gk_input = GKInputGS2()

    elif code == "STELLA":
        dict = {
            "species_knobs": {"nspec": 3},
            "species_parameters_1": {
                "type": "electron",
                "z": -1,
                "mass": e_mass,
                "temp": electron_temp,
                "dens": electron_dens,
            },
            "species_parameters_2": {
                "type": "ion",
                "z": 1,
                "mass": d_mass,
                "temp": 2 * electron_temp,
                "dens": electron_dens * 5.0 / 6.0,
            },
            "species_parameters_3": {
                "type": "ion",
                "z": 6,
                "mass": c_mass,
                "temp": 2 * electron_temp,
                "dens": electron_dens * 1.0 / 6.0,
            },
            "millergeo_parameters": {"rmaj": Rmaj, "rgeo": Rgeo_Rmaj * Rmaj},
        }
        gk_input = GKInputSTELLA()

    elif code == "GENE":
        dict = {
            "box": {"n_spec": 3},
            "species": [
                {
                    "charge": -1,
                    "mass": e_mass,
                    "temp": electron_temp,
                    "dens": electron_dens,
                },
                {
                    "charge": 1,
                    "mass": d_mass,
                    "temp": 2 * electron_temp,
                    "dens": electron_dens * 5.0 / 6.0,
                },
                {
                    "charge": 6,
                    "mass": c_mass,
                    "temp": 2 * electron_temp,
                    "dens": electron_dens * 1.0 / 6.0,
                },
            ],
            "geometry": {"major_r": Rmaj, "minor_r": Rmaj / 3.0},
        }
        gk_input = GKInputGENE()

    elif code == "CGYRO":
        dict = {
            "N_SPECIES": 3,
            "Z_1": -1,
            "MASS_1": e_mass,
            "TEMP_1": electron_temp,
            "DENS_1": electron_dens,
            "Z_2": 1,
            "MASS_2": d_mass,
            "TEMP_2": 2 * electron_temp,
            "DENS_2": electron_dens * 5.0 / 6.0,
            "Z_3": 6,
            "MASS_3": c_mass,
            "TEMP_3": 2 * electron_temp,
            "DENS_3": electron_dens * 1.0 / 6.0,
            "RMAJ": Rmaj,
            "RMIN": Rmaj / 3.0,
        }
        gk_input = GKInputCGYRO()

    elif code == "TGLF":
        dict = {
            "ns": 3,
            "zs_1": -1,
            "mass_1": e_mass,
            "taus_1": electron_temp,
            "as_1": electron_dens,
            "zs_2": 1,
            "mass_2": d_mass,
            "taus_2": 2 * electron_temp,
            "as_2": electron_dens * 5.0 / 6.0,
            "zs_3": 6,
            "mass_3": c_mass,
            "taus_3": 2 * electron_temp,
            "as_3": electron_dens * 1.0 / 6.0,
            "rmaj_loc": Rmaj,
            "rmin_loc": Rmaj / 3.0,
        }
        gk_input = GKInputTGLF()
    elif code == "GKW":
        dict = {
            "gridsize": {"number_of_species": 3},
            "species": [
                {
                    "z": -1,
                    "mass": e_mass,
                    "temp": electron_temp,
                    "dens": electron_dens,
                },
                {
                    "z": 1,
                    "mass": d_mass,
                    "temp": 2 * electron_temp,
                    "dens": electron_dens * 5.0 / 6.0,
                },
                {
                    "z": 6,
                    "mass": c_mass,
                    "temp": 2 * electron_temp,
                    "dens": electron_dens * 1.0 / 6.0,
                },
            ],
        }
        gk_input = GKInputGKW()
    else:
        raise ValueError(f"Code {code} not yet supported in testing")

    gk_input.read_dict(dict)
    return gk_input


e_mass_opts = {
    "deuterium": 0.0002724437107,
    "hydrogen": 0.0005446170214,
    "tritium": 0.0001819200062,
    "electron": 1.0,
    "failure": 0.5,
}
e_temp_opts = {"electron": 1.0, "deuterium": 0.5, "failure": 2.0}
e_dens_opts = {"electron": 1.0, "deuterium": 6.0 / 5.0, "failure": 0.5}
rmaj_opts = {"major_radius": 1.0, "minor_radius": 3.0}
rgeo_rmaj_opts = {"B0": 1.0, "Bgeo": 1.1}


@pytest.mark.parametrize(
    "gk_code",
    [
        "GS2",
        "GENE",
        "CGYRO",
        "TGLF",
        "GKW",
        "STELLA",
    ],
)
def test_non_standard_normalisation_mass(gk_code, geometry_sim_units):
    for spec, mass in e_mass_opts.items():
        gk_input = get_basic_gk_input(e_mass=mass, code=gk_code)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._detect_normalisation()
        elif spec == "deuterium":
            gk_input._detect_normalisation()
            assert gk_input._convention_dict == {}
        else:
            gk_input._detect_normalisation()
            assert gk_input._convention_dict["mref_species"] == spec

            norm = SimulationNormalisation("nonstandard_temp")
            norm.add_convention_normalisation(
                name="nonstandard", convention_dict=gk_input._convention_dict
            )
            assert np.isclose(
                mass * norm.nonstandard.mref, 1.0 * norm.units.mref_electron
            )
            mass_md = mass / e_mass_opts["deuterium"]

            assert np.isclose(
                mass_md**-0.5 * norm.nonstandard.vref,
                1.0 * getattr(norm, gk_code.lower()).vref,
            )

            norm.set_ref_ratios(local_geometry=geometry_sim_units)
            assert np.isclose(
                mass_md**0.5 * norm.nonstandard.rhoref,
                (1.0 * getattr(norm, gk_code.lower()).rhoref).to(
                    norm.nonstandard.rhoref, norm.context
                ),
            )


@pytest.mark.parametrize(
    "gk_code",
    [
        "GS2",
        "GENE",
        "CGYRO",
        "TGLF",
        "GKW",
        "STELLA",
    ],
)
def test_non_standard_normalisation_temp(gk_code, geometry_sim_units):
    for spec, temp in e_temp_opts.items():
        gk_input = get_basic_gk_input(electron_temp=temp, code=gk_code)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._detect_normalisation()
        elif spec == "electron":
            gk_input._detect_normalisation()
            assert gk_input._convention_dict == {}
        else:
            gk_input._detect_normalisation()
            assert gk_input._convention_dict["tref_species"] == spec

            norm = SimulationNormalisation("nonstandard_temp")
            norm.add_convention_normalisation(
                name="nonstandard", convention_dict=gk_input._convention_dict
            )
            assert np.isclose(
                temp * norm.nonstandard.tref, 1.0 * getattr(norm, gk_code.lower()).tref
            )
            assert np.isclose(
                temp**0.5 * norm.nonstandard.vref,
                1.0 * getattr(norm, gk_code.lower()).vref,
            )

            if gk_code in ["TGLF", "CGYRO"]:
                assert np.isclose(
                    temp * norm.nonstandard.beta_ref,
                    1.0 * getattr(norm, gk_code.lower()).beta_ref,
                )
            else:
                assert np.isclose(
                    temp**-1 * norm.nonstandard.beta_ref,
                    1.0 * getattr(norm, gk_code.lower()).beta_ref,
                )

            norm.set_ref_ratios(local_geometry=geometry_sim_units)
            assert np.isclose(
                temp**0.5 * norm.nonstandard.rhoref,
                (1.0 * getattr(norm, gk_code.lower()).rhoref).to(
                    norm.nonstandard.rhoref, norm.context
                ),
            )


@pytest.mark.parametrize(
    "gk_code",
    [
        "GS2",
        "GENE",
        "CGYRO",
        "TGLF",
        "GKW",
        "STELLA",
    ],
)
def test_non_standard_normalisation_dens(gk_code):
    for spec, dens in e_dens_opts.items():
        gk_input = get_basic_gk_input(electron_dens=dens, code=gk_code)

        if spec == "failure":
            with pytest.raises(ValueError):
                gk_input._detect_normalisation()
        elif spec == "electron":
            gk_input._detect_normalisation()
            assert gk_input._convention_dict == {}
        else:
            gk_input._detect_normalisation()
            assert gk_input._convention_dict["nref_species"] == spec

            norm = SimulationNormalisation("nonstandard_dens")
            norm.add_convention_normalisation(
                name="nonstandard", convention_dict=gk_input._convention_dict
            )
            assert np.isclose(
                dens * norm.nonstandard.nref, 1.0 * getattr(norm, gk_code.lower()).nref
            )


@pytest.mark.parametrize(
    "gk_code",
    [
        "GS2",
        "GENE",
        "CGYRO",
        "TGLF",
        "GKW",
    ],
)
def test_non_standard_normalisation_length(gk_code):
    for length, rmaj in rmaj_opts.items():
        gk_input = get_basic_gk_input(Rmaj=rmaj, code=gk_code)

        gk_input._detect_normalisation()
        if gk_code == "GENE":
            assert gk_input._convention_dict == {}
            if length == "minor_radius":
                assert gk_input.norm_convention == "pyrokinetics"
            else:
                assert gk_input.norm_convention == "gene"
        elif gk_code == "GKW":
            assert gk_input._convention_dict == {}
        else:
            if length == "minor_radius":
                assert gk_input._convention_dict == {}
            else:
                assert gk_input._convention_dict["lref"] == length

                norm = SimulationNormalisation("nonstandard_length")
                norm.add_convention_normalisation(
                    name="nonstandard", convention_dict=gk_input._convention_dict
                )

                assert np.isclose(
                    1.0 * norm.nonstandard.lref,
                    1.0 * norm.gene.lref,
                )


@pytest.mark.parametrize(
    "gk_code",
    [
        "GS2",
        "STELLA",
    ],
)
def test_non_standard_normalisation_b(gk_code, geometry_sim_units):
    for b_field, ratio in rgeo_rmaj_opts.items():
        gk_input = get_basic_gk_input(Rgeo_Rmaj=ratio, code=gk_code)

        gk_input._detect_normalisation()
        if b_field == "B0":
            assert gk_input._convention_dict == {}
        else:
            assert gk_input._convention_dict["bref"] == b_field

            norm = SimulationNormalisation("nonstandard_b")
            norm.add_convention_normalisation(
                name="nonstandard", convention_dict=gk_input._convention_dict
            )
            assert np.isclose(
                ratio * norm.nonstandard.bref,
                1.0 * getattr(norm, gk_code.lower()).bref,
            )

            norm.set_ref_ratios(local_geometry=geometry_sim_units)
            assert np.isclose(
                ratio**-1 * norm.nonstandard.rhoref,
                (1.0 * getattr(norm, gk_code.lower()).rhoref).to(
                    norm.nonstandard.rhoref, norm.context
                ),
            )
