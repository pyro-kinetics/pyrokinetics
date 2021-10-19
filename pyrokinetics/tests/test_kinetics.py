from pyrokinetics.kinetics import Kinetics
from pyrokinetics.constants import electron_mass, deuterium_mass, hydrogen_mass

import numpy as np
import pathlib

TEMPLATE_DIR = pathlib.Path(__file__).parent / "../templates"

tritium_mass = 1.5 * deuterium_mass


def check_species(
    species,
    name,
    charge,
    mass,
    midpoint_density,
    midpoint_density_gradient,
    midpoint_temperature,
    midpoint_temperature_gradient,
    midpoint_velocity,
    midpoint_velocity_gradient,
):
    assert species.species_type == name
    assert species.charge == charge
    assert species.mass == mass

    assert np.isclose(species.get_dens(0.5), midpoint_density)
    assert np.isclose(species.get_norm_dens_gradient(0.5), midpoint_density_gradient)
    assert np.isclose(species.get_temp(0.5), midpoint_temperature)
    assert np.isclose(
        species.get_norm_temp_gradient(0.5), midpoint_temperature_gradient
    )
    assert np.isclose(species.get_velocity(0.5), midpoint_velocity)
    assert np.isclose(species.get_norm_vel_gradient(0.5), midpoint_velocity_gradient)


def test_read_scene():
    scene = Kinetics(TEMPLATE_DIR / "scene.cdf", "SCENE")

    assert scene.nspec == 3
    assert sorted(scene.species_names) == sorted(["electron", "deuterium", "tritium"])
    check_species(
        scene.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.5116787e20,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        scene.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=7.558393477464637e19,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        scene.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=7.558393477464637e19,
        midpoint_density_gradient=0.4247526509961558,
        midpoint_temperature=12174.554122236143,
        midpoint_temperature_gradient=2.782385669107711,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )


def test_read_jetto():
    jetto = Kinetics(TEMPLATE_DIR / "jetto.cdf", "JETTO")

    assert jetto.nspec == 5
    assert sorted(jetto.species_names) == sorted(
        ["electron", "deuterium", "tritium", "impurity", "helium"]
    )
    check_species(
        jetto.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=2.078282391282811e20,
        midpoint_density_gradient=0.7407566857690338,
        midpoint_temperature=7520.436894799198,
        midpoint_temperature_gradient=2.4903881194905755,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=1.0550229617783579e20,
        midpoint_density_gradient=0.7673913133177284,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.763031794197728,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=9.918891843529097e19,
        midpoint_density_gradient=0.7124191676802976,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.763031794197728,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["impurity"],
        "impurity",
        54,
        132 * hydrogen_mass,
        midpoint_density=5.809315337899827e16,
        midpoint_density_gradient=0.7407456472981879,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.763031794197728,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["helium"],
        "helium",
        2,
        2 * deuterium_mass,
        midpoint_density=7.914366079876562e16,
        midpoint_density_gradient=10.985863144964545,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.763031794197728,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )


def test_read_transp():
    transp = Kinetics(TEMPLATE_DIR / "transp.cdf", "TRANSP")

    assert transp.nspec == 4
    assert sorted(transp.species_names) == sorted(
        ["electron", "deuterium", "tritium", "impurity"]
    )
    check_species(
        transp.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.5465598699442097e20,
        midpoint_density_gradient=0.2045522220475293,
        midpoint_temperature=12469.654886858232,
        midpoint_temperature_gradient=2.515253525050096,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=6.9783429362652094e19,
        midpoint_density_gradient=0.13986183752938153,
        midpoint_temperature=12469.654886858232,
        midpoint_temperature_gradient=2.515253525050096,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=6.3978785182926684e19,
        midpoint_density_gradient=0.4600323954647866,
        midpoint_temperature=12469.654886858232,
        midpoint_temperature_gradient=2.515253525050096,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        midpoint_density=3.0931187279423063e18,
        midpoint_density_gradient=0.20453530330985722,
        midpoint_temperature=12469.654886858232,
        midpoint_temperature_gradient=2.515253525050096,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
