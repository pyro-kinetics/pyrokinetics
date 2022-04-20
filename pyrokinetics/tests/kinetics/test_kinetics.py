from pyrokinetics.kinetics import Kinetics
from pyrokinetics.constants import electron_mass, deuterium_mass, hydrogen_mass
from pyrokinetics import template_dir

import pytest
import numpy as np

tritium_mass = 1.5 * deuterium_mass


@pytest.fixture
def scene_file():
    return template_dir.joinpath("scene.cdf")


@pytest.fixture
def jetto_file():
    return template_dir.joinpath("jetto.cdf")


@pytest.fixture
def transp_file():
    return template_dir.joinpath("transp.cdf")


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


@pytest.mark.parametrize("kinetics_type", ["SCENE", None])
def test_read_scene(scene_file, kinetics_type):
    scene = Kinetics(scene_file, kinetics_type)
    assert scene.kinetics_type == "SCENE"

    assert scene.nspec == 3
    assert np.array_equal(
        sorted(scene.species_names), sorted(["electron", "deuterium", "tritium"])
    )
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


@pytest.mark.parametrize("kinetics_type", ["JETTO", None])
def test_read_jetto(jetto_file, kinetics_type):
    jetto = Kinetics(jetto_file, kinetics_type)
    assert jetto.kinetics_type == "JETTO"

    assert jetto.nspec == 5
    assert np.array_equal(
        sorted(jetto.species_names),
        sorted(["electron", "deuterium", "tritium", "impurity1", "helium"]),
    )
    check_species(
        jetto.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=2.0855866269392273e20,
        midpoint_density_gradient=0.6521205257186596,
        midpoint_temperature=7520.436894799198,
        midpoint_temperature_gradient=2.1823902554424985,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=1.0550229617783579e20,
        midpoint_density_gradient=0.672484465850412,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.421314820747057,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=9.918891843529097e19,
        midpoint_density_gradient=0.6243109807534643,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.421314820747057,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["impurity1"],
        "impurity1",
        54,
        132 * hydrogen_mass,
        midpoint_density=5.809315337899827e16,
        midpoint_density_gradient=0.6491341930894274,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.421314820747057,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        jetto.species_data["helium"],
        "helium",
        2,
        2 * deuterium_mass,
        midpoint_density=7.914366079876562e16,
        midpoint_density_gradient=9.627190431706618,
        midpoint_temperature=7155.071744869885,
        midpoint_temperature_gradient=2.421314820747057,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )


@pytest.mark.parametrize("kinetics_type", ["TRANSP", None])
def test_read_transp(transp_file, kinetics_type):
    transp = Kinetics(transp_file, kinetics_type)
    assert transp.kinetics_type == "TRANSP"

    assert transp.nspec == 4
    assert np.array_equal(
        sorted(transp.species_names),
        sorted(["electron", "deuterium", "tritium", "impurity"]),
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


@pytest.mark.parametrize("kinetics_type", ["TRANSP", None])
def test_read_transp_kwargs(transp_file, kinetics_type):
    transp = Kinetics(transp_file, kinetics_type, time_index=10)
    assert transp.kinetics_type == "TRANSP"

    assert transp.nspec == 4
    assert np.array_equal(
        sorted(transp.species_names),
        sorted(["electron", "deuterium", "tritium", "impurity"]),
    )
    check_species(
        transp.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=1.54666187e20,
        midpoint_density_gradient=0.20538268693802364,
        midpoint_temperature=12479.79840937,
        midpoint_temperature_gradient=2.5225424443317688,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=6.97865847e19,
        midpoint_density_gradient=0.14042679198682875,
        midpoint_temperature=12479.798409368073,
        midpoint_temperature_gradient=2.5225424443317688,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["tritium"],
        "tritium",
        1,
        tritium_mass,
        midpoint_density=6.544184870368806e19,
        midpoint_density_gradient=0.3731053213184641,
        midpoint_temperature=12479.798409368073,
        midpoint_temperature_gradient=2.5225424443317688,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        transp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        midpoint_density=3.0933239195812495e18,
        midpoint_density_gradient=0.20536537726005905,
        midpoint_temperature=12479.798409368073,
        midpoint_temperature_gradient=2.5225424443317688,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )


@pytest.mark.parametrize(
    "filename,kinetics_type",
    [
        ("scene.cdf", "SCENE"),
        ("jetto.cdf", "JETTO"),
        ("transp.cdf", "TRANSP"),
    ],
)
def test_filetype_inference(filename, kinetics_type):
    kinetics = Kinetics(template_dir.joinpath(filename))
    assert kinetics.kinetics_type == kinetics_type


def test_bad_kinetics_type(scene_file):
    kinetics = Kinetics(scene_file)
    with pytest.raises(ValueError):
        kinetics.kinetics_type = "helloworld"
