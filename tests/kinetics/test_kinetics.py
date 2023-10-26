from pyrokinetics.kinetics import read_kinetics
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
    return template_dir.joinpath("jetto.jsp")


@pytest.fixture
def transp_file():
    return template_dir.joinpath("transp.cdf")


@pytest.fixture
def pfile_file():
    return template_dir.joinpath("pfile.txt")


@pytest.fixture
def geqdsk_file():
    return template_dir.joinpath("test.geqdsk")


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
    assert species.mass == mass

    assert np.isclose(species.get_charge(0.5).m, charge)
    assert np.isclose(species.get_dens(0.5).m, midpoint_density)
    assert np.isclose(species.get_norm_dens_gradient(0.5).m, midpoint_density_gradient)
    assert np.isclose(species.get_temp(0.5).m, midpoint_temperature)
    assert np.isclose(
        species.get_norm_temp_gradient(0.5).m, midpoint_temperature_gradient
    )
    assert np.isclose(species.get_velocity(0.5).m, midpoint_velocity)
    assert np.isclose(species.get_norm_vel_gradient(0.5).m, midpoint_velocity_gradient)


@pytest.mark.parametrize("kinetics_type", ["SCENE", None])
def test_read_scene(scene_file, kinetics_type):
    scene = read_kinetics(scene_file, kinetics_type)
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
    jetto = read_kinetics(jetto_file, kinetics_type)
    assert jetto.kinetics_type == "JETTO"

    assert jetto.nspec == 3
    assert np.array_equal(
        sorted(jetto.species_names),
        sorted(["electron", "deuterium", "impurity1"]),
    )
    check_species(
        jetto.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=3.98442302e19,
        midpoint_density_gradient=0.24934713314306212,
        midpoint_temperature=2048.70870657,
        midpoint_temperature_gradient=1.877960703115299,
        midpoint_velocity=75600.47570394,
        midpoint_velocity_gradient=1.3620162177136412,
    )
    check_species(
        jetto.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=3.36404139e19,
        midpoint_density_gradient=0.15912588033334082,
        midpoint_temperature=1881.28998733,
        midpoint_temperature_gradient=1.2290413714311896,
        midpoint_velocity=75600.47570394,
        midpoint_velocity_gradient=1.3620162177136412,
    )
    check_species(
        jetto.species_data["impurity1"],
        "impurity1",
        5.99947137,
        12 * hydrogen_mass,
        midpoint_density=8.99100966e17,
        midpoint_density_gradient=0.37761249,
        midpoint_temperature=1881.28998733,
        midpoint_temperature_gradient=1.2290413714311896,
        midpoint_velocity=75600.47570394,
        midpoint_velocity_gradient=1.3620162177136412,
    )


@pytest.mark.parametrize("kinetics_type", ["TRANSP", None])
def test_read_transp(transp_file, kinetics_type):
    transp = read_kinetics(transp_file, kinetics_type)
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
    transp = read_kinetics(transp_file, kinetics_type, time_index=10)
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


@pytest.mark.parametrize("kinetics_type", ["pFile", None])
def test_read_pFile(pfile_file, geqdsk_file, kinetics_type):
    pfile = read_kinetics(pfile_file, kinetics_type, eq_file=geqdsk_file)
    assert pfile.kinetics_type == "pFile"

    assert pfile.nspec == 4
    assert np.array_equal(
        sorted(pfile.species_names),
        sorted(["deuterium", "deuterium_fast", "electron", "impurity"]),
    )
    check_species(
        pfile.species_data["electron"],
        "electron",
        -1,
        electron_mass,
        midpoint_density=7.63899297e19,
        midpoint_density_gradient=1.10742399,
        midpoint_temperature=770.37876268,
        midpoint_temperature_gradient=3.1457586490506135,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        pfile.species_data["deuterium"],
        "deuterium",
        1,
        deuterium_mass,
        midpoint_density=5.99025662e19,
        midpoint_density_gradient=1.7807398428788435,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        pfile.species_data["impurity"],
        "impurity",
        6,
        6 * deuterium_mass,
        midpoint_density=2.74789247e18,
        midpoint_density_gradient=-1.3392585682314078,
        midpoint_temperature=742.54533496,
        midpoint_temperature_gradient=2.410566291534264,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )
    check_species(
        pfile.species_data["deuterium_fast"],
        "deuterium_fast",
        1,
        deuterium_mass,
        midpoint_density=7.63899297e18,
        midpoint_density_gradient=1.1074239891222437,
        midpoint_temperature=1379.36939199,
        midpoint_temperature_gradient=3.0580150015690317,
        midpoint_velocity=0.0,
        midpoint_velocity_gradient=0.0,
    )


@pytest.mark.parametrize(
    "filename,kinetics_type",
    [
        ("scene.cdf", "SCENE"),
        ("jetto.jsp", "JETTO"),
        ("transp.cdf", "TRANSP"),
    ],
)
def test_filetype_inference(filename, kinetics_type):
    kinetics = read_kinetics(template_dir.joinpath(filename))
    assert kinetics.kinetics_type == kinetics_type


def test_filetype_inference_pfile(pfile_file, geqdsk_file):
    kinetics = read_kinetics(pfile_file, eq_file=geqdsk_file)
    assert kinetics.kinetics_type == "pFile"


def test_bad_kinetics_type(scene_file):
    kinetics = read_kinetics(scene_file)
    with pytest.raises(ValueError):
        kinetics.kinetics_type = "helloworld"
