from pyrokinetics.kinetics import Kinetics
from pyrokinetics.constants import electron_mass, deuterium_mass, hydrogen_mass

import numpy as np
import pathlib

TEMPLATE_DIR = pathlib.Path(__file__).parent / "../templates"

tritium_mass = 1.5 * deuterium_mass


def check_species(species, name, charge, mass, midpoint_density=0.1):
    assert species.species_type == name
    assert species.charge == charge
    assert species.mass == mass
    assert np.isclose(species.get_dens(0.5), midpoint_density)


def test_read_scene():
    scene = Kinetics(TEMPLATE_DIR / "scene.cdf", "SCENE")

    assert scene.nspec == 3
    assert sorted(scene.species_names) == sorted(["electron", "deuterium", "tritium"])
    check_species(
        scene.species_data["electron"], "electron", -1, electron_mass, 1.5116787e20
    )
    check_species(
        scene.species_data["deuterium"], "deuterium", 1, deuterium_mass, 7.55839348e19
    )
    check_species(
        scene.species_data["tritium"], "tritium", 1, tritium_mass, 7.55839348e19
    )


def test_read_jetto():
    jetto = Kinetics(TEMPLATE_DIR / "jetto.cdf", "JETTO")

    assert jetto.nspec == 5
    assert sorted(jetto.species_names) == sorted(
        ["electron", "deuterium", "tritium", "impurity", "helium"]
    )
    check_species(
        jetto.species_data["electron"], "electron", -1, electron_mass, 2.07828239e20
    )
    check_species(
        jetto.species_data["deuterium"], "deuterium", 1, deuterium_mass, 1.05502296e20
    )
    check_species(
        jetto.species_data["tritium"], "tritium", 1, tritium_mass, 9.91889184e19
    )
    check_species(
        jetto.species_data["impurity"],
        "impurity",
        54,
        132 * hydrogen_mass,
        5.80931534e16,
    )
    check_species(
        jetto.species_data["helium"], "helium", 2, 2 * deuterium_mass, 7.91436608e16
    )


def test_read_transp():
    transp = Kinetics(TEMPLATE_DIR / "transp.cdf", "TRANSP")

    assert transp.nspec == 4
    assert sorted(transp.species_names) == sorted(
        ["electron", "deuterium", "tritium", "impurity"]
    )
    check_species(
        transp.species_data["electron"], "electron", -1, electron_mass, 1.54655987e20
    )
    check_species(
        transp.species_data["deuterium"], "deuterium", 1, deuterium_mass, 6.97834294e19
    )
    check_species(
        transp.species_data["tritium"], "tritium", 1, tritium_mass, 6.39787852e19
    )
    check_species(
        transp.species_data["impurity"],
        "impurity",
        6,
        12 * hydrogen_mass,
        3.09311873e18,
    )
