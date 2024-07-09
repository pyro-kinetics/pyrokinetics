from pathlib import Path
from typing import Dict, Optional

import f90nml
import numpy as np
import pytest
import sys

from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputSTELLA
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

docs_dir = Path(__file__).parent.parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO  # noqa

template_file = template_dir / "input.stella"


@pytest.fixture
def default_stella():
    return GKInputSTELLA()


@pytest.fixture
def stella():
    return GKInputSTELLA(template_file)


def modified_stella_input(replacements: Dict[str, Optional[Dict[str, Optional[str]]]]):
    """Return a STELLA input file based on the template file, but with
    updated keys/namelists from replacements. Keys that are `None` will be deleted
    """

    input_file = f90nml.read(template_file)

    for namelist, keys in replacements.items():
        if namelist not in input_file:
            if keys is None:
                continue
            input_file[namelist] = {}

        if keys is None:
            del input_file[namelist]
            continue

        for key, value in keys.items():
            if value is None:
                try:
                    del input_file[namelist][key]
                except KeyError:
                    continue
            else:
                input_file[namelist][key] = value

    return GKInputSTELLA.from_str(str(input_file))


def test_read(stella):
    """Ensure a stella file can be read, and that the 'data' attribute is set"""
    params = ["millergeo_parameters", "stella_diagnostics_knobs", "kt_grids_knobs"]
    assert np.all(np.isin(params, list(stella.data)))


def test_read_str():
    """Ensure a stella file can be read as a string, and that the 'data' attribute is set"""
    params = ["millergeo_parameters", "stella_diagnostics_knobs", "kt_grids_knobs"]
    with open(template_file, "r") as f:
        stella = GKInputSTELLA.from_str(f.read())
        assert np.all(np.isin(params, list(stella.data)))


def test_verify_file_type(stella):
    """Ensure that 'verify_file_type' does not raise exception on STELLA file"""
    stella.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_file_type_bad_inputs(stella, filename):
    """Ensure that 'verify' raises exception on non-STELLA file"""
    with pytest.raises(Exception):
        stella.verify_file_type(template_dir / filename)


def test_is_nonlinear(stella):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    assert stella.is_linear()
    assert not stella.is_nonlinear()
    stella.data["kt_grids_knobs"]["grid_option"] = "box"
    assert stella.is_linear()
    assert not stella.is_nonlinear()
    stella.data["physics_flags"] = {"nonlinear": True}
    assert not stella.is_linear()
    assert stella.is_nonlinear()


def test_add_flags(stella):
    stella.add_flags({"foo": {"bar": "baz"}})
    assert stella.data["foo"]["bar"] == "baz"


def test_get_local_geometry(stella):
    # TODO test it has the correct values
    local_geometry = stella.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(stella):
    local_species = stella.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    # TODO test it has the correct values
    assert local_species["electron"]
    assert local_species["ion1"]


def test_get_numerics(stella):
    # TODO test it has the correct values
    numerics = stella.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, stella):
    """Ensure a stella file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = stella.get_local_geometry()
    local_species = stella.get_local_species()
    numerics = stella.get_numerics()
    # Set output path
    filename = tmp_path / "input.in"
    # Write out a new input file
    stella_writer = GKInputSTELLA()
    stella_writer.set(local_geometry, local_species, numerics)
    stella_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputSTELLA().verify_file_type(filename)
    stella_reader = GKInputSTELLA(filename)
    new_local_geometry = stella_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = stella_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = stella_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_stella_linear_box(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8, "jtwist": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    shat = stella.get_local_geometry().shat
    jtwist = 8
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    assert np.isclose(numerics.kx, 2 * np.pi * numerics.ky * shat / jtwist)


def test_stella_linear_box_no_jtwist(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    shat = stella.get_local_geometry().shat
    jtwist = 2 * np.pi * shat
    expected_kx = numerics.ky * jtwist / int(jtwist)
    assert np.isclose(numerics.kx, expected_kx)


def test_stella_linear_range(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "range"},
        "kt_grids_range_parameters": {"naky": 12, "aky_min": 2, "aky_max": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    assert numerics.nkx == 1
    assert numerics.nky == 12
    expected_ky = np.linspace(2, 8, 12)
    assert np.allclose(numerics.ky.m, expected_ky)


def test_drop_species(tmp_path):
    pyro = example_JETTO.main(tmp_path)
    pyro.gk_code = "STELLA"

    n_species = pyro.local_species.nspec
    stored_species = len(
        [key for key in pyro.gk_input.data.keys() if "species_parameters_" in key]
    )
    assert stored_species == n_species

    pyro.local_species.merge_species(
        base_species="deuterium",
        merge_species=["deuterium", "impurity1"],
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    pyro.update_gk_code()
    n_species = pyro.local_species.nspec
    stored_species = len(
        [key for key in pyro.gk_input.data.keys() if "species_parameters_" in key]
    )
    assert stored_species == n_species
