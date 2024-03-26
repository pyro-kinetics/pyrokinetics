from pyrokinetics.gk_code import GKInputGS2
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

from pathlib import Path
import numpy as np
import pytest
import f90nml

from typing import Dict, Optional

template_file = template_dir / "input.gs2"


@pytest.fixture
def default_gs2():
    return GKInputGS2()


@pytest.fixture
def gs2():
    return GKInputGS2(template_file)


def modified_gs2_input(replacements: Dict[str, Optional[Dict[str, Optional[str]]]]):
    """Return a GS2 input file based on the template file, but with
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

    return GKInputGS2.from_str(str(input_file))


def test_read(gs2):
    """Ensure a gs2 file can be read, and that the 'data' attribute is set"""
    params = ["theta_grid_parameters", "theta_grid_eik_knobs", "kt_grids_knobs"]
    assert np.all(np.isin(params, list(gs2.data)))


def test_read_str():
    """Ensure a gs2 file can be read as a string, and that the 'data' attribute is set"""
    params = ["theta_grid_parameters", "theta_grid_eik_knobs", "kt_grids_knobs"]
    with open(template_file, "r") as f:
        gs2 = GKInputGS2.from_str(f.read())
        assert np.all(np.isin(params, list(gs2.data)))


def test_verify_file_type(gs2):
    """Ensure that 'verify_file_type' does not raise exception on GS2 file"""
    gs2.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_file_type_bad_inputs(gs2, filename):
    """Ensure that 'verify' raises exception on non-GS2 file"""
    with pytest.raises(Exception):
        gs2.verify_file_type(template_dir / filename)


def test_is_nonlinear(gs2):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    assert gs2.is_linear()
    assert not gs2.is_nonlinear()
    gs2.data["kt_grids_knobs"]["grid_option"] = "box"
    assert gs2.is_linear()
    assert not gs2.is_nonlinear()
    gs2.data["nonlinear_terms_knobs"] = {"nonlinear_mode": "on"}
    assert not gs2.is_linear()
    assert gs2.is_nonlinear()


def test_add_flags(gs2):
    gs2.add_flags({"foo": {"bar": "baz"}})
    assert gs2.data["foo"]["bar"] == "baz"


def test_get_local_geometry(gs2):
    # TODO test it has the correct values
    local_geometry = gs2.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(gs2):
    local_species = gs2.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    # TODO test it has the correct values
    assert local_species["electron"]
    assert local_species["ion1"]


def test_get_numerics(gs2):
    # TODO test it has the correct values
    numerics = gs2.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gs2):
    """Ensure a gs2 file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gs2.get_local_geometry()
    local_species = gs2.get_local_species()
    numerics = gs2.get_numerics()
    # Set output path
    filename = tmp_path / "input.in"
    # Write out a new input file
    gs2_writer = GKInputGS2()
    gs2_writer.set(local_geometry, local_species, numerics)
    gs2_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputGS2().verify_file_type(filename)
    gs2_reader = GKInputGS2(filename)
    new_local_geometry = gs2_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gs2_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gs2_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_gs2_linear_box(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8, "jtwist": 8},
    }
    gs2 = modified_gs2_input(replacements)

    numerics = gs2.get_numerics()
    shat = gs2.get_local_geometry().shat
    jtwist = 8
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky, np.sqrt(2) / 4)
    assert np.isclose(numerics.kx, 2 * np.pi * numerics.ky * shat / jtwist)


def test_gs2_linear_box_no_jtwist(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8},
    }
    gs2 = modified_gs2_input(replacements)

    numerics = gs2.get_numerics()
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky, np.sqrt(2) / 4)
    shat = gs2.get_local_geometry().shat
    jtwist = 2 * np.pi * shat
    expected_kx = numerics.ky * jtwist / int(jtwist)
    assert np.isclose(numerics.kx, expected_kx)


def test_gs2_linear_range(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "range"},
        "kt_grids_range_parameters": {"naky": 12, "aky_min": 2, "aky_max": 8},
    }
    gs2 = modified_gs2_input(replacements)

    numerics = gs2.get_numerics()
    assert numerics.nkx == 1
    assert numerics.nky == 12
    expected_ky = np.linspace(2, 8, 12) / np.sqrt(2)
    assert np.allclose(numerics.ky, expected_ky)


def test_gs2_linear_missing_ngauss(tmp_path):
    replacements = {"le_grids_knobs": {"ngauss": None}}
    gs2 = modified_gs2_input(replacements)
    numerics = gs2.get_numerics()

    assert numerics.npitch == 10


def test_gs2_linear_nesubsuper(tmp_path):
    replacements = {"le_grids_knobs": {"nesub": 11, "nesuper": 15, "negrid": None}}
    gs2 = modified_gs2_input(replacements)
    numerics = gs2.get_numerics()

    assert numerics.nenergy == 26


def test_gs2_invalid_nonlinear(tmp_path):
    replacements = {
        "nonlinear_terms_knobs": {"nonlinear_mode": "on"},
        "kt_grids_knobs": {"grid_option": "box"},
        "knobs": {"wstar_units": True},
    }
    with pytest.raises(RuntimeError):
        modified_gs2_input(replacements)


def test_gs2_valid_nonlinear_with_wstart_units(tmp_path):
    replacements = {
        "nonlinear_terms_knobs": {"nonlinear_mode": "on"},
        "kt_grids_knobs": {"grid_option": "box"},
        "knobs": {"wstar_units": False},
    }

    gs2 = modified_gs2_input(replacements)
    assert gs2.is_nonlinear()
