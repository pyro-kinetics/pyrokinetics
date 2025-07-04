from pathlib import Path
from typing import Dict, Optional

import toml
import numpy as np
import pytest
import sys

from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputGX
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

docs_dir = Path(__file__).parent.parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO  # noqa

template_file = template_dir / "input.gx"


@pytest.fixture
def default_gx():
    return GKInputGX()


@pytest.fixture
def gx():
    return GKInputGX(template_file)


def modified_gx_input(replacements: Dict[str, Optional[Dict[str, Optional[str]]]]):
    """Return a GX input file based on the template file, but with
    updated keys/namelists from replacements. Keys that are `None` will be deleted
    """

    with open(template_file, "r") as f:
        input_file = toml.load(f)

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

    new_gx = GKInputGX()
    new_gx.read_dict(input_file)
    return new_gx


def test_read(gx):
    """Ensure a gx file can be read, and that the 'data' attribute is set"""
    params = ["Domain", "Diagnostics", "species"]
    assert np.all(np.isin(params, list(gx.data)))


def test_read_str():
    """Ensure a gx file can be read as a string, and that the 'data' attribute is set"""
    params = ["Domain", "Diagnostics", "species"]
    with open(template_file, "r") as f:
        gx = GKInputGX.from_str(f.read())
        assert np.all(np.isin(params, list(gx.data)))


def test_verify_file_type(gx):
    """Ensure that 'verify_file_type' does not raise exception on GX file"""
    gx.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_file_type_bad_inputs(gx, filename):
    """Ensure that 'verify' raises exception on non-GX file"""
    with pytest.raises(Exception):
        gx.verify_file_type(template_dir / filename)


def test_is_nonlinear(gx):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    assert gx.is_linear()
    assert not gx.is_nonlinear()
    gx.data["Physics"]["nonlinear_mode"] = True
    assert not gx.is_linear()
    assert gx.is_nonlinear()


def test_add_flags(gx):
    gx.add_flags({"foo": {"bar": "baz"}})
    assert gx.data["foo"]["bar"] == "baz"


def test_get_local_geometry(gx):
    # TODO test it has the correct values
    local_geometry = gx.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(gx):
    local_species = gx.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    # TODO test it has the correct values
    assert local_species["electron"]
    assert local_species["ion1"]


def test_get_numerics(gx):
    # TODO test it has the correct values
    numerics = gx.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gx):
    """Ensure a gx file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gx.get_local_geometry()
    local_species = gx.get_local_species()
    numerics = gx.get_numerics()
    # Set output path
    filename = tmp_path / "input.in"
    # Write out a new input file
    gx_writer = GKInputGX()
    gx_writer.set(local_geometry, local_species, numerics)
    gx_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputGX().verify_file_type(filename)
    gx_reader = GKInputGX(filename)
    new_local_geometry = gx_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gx_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gx_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_gx_linear_box(tmp_path):
    replacements = {
        "Dimensions": {"ny": 12, "nx": 8},
        "Domain": {"y0": 2, "x0": 20},
    }
    gx = modified_gx_input(replacements)

    numerics = gx.get_numerics()

    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    assert np.isclose(numerics.kx.m, 1 / 40)


def test_drop_species(tmp_path):
    pyro = example_JETTO.main(tmp_path)
    pyro.gk_code = "GX"

    n_species = pyro.local_species.nspec

    keys = ["mass", "z", "dens", "temp", "vnewk", "tprim", "fprim"]
    for key in keys:
        stored_species = len(pyro.gk_input.data["species"][key])
        assert stored_species == n_species

    pyro.local_species.merge_species(
        base_species="deuterium",
        merge_species=["deuterium", "impurity1"],
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    pyro.update_gk_code()
    n_species = pyro.local_species.nspec
    for key in keys:
        stored_species = len(pyro.gk_input.data["species"][key])
        assert stored_species == n_species
