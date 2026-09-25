import sys
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputGFTM, GKInputTGLF, read_gk_input
from pyrokinetics.local_geometry import LocalGeometryMiller, LocalGeometryMXH
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

docs_dir = Path(__file__).parent.parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO  # noqa

template_file = template_dir / "input.gftm"


@pytest.fixture
def default_gftm():
    return GKInputGFTM()


@pytest.fixture
def gftm():
    return GKInputGFTM(template_file)


def test_read(gftm):
    """Ensure a gftm file can be read, and that the 'data' attribute is set"""
    params = ["geometry_flag", "q_loc", "zs_1"]
    assert np.all(np.isin(params, list(gftm.data)))


def test_read_str():
    """Ensure a gftm file can be read as a string, and that the 'data' attribute is set"""
    params = ["geometry_flag", "q_loc", "zs_1"]
    with open(template_file, "r") as f:
        gftm = GKInputGFTM.from_str(f.read())
        assert np.all(np.isin(params, list(gftm.data)))


def test_verify_file_type(gftm):
    """Ensure that 'verify_file_type' does not raise exception on GFTM file"""
    gftm.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename, expected_type",
    [
        ("input.gftm", GKInputGFTM),
        ("input.GfTm", GKInputGFTM),
        ("gftm_scan.input", GKInputGFTM),
        ("input.tglf", GKInputTGLF),
        ("pyroscan_base.input", GKInputTGLF),
    ],
)
def test_autodetect_shared_input(filename, expected_type, tmp_path):
    # A parent directory naming GFTM must not override the input's basename.
    directory = tmp_path / "gftm_runs"
    directory.mkdir()
    path = directory / filename
    path.write_text(template_file.read_text())

    assert isinstance(read_gk_input(path), expected_type)
    other_type = GKInputTGLF if expected_type is GKInputGFTM else GKInputGFTM
    with pytest.raises(ValueError):
        other_type().verify_file_type(path)


def test_explicit_gftm_with_generic_filename(tmp_path):
    path = tmp_path / "input.in"
    path.write_text(template_file.read_text())

    assert isinstance(read_gk_input(path, file_type="GFTM"), GKInputGFTM)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_bad_inputs(gftm, filename):
    """Ensure that 'verify' raises exception on non-GFTM file"""
    with pytest.raises(Exception):
        gftm.verify_file_type(template_dir / filename)


def test_add_flags(gftm):
    gftm.add_flags({"foo": "bar"})
    assert gftm.data["foo"] == "bar"


def test_get_local_geometry(gftm):
    # TODO test it has the correct values
    local_geometry = gftm.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_geometry_mxh_from_miller_family_flag(gftm):
    gftm.data["geometry_flag"] = 1
    gftm.data["shape_cos3"] = 0.05
    local_geometry = gftm.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMXH)


def test_get_local_species(gftm):
    local_species = gftm.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    # TODO test it has the correct values
    assert local_species["electron"]
    assert local_species["ion1"]


def test_get_numerics(gftm):
    # TODO test it has the correct values
    numerics = gftm.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gftm):
    """Ensure a gftm file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gftm.get_local_geometry()
    local_species = gftm.get_local_species()
    numerics = gftm.get_numerics()
    # Set output path
    filename = tmp_path / "input.gftm"
    # Write out a new input file
    gftm_writer = GKInputGFTM()
    gftm_writer.set(local_geometry, local_species, numerics)
    gftm_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputGFTM().verify_file_type(filename)
    gftm_reader = GKInputGFTM(filename)
    new_local_geometry = gftm_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gftm_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gftm_reader.get_numerics()
    assert numerics.ky == new_numerics.ky


def test_drop_species(tmp_path):
    pyro = example_JETTO.main(tmp_path)
    pyro.gk_code = "GFTM"

    n_species = pyro.local_species.nspec
    stored_species = len([key for key in pyro.gk_input.data.keys() if "zs_" in key])
    assert stored_species == n_species

    pyro.local_species.merge_species(
        base_species="deuterium",
        merge_species=["deuterium", "impurity1"],
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    pyro.update_gk_code()
    n_species = pyro.local_species.nspec
    stored_species = len([key for key in pyro.gk_input.data.keys() if "zs_" in key])
    assert stored_species == n_species


def test_set_nxgrid_above_gftm_limit():
    gftm = GKInputGFTM(template_dir / "input.gftm")
    local_geometry = gftm.get_local_geometry()
    local_species = gftm.get_local_species()
    numerics = gftm.get_numerics()
    numerics.ntheta = 64

    gftm.set(local_geometry, local_species, numerics)

    assert gftm.data["nxgrid"] == 64
