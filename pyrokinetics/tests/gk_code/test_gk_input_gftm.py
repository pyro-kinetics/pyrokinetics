from pyrokinetics.gk_code import GKInputGFTM
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

from pathlib import Path
import numpy as np
import pytest

template_file = template_dir.joinpath("input.gftm")


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


def test_verify(gftm):
    """Ensure that 'verify' does not raise exception on GFTM file"""
    gftm.verify(template_file)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_bad_inputs(gftm, filename):
    """Ensure that 'verify' raises exception on non-GFTM file"""
    with pytest.raises(Exception):
        gftm.verify(template_dir.joinpath(filename))


def test_is_nonlinear(gftm):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    gftm.data["use_transport_model"] = False
    assert gftm.is_linear()
    assert not gftm.is_nonlinear()
    gftm.data["use_transport_model"] = True
    assert not gftm.is_linear()
    assert gftm.is_nonlinear()


def test_add_flags(gftm):
    gftm.add_flags({"foo": "bar"})
    assert gftm.data["foo"] == "bar"


def test_get_local_geometry(gftm):
    # TODO test it has the correct values
    local_geometry = gftm.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


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
    filename = tmp_path / "input.in"
    # Write out a new input file
    gftm_writer = GKInputGFTM()
    gftm_writer.set(local_geometry, local_species, numerics)
    gftm_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputGFTM().verify(filename)
    gftm_reader = GKInputGFTM(filename)
    new_local_geometry = gftm_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gftm_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gftm_reader.get_numerics()
    assert numerics.ky == new_numerics.ky
