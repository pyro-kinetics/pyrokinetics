from pyrokinetics.gk_code import GKInputGKW
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

from pathlib import Path
import numpy as np
import pytest

from examples import example_JETTO  # noqa

template_file = template_dir / "input.gkw"


@pytest.fixture
def default_gkw():
    return GKInputGKW()


@pytest.fixture
def gkw():
    return GKInputGKW(template_file)


def test_read(gkw):
    """Ensure a gkw file can be read, and that the 'data' attribute is set"""
    params = ["gridsize", "mode", "geom"]
    assert np.all(np.isin(params, list(gkw.data)))


def test_read_str():
    """Ensure a gkw file can be read as a string, and that the 'data' attribute is set"""
    params = ["gridsize", "mode", "geom"]
    with open(template_file, "r") as f:
        gkw = GKInputGKW.from_str(f.read())
        assert np.all(np.isin(params, list(gkw.data)))


def test_verify(gkw):
    """Ensure that 'verify' does not raise exception on GKW file"""
    gkw.verify(template_file)


@pytest.mark.parametrize(
    "filename", ["input.gs2", "input.cgyro", "transp.cdf", "helloworld", "input.gene"]
)
def test_verify_bad_inputs(gkw, filename):
    """Ensure that 'verify' raises exception on non-GKW file"""
    with pytest.raises(Exception):
        gkw.verify(template_dir.joinpath(filename))


def test_is_nonlinear(gkw):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    gkw.data["control"]["non_linear"] = 0
    assert gkw.is_linear()
    assert not gkw.is_nonlinear()
    gkw.data["control"]["non_linear"] = 1
    gkw.data["mode"]["mode_box"] = True
    assert not gkw.is_linear()
    assert gkw.is_nonlinear()


def test_add_flags(gkw):
    gkw.add_flags({"foo": {"bar": "baz"}})
    assert gkw.data["foo"]["bar"] == "baz"


def test_get_local_geometry(gkw):
    # TODO test it has the correct values
    local_geometry = gkw.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(gkw):
    local_species = gkw.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    assert len(gkw.data["species"]) == 2
    # Ensure you can index gkw.data["species"] (doesn't work on some f90nml versions)
    assert gkw.data["species"][0]
    assert gkw.data["species"][1]
    assert local_species["electron"]
    assert local_species["ion1"]
    # TODO test it has the correct values


def test_get_numerics(gkw):
    # TODO test it has the correct values
    numerics = gkw.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gkw):
    """Ensure a gkw file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gkw.get_local_geometry()
    local_species = gkw.get_local_species()
    numerics = gkw.get_numerics()

    # Set output path
    filename = tmp_path / "input.in"

    # Write out a new input file
    gkw_writer = GKInputGKW()
    gkw_writer.set(local_geometry, local_species, numerics)

    # Ensure you can index gkw.data["species"] (doesn't work on some f90nml versions)
    assert len(gkw_writer.data["species"]) == 2
    assert gkw_writer.data["species"][0]
    assert gkw_writer.data["species"][1]

    # Write to disk
    gkw_writer.write(filename)

    # Ensure a new file exists
    assert Path(filename).exists()

    # Ensure it is a valid file
    GKInputGKW().verify(filename)
    gkw_reader = GKInputGKW(filename)
    new_local_geometry = gkw_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gkw_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gkw_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_species_order(tmp_path):
    pyro = example_JETTO.main(tmp_path)

    # Reverse species order so electron is last
    pyro.local_species.names = pyro.local_species.names[::-1]
    pyro.gk_code = "GKW"

    pyro.write_gk_file(file_name=tmp_path / "input.in")

    assert Path(tmp_path / "input.in").exists()
