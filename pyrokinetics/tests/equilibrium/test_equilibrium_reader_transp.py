from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.equilibrium.transp import TRANSPEquilibriumReader
from pyrokinetics import template_dir
import pytest


@pytest.fixture
def transp_reader():
    return TRANSPEquilibriumReader()


@pytest.fixture
def example_file():
    return template_dir / "transp_eq.cdf"


def test_read(transp_reader, example_file):
    """
    Ensure it can read the example TRANSP file, and that it produces an Equilibrium
    """
    result = transp_reader(example_file)
    assert isinstance(result, Equilibrium)


def test_verify(transp_reader, example_file):
    """Ensure verify completes without throwing an error"""
    transp_reader.verify(example_file)


def test_read_file_does_not_exist(transp_reader):
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        transp_reader(filename)


def test_read_file_is_not_netcdf(transp_reader):
    """Ensure failure when given a non-netcdf file"""
    filename = template_dir / "input.gs2"
    with pytest.raises(Exception):
        transp_reader(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_read_file_is_not_transp(transp_reader, filename):
    """Ensure failure when given a non-transp netcdf file

    This could fail for any number of reasons during processing.
    """
    filename = template_dir / filename
    with pytest.raises(Exception):
        transp_reader(filename)


def test_verify_file_does_not_exist(transp_reader):
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        transp_reader.verify(filename)


def test_verify_file_is_not_netcdf(transp_reader):
    """Ensure failure when given a non-netcdf file"""
    filename = template_dir / "input.gs2"
    with pytest.raises(Exception):
        transp_reader.verify(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_verify_file_is_not_transp(transp_reader, filename):
    """Ensure failure when given a non-transp netcdf file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        transp_reader.verify(filename)
