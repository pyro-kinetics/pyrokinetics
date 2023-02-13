from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.equilibrium.geqdsk import EquilibriumReaderGEQDSK
from pyrokinetics import template_dir
import pytest


@pytest.fixture
def geqdsk_reader():
    return EquilibriumReaderGEQDSK()


@pytest.fixture
def example_file():
    return template_dir / "transp_eq.geqdsk"


def test_read(geqdsk_reader, example_file):
    """
    Ensure it can read the example GEQDSK file, and that it produces an Equilibrium
    """
    result = geqdsk_reader(example_file)
    assert isinstance(result, Equilibrium)


def test_verify(geqdsk_reader, example_file):
    """Ensure verify completes without throwing an error"""
    geqdsk_reader.verify(example_file)


def test_read_file_does_not_exist(geqdsk_reader):
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        geqdsk_reader(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
def test_read_file_is_not_geqdsk(geqdsk_reader, filename):
    """Ensure failure when given a non-geqdsk file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        geqdsk_reader(filename)


def test_verify_file_does_not_exist(geqdsk_reader):
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        geqdsk_reader.verify(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
def test_verify_file_is_not_geqdsk(geqdsk_reader, filename):
    """Ensure failure when given a non-geqdsk file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        geqdsk_reader.verify(filename)
