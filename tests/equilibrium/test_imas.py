import warnings

import pytest
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
)
from pyrokinetics.equilibrium.imas import EquilibriumReaderIMAS


@pytest.fixture
def example_file():
    return template_dir / "equilibrium.h5"


def test_read(example_file):
    """
    Ensure it can read the example IMAS file, and that it produces an Equilibrium
    """
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    result = EquilibriumReaderIMAS()(example_file)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    assert isinstance(result, Equilibrium)


def test_verify_file_type(example_file):
    """Ensure verify_file_type completes without throwing an error"""
    EquilibriumReaderIMAS().verify_file_type(example_file)


def test_read_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderIMAS()(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp.cdf"])
def test_read_file_is_not_imas(filename):
    """Ensure failure when given a non-imas file"""
    filename = template_dir / filename
    with pytest.raises((ValueError, OSError)):
        EquilibriumReaderIMAS()(filename)


def test_verify_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderIMAS().verify_file_type(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp.cdf"])
def test_verify_file_is_not_imas(filename):
    """Ensure failure when given a non-imas file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderIMAS().verify_file_type(filename)
