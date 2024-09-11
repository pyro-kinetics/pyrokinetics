import warnings
import numpy as np

import pytest
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
)
from pyrokinetics.equilibrium.imas import EquilibriumReaderIMAS
from pyrokinetics.normalisation import ureg as units


@pytest.fixture
def example_file():
    return template_dir / "equilibrium.h5"

@pytest.fixture(scope="module")
def imas_equilibrium():
    warnings.simplefilter("ignore", category=EquilibriumCOCOSWarning)
    eq = read_equilibrium(template_dir / "equilibrium.h5")
    warnings.simplefilter("default", category=EquilibriumCOCOSWarning)
    return eq


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


def test_read_imas(imas_equilibrium):
    eq = imas_equilibrium

    assert len(eq["R"]) == 33
    assert len(eq["Z"]) == 33
    assert eq["psi_RZ"].shape[0] == 33
    assert eq["psi_RZ"].shape[1] == 33
    assert len(eq["psi"]) == 33
    assert len(eq["F"]) == 33
    assert len(eq["p"]) == 33
    assert len(eq["q"]) == 33
    assert len(eq["FF_prime"]) == 33
    assert len(eq["p_prime"]) == 33
    assert len(eq["R_major"]) == 33
    assert len(eq["r_minor"]) == 33
    assert len(eq["Z_mid"]) == 33


def test_get_flux_surface(imas_equilibrium):
    fs = imas_equilibrium.flux_surface(0.5)
    assert np.isclose(min(fs["R"].data), 2.3371174692540326 * units.m)
    assert np.isclose(max(fs["R"].data), 3.5241002158466936 * units.m)
    assert np.isclose(min(fs["Z"].data), -0.45710383888731737 * units.m)
    assert np.isclose(max(fs["Z"].data), 1.0969616157779112 * units.m)


def test_get_lcfs(imas_equilibrium):
    lcfs = imas_equilibrium.flux_surface(1.0)
    assert np.isclose(min(lcfs["R"].data), 1.9065626761237917 * units.m)
    assert np.isclose(max(lcfs["R"].data), 3.860072136187683 * units.m)
    assert np.isclose(min(lcfs["Z"].data), -1.3761737225531043 * units.m)
    assert np.isclose(max(lcfs["Z"].data), 1.7911656823750293 * units.m)


def test_B_radial(imas_equilibrium):
    assert np.isclose(imas_equilibrium.B_radial(2.3, 3.1), 0.059839079250066346 * units.tesla)
    assert np.isclose(
        imas_equilibrium.B_radial(3.8, 0.0), -0.0930589646013569 * units.tesla
    )


def test_B_vertical(imas_equilibrium):
    assert np.isclose(
        imas_equilibrium.B_vertical(2.3, 3.1), 0.027785791771171135 * units.tesla
    )
    assert np.isclose(
        imas_equilibrium.B_vertical(3.8, 0.0), -0.4366584225812486 * units.tesla
    )


def test_B_poloidal(imas_equilibrium):
    assert np.isclose(
        imas_equilibrium.B_poloidal(2.3, 3.1), 0.0659754926457287 * units.tesla
    )
    assert np.isclose(
        imas_equilibrium.B_poloidal(3.8, 0.0), 0.4464644990408766 * units.tesla
    )


def test_B_toroidal(imas_equilibrium):
    assert np.isclose(
        imas_equilibrium.B_toroidal(2.3, 3.1), -3.115763998771892 * units.tesla
    )
    assert np.isclose(
        imas_equilibrium.B_toroidal(3.8, 0.0), -1.8854644287949809 * units.tesla
    )
