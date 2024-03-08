import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
)
from pyrokinetics.equilibrium.gacode import EquilibriumReaderGACODE


@pytest.fixture
def example_file():
    return template_dir / "input.gacode"


def test_read(example_file):
    """
    Ensure it can read the example GACODE file, and that it produces an Equilibrium
    """
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    result = EquilibriumReaderGACODE()(example_file)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    assert isinstance(result, Equilibrium)


def test_verify_file_type(example_file):
    """Ensure verify_file_type completes without throwing an error"""
    EquilibriumReaderGACODE().verify_file_type(example_file)


def test_read_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderGACODE()(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp.cdf"])
def test_read_file_is_not_gacode(filename):
    """Ensure failure when given a non-gacode file"""
    filename = template_dir / filename
    with pytest.raises(ValueError):
        EquilibriumReaderGACODE()(filename)


def test_verify_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderGACODE().verify_file_type(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp.cdf"])
def test_verify_file_is_not_gacode(filename):
    """Ensure failure when given a non-gacode file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderGACODE().verify_file_type(filename)


# Compare GEQDSK and GACODE files of the same Equilibrium
# Compare only the flux surface at ``psi_n=0.5``.
@pytest.fixture(scope="module")
def fs_gacode():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    fs = read_equilibrium(template_dir / "input.gacode").flux_surface(0.5)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return fs


@pytest.fixture(scope="module")
def fs_geqdsk():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    fs = read_equilibrium(template_dir / "transp_eq.geqdsk").flux_surface(0.5)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return fs


@pytest.mark.parametrize(
    "attr",
    [
        "R_major",
        "r_minor",
        "Z_mid",
        "a_minor",
        "rho",
        "F",
        "p",
        "q",
        "magnetic_shear",
        "shafranov_shift",
        "midplane_shift",
        "pressure_gradient",
        "psi_gradient",
    ],
)
def test_compare_gacode_geqdsk_attrs(fs_gacode, fs_geqdsk, attr):
    """
    Compare attributes between equivalent flux surfaces from GACODE CDF and GEQDSK
    files. Assumes FluxSurface has handled units correctly, and only checks that values
    are within 10%.
    """
    assert np.isclose(
        abs(getattr(fs_gacode, attr).magnitude),
        abs(getattr(fs_geqdsk, attr).magnitude),
        rtol=1e-2,
    )


@pytest.mark.parametrize("data_var", ["R", "Z", "B_poloidal"])
def test_compare_gacode_geqdsk_data_vars(fs_gacode, fs_geqdsk, data_var):
    """
    Compare data vars between equivalent flux surfaces from GACODE CDF and GEQDSK
    files. Assumes FluxSurface has handled units correctly, and only checks that values
    are within 10%. Interpolates from the CDF theta grid to the GEQDSK one.
    """
    theta_geqdsk = fs_geqdsk["theta"].data.magnitude
    data_var_geqdsk = fs_geqdsk[data_var].data.magnitude
    theta_cdf = fs_gacode["theta"].data.magnitude
    data_var_cdf = np.interp(
        theta_geqdsk,
        theta_cdf,
        fs_gacode[data_var].data.magnitude,
        period=2 * np.pi,
    )
    assert_allclose(data_var_cdf, data_var_geqdsk, rtol=1e-2, atol=1e-4)
