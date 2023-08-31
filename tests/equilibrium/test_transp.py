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
from pyrokinetics.equilibrium.transp import EquilibriumReaderTRANSP


@pytest.fixture
def example_file():
    return template_dir / "transp_eq.cdf"


def test_read(example_file):
    """
    Ensure it can read the example TRANSP file, and that it produces an Equilibrium
    """
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    result = EquilibriumReaderTRANSP()(example_file)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    assert isinstance(result, Equilibrium)


def test_verify_file_type(example_file):
    """Ensure verify_file_type completes without throwing an error"""
    EquilibriumReaderTRANSP().verify_file_type(example_file)


def test_read_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderTRANSP()(filename)


def test_read_file_is_not_netcdf():
    """Ensure failure when given a non-netcdf file"""
    filename = template_dir / "input.gs2"
    with pytest.raises(Exception):
        EquilibriumReaderTRANSP()(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_read_file_is_not_transp(filename):
    """Ensure failure when given a non-transp netcdf file
    This could fail for any number of reasons during processing.
    """
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderTRANSP()(filename)


def test_verify_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderTRANSP().verify_file_type(filename)


def test_verify_file_is_not_netcdf():
    """Ensure failure when given a non-netcdf file"""
    filename = template_dir / "input.gs2"
    with pytest.raises(Exception):
        EquilibriumReaderTRANSP().verify(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_verify_file_is_not_transp(filename):
    """Ensure failure when given a non-transp netcdf file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderTRANSP().verify(filename)


# Compare GEQDSK and TRANSP files of the same Equilibrium
# Compare only the flux surface at ``psi_n=0.5``.


@pytest.fixture(scope="module")
def fs_cdf():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    fs = read_equilibrium(template_dir / "transp_eq.cdf", time=0.2).flux_surface(0.5)
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
def test_compare_transp_cdf_geqdsk_attrs(fs_cdf, fs_geqdsk, attr):
    """
    Compare attributes between equivalent flux surfaces from TRANSP CDF and GEQDSK
    files. Assumes FluxSurface has handled units correctly, and only checks that values
    are within 10%.
    """
    assert np.isclose(
        getattr(fs_cdf, attr).magnitude,
        getattr(fs_geqdsk, attr).magnitude,
        rtol=1e-1,
    )


@pytest.mark.parametrize("data_var", ["R", "Z", "B_poloidal"])
def test_compare_transp_cdf_geqdsk_data_vars(fs_cdf, fs_geqdsk, data_var):
    """
    Compare data vars between equivalent flux surfaces from TRANSP CDF and GEQDSK
    files. Assumes FluxSurface has handled units correctly, and only checks that values
    are within 10%. Interpolates from the CDF theta grid to the GEQDSK one.
    """
    theta_geqdsk = fs_geqdsk["theta"].data.magnitude
    data_var_geqdsk = fs_geqdsk[data_var].data.magnitude
    theta_cdf = fs_cdf["theta"].data.magnitude
    data_var_cdf = np.interp(
        theta_geqdsk,
        theta_cdf,
        fs_cdf[data_var].data.magnitude,
        period=2 * np.pi,
    )
    assert_allclose(data_var_cdf, data_var_geqdsk, rtol=1e-1, atol=1e-2)
