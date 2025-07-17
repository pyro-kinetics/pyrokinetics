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
from pyrokinetics.equilibrium.eliteinp import EquilibriumReaderELITEINP, read_eqin


@pytest.fixture
def example_file():
    return template_dir / "test.eliteinp"


def test_read(example_file):
    """
    Ensure it can read the example ELITEINP file, and that it produces an Equilibrium
    """
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    result = EquilibriumReaderELITEINP()(example_file)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    assert isinstance(result, Equilibrium)


def test_verify_file_type(example_file):
    """Ensure verify_file_type completes without throwing an error"""
    EquilibriumReaderELITEINP().verify_file_type(example_file)


def test_read_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderELITEINP()(filename)


def test_read_file_is_not_netcdf():
    """Ensure failure when given a non-netcdf file"""
    filename = template_dir / "input.gs2"
    with pytest.raises(Exception):
        EquilibriumReaderELITEINP()(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_read_file_is_not_eliteinp(filename):
    """Ensure failure when given a non-eliteinp file
    This could fail for any number of reasons during processing.
    """
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderELITEINP()(filename)


def test_verify_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderELITEINP().verify_file_type(filename)


@pytest.mark.parametrize("filename", ["jetto.cdf", "scene.cdf"])
def test_verify_file_is_not_eliteinp(filename):
    """Ensure failure when given a non-eliteinp netcdf file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderELITEINP().verify_type(filename)


# Compare GEQDSK and ELITEINP files of the same Equilibrium
# Compare only the flux surface at ``psi_n=0.6080095779502469``.


@pytest.fixture(scope="module")
def fs_eliteinp():
    data_dict = read_eqin(template_dir / "test.eliteinp")
    idx = -80
    R = data_dict["R"][idx, :]
    Z = data_dict["z"][idx, :]
    Bpol = data_dict["Bp"][idx, :]

    Z_mid = (max(Z) + min(Z)) / 2
    R_major = (max(R) + min(R)) / 2

    theta = np.arctan2(Z_mid - Z, R - R_major)

    fs = {"R": R, "Z": Z, "B_poloidal": Bpol, "theta": theta}
    return fs


@pytest.fixture(scope="module")
def fs_pyro():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    fs = read_equilibrium(template_dir / "test.eliteinp").flux_surface(
        0.6080095779502469
    )
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return fs


@pytest.mark.parametrize("data_var", ["R", "Z", "B_poloidal"])
def test_compare_transp_elite_pyro_fs(fs_eliteinp, fs_pyro, data_var):
    """
    Compare data vars between equivalent flux surfaces from direct ELITEINP and Pyro
    files. Assumes FluxSurface has handled units correctly, and only checks that values
    are within 0.1%. Interpolates from the ELITE theta grid to the Pyro one.
    """
    theta_pyro = fs_pyro["theta"].data.magnitude
    data_var_pyro = fs_pyro[data_var].data.magnitude
    theta_eliteinp = fs_eliteinp["theta"]
    data_var_eliteinp = np.interp(
        theta_pyro,
        theta_eliteinp,
        fs_eliteinp[data_var],
        period=2 * np.pi,
    )
    assert_allclose(data_var_eliteinp, data_var_pyro, rtol=1e-3, atol=1e-3)
