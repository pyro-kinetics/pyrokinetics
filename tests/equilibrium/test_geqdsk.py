import warnings

import numpy as np
import pytest
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import (
    Equilibrium,
    EquilibriumCOCOSWarning,
    read_equilibrium,
)
from pyrokinetics.equilibrium.geqdsk import EquilibriumReaderGEQDSK
from pyrokinetics.normalisation import ureg as units


@pytest.fixture(params=["test.geqdsk", "transp_eq.geqdsk"], scope="module")
def example_file(request):
    return template_dir / request.param


@pytest.fixture(scope="module")
def geqdsk_equilibrium():
    warnings.simplefilter("ignore", category=EquilibriumCOCOSWarning)
    eq = read_equilibrium(template_dir / "test.geqdsk")
    warnings.simplefilter("default", category=EquilibriumCOCOSWarning)
    return eq


def test_read(example_file):
    """
    Ensure it can read the example GEQDSK file, and that it produces an Equilibrium
    """
    warnings.simplefilter("ignore", category=EquilibriumCOCOSWarning)
    result = EquilibriumReaderGEQDSK()(example_file)
    warnings.simplefilter("default", category=EquilibriumCOCOSWarning)
    assert isinstance(result, Equilibrium)


def test_verify_file_type(example_file):
    """Ensure verify_file_type completes without throwing an error"""
    EquilibriumReaderGEQDSK().verify_file_type(example_file)


def test_read_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderGEQDSK()(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
def test_read_file_is_not_geqdsk(filename):
    """Ensure failure when given a non-geqdsk file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderGEQDSK()(filename)


def test_verify_file_does_not_exist():
    """Ensure failure when given a non-existent file"""
    filename = template_dir / "helloworld"
    with pytest.raises((FileNotFoundError, ValueError)):
        EquilibriumReaderGEQDSK().verify_file_type(filename)


@pytest.mark.parametrize("filename", ["input.gs2", "transp_eq.cdf"])
def test_verify_file_is_not_geqdsk(filename):
    """Ensure failure when given a non-geqdsk file"""
    filename = template_dir / filename
    with pytest.raises(Exception):
        EquilibriumReaderGEQDSK().verify_file_type(filename)


@pytest.mark.parametrize("clockwise_phi", (True, False))
def test_geqdsk_auto_cocos(example_file, clockwise_phi):
    """
    The file 'test.geqdsk' is known to be COCOS 1, while 'transp_eq.geqdsk' is known
    to be COCOS 5. Test that this is captured.
    Setting ``clockwise_phi`` to True will shift these to COCOS 2 or 6 respectively.
    """
    expected_cocos = (5 if "transp" in example_file.name else 1) + clockwise_phi
    with warnings.catch_warnings(record=True) as w_log:
        try:
            read_equilibrium(example_file, clockwise_phi=clockwise_phi)
        except Exception:
            # As these files don't actually use clockwise phi, we should expect these
            # to fail. This simply checks that the value is correctly passed on to
            # Equilibrium and that it infers COCOS correctly.
            pass
        assert len(w_log) == 1
        assert issubclass(w_log[0].category, EquilibriumCOCOSWarning)
        assert f"COCOS {expected_cocos}." in str(w_log[0].message)


def test_geqdsk_psi_n_lcfs(example_file):
    """
    Test that psi_n_lcfs works as expected, i.e. shrinking The psi_grid.
    We use psi_n_lcfs=0.5 here for convenience. This is far lower than it should be in
    actual usage!
    """
    warnings.simplefilter("ignore", category=EquilibriumCOCOSWarning)
    eq_full = read_equilibrium(example_file)
    eq_half = read_equilibrium(example_file, psi_n_lcfs=0.5)
    warnings.simplefilter("default", category=EquilibriumCOCOSWarning)

    # The last psi should be halved, but not the first one
    assert np.isclose(
        eq_half["psi"][0].data.magnitude, eq_full["psi"][0].data.magnitude
    )
    assert np.isclose(
        eq_half["psi"][-1].data.magnitude, eq_full["psi"][-1].data.magnitude / 2
    )
    assert np.isclose(eq_half.psi_axis.magnitude, eq_full.psi_axis.magnitude)
    assert np.isclose(eq_half.psi_lcfs.magnitude, eq_full.psi_lcfs.magnitude / 2)

    # The values of each spline function at psi_n=0.5 on eq_full should match those at
    # psi_n=1.0 on eq_half
    # Don't check prime values besides those passed in to Equilibrium, as truncating the
    # psi grid can cause the derivatives at the end to vary a lot.
    # TODO is this a bug?
    assert np.isclose(eq_half.F(1.0).magnitude, eq_full.F(0.5).magnitude)
    assert np.isclose(eq_half.FF_prime(1.0).magnitude, eq_full.FF_prime(0.5).magnitude)
    assert np.isclose(eq_half.p(1.0).magnitude, eq_full.p(0.5).magnitude)
    assert np.isclose(eq_half.p_prime(1.0).magnitude, eq_full.p_prime(0.5).magnitude)
    assert np.isclose(eq_half.q(1.0).magnitude, eq_full.q(0.5).magnitude)
    assert np.isclose(eq_half.R_major(1.0).magnitude, eq_full.R_major(0.5).magnitude)
    assert np.isclose(eq_half.r_minor(1.0).magnitude, eq_full.r_minor(0.5).magnitude)
    assert np.isclose(eq_half.Z_mid(1.0).magnitude, eq_full.Z_mid(0.5).magnitude)
    assert np.isclose(eq_half.psi(1.0).magnitude, eq_full.psi(0.5).magnitude)


# The following tests use 'golden answers', and depend on template files.
# They may fail if algorithms are updated such that the end results aren't accurate to
# within 1e-5 relative error.


def test_read_geqdsk(geqdsk_equilibrium):
    eq = geqdsk_equilibrium

    assert len(eq["R"]) == 69
    assert len(eq["Z"]) == 175
    assert eq["psi_RZ"].shape[0] == 69
    assert eq["psi_RZ"].shape[1] == 175
    assert len(eq["psi"]) == 69
    assert len(eq["F"]) == 69
    assert len(eq["p"]) == 69
    assert len(eq["q"]) == 69
    assert len(eq["FF_prime"]) == 69
    assert len(eq["p_prime"]) == 69
    assert len(eq["R_major"]) == 69
    assert len(eq["r_minor"]) == 69
    assert len(eq["Z_mid"]) == 69


def test_get_flux_surface(geqdsk_equilibrium):
    fs = geqdsk_equilibrium.flux_surface(0.5)
    assert np.isclose(min(fs["R"].data), 1.747667428494825 * units.m)
    assert np.isclose(max(fs["R"].data), 3.8021621078549717 * units.m)
    assert np.isclose(min(fs["Z"].data), -3.112902507930995 * units.m)
    assert np.isclose(max(fs["Z"].data), 3.112770914245634 * units.m)


def test_get_lcfs(geqdsk_equilibrium):
    lcfs = geqdsk_equilibrium.flux_surface(1.0)
    assert np.isclose(min(lcfs["R"].data), 1.0 * units.m)
    assert np.isclose(max(lcfs["R"].data), 4.0001495 * units.m)
    assert np.isclose(min(lcfs["Z"].data), -4.19975 * units.m)
    assert np.isclose(max(lcfs["Z"].data), 4.19975 * units.m)


def test_B_radial(geqdsk_equilibrium):
    assert np.isclose(geqdsk_equilibrium.B_radial(2.3, 3.1), -0.321247509 * units.tesla)
    assert np.isclose(
        geqdsk_equilibrium.B_radial(3.8, 0.0), -6.101095916e-06 * units.tesla
    )


def test_B_vertical(geqdsk_equilibrium):
    assert np.isclose(
        geqdsk_equilibrium.B_vertical(2.3, 3.1), -0.026254786738 * units.tesla
    )
    assert np.isclose(
        geqdsk_equilibrium.B_vertical(3.8, 0.0), 1.1586967264 * units.tesla
    )


def test_B_poloidal(geqdsk_equilibrium):
    assert np.isclose(
        geqdsk_equilibrium.B_poloidal(2.3, 3.1), 0.3223185944 * units.tesla
    )
    assert np.isclose(
        geqdsk_equilibrium.B_poloidal(3.8, 0.0), 1.1586967264 * units.tesla
    )


def test_B_toroidal(geqdsk_equilibrium):
    assert np.isclose(
        geqdsk_equilibrium.B_toroidal(2.3, 3.1), 2.6499210636 * units.tesla
    )
    assert np.isclose(
        geqdsk_equilibrium.B_toroidal(3.8, 0.0), 1.6038789003 * units.tesla
    )
