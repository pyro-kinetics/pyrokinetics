"""This set of tests use golden answers. If templates/test.geqdsk
changes, these tests may fail.

"""


from pyrokinetics.equilibrium import Equilibrium
import numpy as np
import pathlib
import pytest


@pytest.fixture(scope="module")
def geqdsk_equilibrium():
    template_dir = pathlib.Path(__file__).parent / "../templates"
    return Equilibrium(eq_type="GEQDSK", eq_file=template_dir / "test.geqdsk")


def test_read_geqdsk(geqdsk_equilibrium):
    eq = geqdsk_equilibrium

    assert eq.nr == 69
    assert eq.nz == 175
    assert np.isclose(np.min(eq.lcfs_R), 1.0)
    assert np.isclose(np.max(eq.lcfs_R), 4.0001495)
    assert np.isclose(np.min(eq.lcfs_Z), -4.19975)
    assert np.isclose(np.max(eq.lcfs_Z), 4.19975)


def test_get_flux_surface(geqdsk_equilibrium):

    R, Z = geqdsk_equilibrium.get_flux_surface(0.5)

    assert np.allclose(
        (min(R), max(R), min(Z), max(Z)),
        (
            1.747667428494825,
            3.8021621078549717,
            -3.112902507930995,
            3.112770914245634,
        ),
    )


def test_b_radial(geqdsk_equilibrium):
    assert np.isclose(geqdsk_equilibrium.get_b_radial(2.3, 3.1), -0.321247509)
    assert np.isclose(geqdsk_equilibrium.get_b_radial(3.8, 0.0), -6.101095916e-06)


def test_b_vertical(geqdsk_equilibrium):
    assert np.isclose(geqdsk_equilibrium.get_b_vertical(2.3, 3.1), -0.026254786738)
    assert np.isclose(geqdsk_equilibrium.get_b_vertical(3.8, 0.0), 1.1586967264)


def test_b_poloidal(geqdsk_equilibrium):
    assert np.isclose(geqdsk_equilibrium.get_b_poloidal(2.3, 3.1), 0.3223185944)
    assert np.isclose(geqdsk_equilibrium.get_b_poloidal(3.8, 0.0), 1.1586967264)


def test_b_toroidal(geqdsk_equilibrium):
    assert np.isclose(geqdsk_equilibrium.get_b_toroidal(2.3, 3.1), 2.6499210636)
    assert np.isclose(geqdsk_equilibrium.get_b_toroidal(3.8, 0.0), 1.6038789003)
