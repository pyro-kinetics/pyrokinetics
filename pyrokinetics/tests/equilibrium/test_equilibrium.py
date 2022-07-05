"""This set of tests use golden answers. If templates/test.geqdsk
changes, these tests may fail.

"""

from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics import Pyro
from pyrokinetics import template_dir
import numpy as np
import pytest


@pytest.fixture(scope="module")
def geqdsk_equilibrium():
    return Equilibrium(eq_type="GEQDSK", eq_file=template_dir.joinpath("test.geqdsk"))


@pytest.fixture(scope="module")
def geqdsk_equilibrium_kwargs():
    return Equilibrium(
        eq_type="GEQDSK", eq_file=template_dir.joinpath("test.geqdsk"), psi_n_lcfs=0.99
    )


@pytest.fixture(scope="module")
def transp_cdf_equilibrium():
    return Equilibrium(
        eq_type="TRANSP", eq_file=template_dir.joinpath("transp_eq.cdf"), time=0.2
    )


@pytest.fixture(scope="module")
def transp_gq_equilibrium():
    return Equilibrium(
        eq_type="GEQDSK", eq_file=template_dir.joinpath("transp_eq.geqdsk")
    )


def test_read_geqdsk(geqdsk_equilibrium):
    eq = geqdsk_equilibrium

    assert eq.nr == 69
    assert eq.nz == 175
    assert np.isclose(np.min(eq.lcfs_R), 1.0)
    assert np.isclose(np.max(eq.lcfs_R), 4.0001495)
    assert np.isclose(np.min(eq.lcfs_Z), -4.19975)
    assert np.isclose(np.max(eq.lcfs_Z), 4.19975)


def test_read_geqdsk_kwargs(geqdsk_equilibrium_kwargs):
    eq = geqdsk_equilibrium_kwargs

    assert eq.nr == 69
    assert eq.nz == 175
    assert np.isclose(np.min(eq.lcfs_R), 1.0128680915032113)
    assert np.isclose(np.max(eq.lcfs_R), 3.996755515)
    assert np.isclose(np.min(eq.lcfs_Z), -4.179148464960308)
    assert np.isclose(np.max(eq.lcfs_Z), 4.179148464960308)


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


def assert_within_ten_percent(key, cdf_value, gq_value):

    difference = np.abs((cdf_value - gq_value))
    smallest_value = np.min(np.abs([cdf_value, gq_value]))

    if smallest_value == 0.0:
        if difference == 0.0:
            assert True
        else:
            assert (
                np.abs((cdf_value - gq_value) / np.min(np.abs([cdf_value, gq_value])))
                < 0.1
            )
    else:
        assert difference / smallest_value < 0.5


def test_compare_transp_cdf_geqdsk(transp_cdf_equilibrium, transp_gq_equilibrium):
    psi_surface = 0.5

    # Load up pyro object and generate local Miller parameters at psi_n=0.5
    # FIXME Pyro should read eq file, should not be inserting it manually
    pyro_gq = Pyro()
    pyro_gq.eq = transp_gq_equilibrium
    pyro_gq.load_local_geometry(psi_n=psi_surface, local_geometry="Miller")

    # Load up pyro object
    pyro_cdf = Pyro()
    pyro_cdf.eq = transp_cdf_equilibrium
    pyro_cdf.load_local_geometry(psi_n=psi_surface, local_geometry="Miller")

    ignored_geometry_attrs = [
        "B0",
        "psi_n",
        "r_minor",
        "a_minor",
        "f_psi",
        "R",
        "Z",
        "theta",
        "b_poloidal",
        "dpsidr",
        "pressure",
        "dpressure_drho",
        "Z0",
        "local_geometry",
    ]

    for key in pyro_gq.local_geometry.keys():
        if key in ignored_geometry_attrs:
            continue
        assert_within_ten_percent(
            key, pyro_cdf.local_geometry[key], pyro_gq.local_geometry[key]
        )


@pytest.mark.parametrize(
    "filename, eq_type",
    [
        ("transp_eq.cdf", "TRANSP"),
        ("transp_eq.geqdsk", "GEQDSK"),
        ("test.geqdsk", "GEQDSK"),
    ],
)
def test_filetype_inference(filename, eq_type):
    eq = Equilibrium(template_dir.joinpath(filename))
    assert eq.eq_type == eq_type


def test_bad_eq_type(geqdsk_equilibrium):
    with pytest.raises(ValueError):
        geqdsk_equilibrium.eq_type = "helloworld"
