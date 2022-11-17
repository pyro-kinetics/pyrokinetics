from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryFourier
from pyrokinetics.local_geometry.LocalGeometryFourier import (
    grad_r,
    flux_surface,
    get_b_poloidal,
)

from pyrokinetics.local_geometry import LocalGeometryMiller

from pyrokinetics.equilibrium import Equilibrium
from scipy.integrate import simpson

import numpy as np
import pytest


def generate_miller(theta, Rcen=3.0, rmin=1.0, kappa=1.0, delta=0.0):
    miller = LocalGeometryMiller()

    miller.Rmaj = Rcen
    miller.r_minor = rmin
    miller.kappa = kappa
    miller.delta = delta
    miller.dpsidr = 1.0
    miller.shift = 0.0

    miller.set_R_Z_b_poloidal(theta)

    return miller


"""
def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    cN = np.array([1.0, *[0.0] * 31])
    sN = np.array([0.0, *[0.0] * 31])

    R, Z = get_flux_surface(theta=theta, cN=cN, sN=sN, R_major=0.0, a_minor=1.0, Zmid=0.0)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 501
    theta = np.linspace(0.0, 2 * np.pi, length)

    miller = generate_miller(theta, kappa=5.0)

    local_geometry = LocalGeometryFourier()
    local_geometry.from_local_geometry(miller, show_fit=True)

    R = local_geometry.R
    Z = local_geometry.Z

    assert np.isclose(np.min(R), 2.0)
    assert np.isclose(np.max(R), 4.0)
    assert np.isclose(np.min(Z), -5.0)
    assert np.isclose(np.max(Z), 5.0)

def test_flux_surface_triangularity():
    length = 501
    theta = np.linspace(0.0, 2 * np.pi, length)

    miller = generate_miller(theta, delta=0.5)

    local_geometry = LocalGeometryFourier()
    local_geometry.from_local_geometry(miller, show_fit=True)

    R = local_geometry.R
    Z = local_geometry.Z

    assert np.isclose(np.min(R), 2.0)
    assert np.isclose(np.max(R), 4.0)
    assert np.isclose(np.min(Z), -1.0, atol=1e-3)
    assert np.isclose(np.max(Z), 1.0, atol=1e-2)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 2.4983638406992683)
    assert np.isclose(Z[top_corner], 1.0, atol=1e-2)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 2.4983638406992683)
    assert np.isclose(Z[bottom_corner], -1.0, atol=1e-2)


def test_flux_surface_long_triangularity():
    length = 501
    theta = np.linspace(0.0, 2*np.pi, length)

    miller = generate_miller(theta, kappa=2.0, delta=0.5, Rcen=3.0, rmin=2.0)

    local_geometry = LocalGeometryFourier()
    local_geometry.from_local_geometry(miller, show_fit=True)

    R = local_geometry.R
    Z = local_geometry.Z

    assert np.isclose(R[0], 5.0)
    assert np.isclose(Z[0], 0.0)
    assert np.isclose(R[length // 2], 1.0)
    assert np.isclose(Z[length // 2], 0.0)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 2.0265033168804836)
    assert np.isclose(Z[bottom_corner], -4.0, atol=1e-2)
"""


def test_default_bunit_over_b0():
    fourier = LocalGeometryFourier()

    assert np.isclose(fourier.get_bunit_over_b0(), 1.0140827407220696)


def test_load_from_eq():
    """Golden answer test"""

    eq = Equilibrium(template_dir / "test.geqdsk", "GEQDSK")
    miller = LocalGeometryMiller()
    miller.from_global_eq(eq, 0.5)

    fourier = LocalGeometryFourier()
    fourier.from_global_eq(eq, 0.5)
    miller.get_bunit_over_b0()
    fourier.get_bunit_over_b0()
    print(miller.dpsidr)
    print(fourier.dpsidr)

    """
    n_moments = len(fourier.cN)
    n = np.linspace(0, n_moments - 1, n_moments)

    ntheta = n[:, None] * fourier.theta[None, :]
    aN = (
        np.sum(fourier.cN[:, None] * np.cos(ntheta) + fourier.sN[:, None] * np.sin(ntheta), axis=0)
    )
    import matplotlib.pyplot as plt
    plt.plot(fourier.theta, np.gradient(aN, fourier.theta))

    myaN = np.sqrt((fourier.R - fourier.Rmaj*fourier.a_minor) ** 2 + (fourier.Z - fourier.Z0*fourier.a_minor) ** 2) / fourier.a_minor
    plt.plot(fourier.theta, np.gradient(myaN, fourier.theta), '--')
    plt.show()
    """
    assert fourier["local_geometry"] == "Fourier"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "btccw": -1,
        "bunit_over_b0": 3.552564715038472,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.887870561484361,
        "f_psi": 6.096777229999999,
        "ipccw": -1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "shift": -0.5768859822950385,
        "zeta": 0.0,
    }
    for key, value in expected.items():
        assert np.isclose(
            fourier[key], value
        ), f"{key} difference: {fourier[key] - value}"

    assert np.isclose(min(fourier.R), 1.747667428494825)
    assert np.isclose(max(fourier.R), 3.8021621078549717)
    assert np.isclose(min(fourier.Z), -3.112902507930995)
    assert np.isclose(max(fourier.Z), 3.112770914245634)
    assert all(fourier.theta < np.pi)
    assert all(fourier.theta > -np.pi)


# @pytest.mark.parametrize(
#     ["parameters", "expected"],
#     [
#         (
#             {
#                 "kappa": 1.0,
#                 "delta": 0.0,
#                 "s_kappa": 0.0,
#                 "s_delta": 0.0,
#                 "shift": 0.0,
#                 "dpsi_dr": 1.0,
#                 "R": 1.0,
#             },
#             lambda theta: np.ones(theta.shape),
#         ),
#         (
#             {
#                 "kappa": 1.0,
#                 "delta": 0.0,
#                 "s_kappa": 1.0,
#                 "s_delta": 0.0,
#                 "shift": 0.0,
#                 "dpsi_dr": 3.0,
#                 "R": 2.5,
#             },
#             lambda theta: 1.2 / (np.sin(theta) ** 2 + 1),
#         ),
#         (
#             {
#                 "kappa": 2.0,
#                 "delta": 0.5,
#                 "s_kappa": 0.5,
#                 "s_delta": 0.2,
#                 "shift": 0.1,
#                 "dpsi_dr": 0.3,
#                 "R": 2.5,
#             },
#             lambda theta: 0.24
#             * np.sqrt(
#                 0.25
#                 * (0.523598775598299 * np.cos(theta) + 1) ** 2
#                 * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
#                 + np.cos(theta) ** 2
#             )
#             / (
#                 2.0
#                 * (0.585398163397448 * np.cos(theta) + 0.5)
#                 * np.sin(theta)
#                 * np.sin(theta + 0.523598775598299 * np.sin(theta))
#                 + 0.2 * np.cos(theta)
#                 + 2.0 * np.cos(0.523598775598299 * np.sin(theta))
#             ),
#         ),
#     ],
# )
# def test_b_poloidal(parameters, expected):
#     """Analytic answers for this test generated using sympy"""
#     length = 65
#     theta = np.linspace(-np.pi, np.pi, length)
#     assert np.allclose(
#         get_b_poloidal(**parameters, theta=theta),
#         expected(theta),
#     )
