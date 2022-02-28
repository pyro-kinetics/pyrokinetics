from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_geometry.LocalGeometryMiller import (
    grad_r,
    flux_surface,
    b_poloidal,
)
from pyrokinetics.equilibrium import Equilibrium

import numpy as np
import pytest


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=1.0, delta=0.0, Rcen=0.0, rmin=1.0, Zmid=0.0)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(
        theta=theta, kappa=10.0, delta=0.0, Rcen=0.0, rmin=1.0, Zmid=0.0
    )

    assert np.isclose(np.min(R), -1.0)
    assert np.isclose(np.max(R), 1.0)
    assert np.isclose(np.min(Z), -10.0)
    assert np.isclose(np.max(Z), 10.0)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=1.0, delta=1.0, Rcen=0.0, rmin=1.0, Zmid=0.0)

    assert np.isclose(np.min(R), -1.0)
    assert np.isclose(np.max(R), 1.0)
    assert np.isclose(np.min(Z), -1.0)
    assert np.isclose(np.max(Z), 1.0)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], -1.0)
    assert np.isclose(Z[top_corner], 1.0)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], -1.0)
    assert np.isclose(Z[bottom_corner], -1.0)


def test_flux_surface_long_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=2.0, delta=0.5, Rcen=1.0, rmin=2.0, Zmid=0.0)

    assert np.isclose(R[0], -1.0)
    assert np.isclose(Z[0], 0.0)
    assert np.isclose(R[length // 4], 0.0)
    assert np.isclose(Z[length // 4], -4.0)
    assert np.isclose(R[length // 2], 3.0)
    assert np.isclose(Z[length // 2], 0.0)
    assert np.isclose(R[length * 3 // 4], 0.0)
    assert np.isclose(Z[length * 3 // 4], 4.0)


def test_default_bunit_over_b0():
    miller = LocalGeometryMiller()
    print(miller.get_bunit_over_b0())
    assert np.isclose(miller.get_bunit_over_b0(), 1.0481789952353437)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    [
        (
            {"kappa": 1.0, "delta": 0.0, "s_kappa": 0.0, "s_delta": 0.0, "shift": 0.0},
            lambda theta: np.ones(theta.shape),
        ),
        (
            {"kappa": 1.0, "delta": 0.0, "s_kappa": 1.0, "s_delta": 0.0, "shift": 0.0},
            lambda theta: 1.0 / (np.sin(theta) ** 2 + 1),
        ),
        (
            {"kappa": 2.0, "delta": 0.5, "s_kappa": 0.5, "s_delta": 0.2, "shift": 0.1},
            lambda theta: 2.0
            * np.sqrt(
                0.25
                * (0.523598775598299 * np.cos(theta) + 1) ** 2
                * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
                + np.cos(theta) ** 2
            )
            / (
                2.0
                * (0.585398163397448 * np.cos(theta) + 0.5)
                * np.sin(theta)
                * np.sin(theta + 0.523598775598299 * np.sin(theta))
                + 0.2 * np.cos(theta)
                + 2.0 * np.cos(0.523598775598299 * np.sin(theta))
            ),
        ),
    ],
)
def test_grad_r(parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 65
    theta = np.linspace(-np.pi, np.pi, length)
    assert np.allclose(
        grad_r(**parameters, theta=theta),
        expected(theta),
    )


def test_load_from_eq():
    """Golden answer test"""

    eq = Equilibrium(template_dir / "test.geqdsk", "GEQDSK")
    miller = LocalGeometryMiller()
    miller.load_from_eq(eq, 0.5)

    assert miller["local_geometry"] == "Miller"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "btccw": -1,
        "bunit_over_b0": 3.552564715038472,
        "delta": 0.4623178370292059,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.887870561484361,
        "f_psi": 6.096777229999999,
        "ipccw": -1,
        "kappa": 3.0302699173285554,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "s_delta": 0.24389301726720242,
        "s_kappa": -0.13811725411565282,
        "s_zeta": 0.0,
        "shat": 0.7706147138551124,
        "shift": -0.5768859822950385,
        "zeta": 0.0,
    }
    for key, value in expected.items():
        assert np.isclose(
            miller[key], value
        ), f"{key} difference: {miller[key] - value}"

    assert np.isclose(min(miller.R), 1.747667428494825)
    assert np.isclose(max(miller.R), 3.8021621078549717)
    assert np.isclose(min(miller.Z), -3.112902507930995)
    assert np.isclose(max(miller.Z), 3.112770914245634)
    assert all(miller.theta < np.pi)
    assert all(miller.theta > -np.pi)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    [
        (
            {
                "kappa": 1.0,
                "delta": 0.0,
                "s_kappa": 0.0,
                "s_delta": 0.0,
                "shift": 0.0,
                "dpsi_dr": 1.0,
                "R": 1.0,
            },
            lambda theta: np.ones(theta.shape),
        ),
        (
            {
                "kappa": 1.0,
                "delta": 0.0,
                "s_kappa": 1.0,
                "s_delta": 0.0,
                "shift": 0.0,
                "dpsi_dr": 3.0,
                "R": 2.5,
            },
            lambda theta: 1.2 / (np.sin(theta) ** 2 + 1),
        ),
        (
            {
                "kappa": 2.0,
                "delta": 0.5,
                "s_kappa": 0.5,
                "s_delta": 0.2,
                "shift": 0.1,
                "dpsi_dr": 0.3,
                "R": 2.5,
            },
            lambda theta: 0.24
            * np.sqrt(
                0.25
                * (0.523598775598299 * np.cos(theta) + 1) ** 2
                * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
                + np.cos(theta) ** 2
            )
            / (
                2.0
                * (0.585398163397448 * np.cos(theta) + 0.5)
                * np.sin(theta)
                * np.sin(theta + 0.523598775598299 * np.sin(theta))
                + 0.2 * np.cos(theta)
                + 2.0 * np.cos(0.523598775598299 * np.sin(theta))
            ),
        ),
    ],
)
def test_b_poloidal(parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 65
    theta = np.linspace(-np.pi, np.pi, length)
    assert np.allclose(
        b_poloidal(**parameters, theta=theta),
        expected(theta),
    )
