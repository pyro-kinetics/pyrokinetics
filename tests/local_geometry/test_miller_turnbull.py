from textwrap import dedent
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMillerTurnbull

from pyrokinetics.equilibrium import read_equilibrium

import numpy as np
import pytest


def generate_miller(theta, Rmaj=3.0, rho=0.5, kappa=1.0, delta=0.0, Z0=0.0, dict={}):
    miller = LocalGeometryMillerTurnbull()

    miller.Rmaj = Rmaj
    miller.Z0 = Z0
    miller.rho = rho
    miller.kappa = kappa
    miller.delta = delta
    miller.dpsidr = 1.0
    miller.shift = 0.0
    miller.theta = theta

    if dict:
        for key, value in dict.items():
            miller[key] = value

    miller.R_eq, miller.Z_eq = miller.get_flux_surface(theta)
    miller.R = miller.R_eq
    miller.Z = miller.Z_eq

    miller.b_poloidal_eq = miller.get_b_poloidal(
        theta=miller.theta,
    )
    (
        miller.dRdtheta,
        miller.dRdr,
        miller.dZdtheta,
        miller.dZdr,
    ) = miller.get_RZ_derivatives(miller.theta)

    return miller


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=0.0, Rmaj=0.0, rho=1.0, Z0=0.0
    )

    R, Z = miller.get_flux_surface(theta)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=10.0, delta=0.0, Rmaj=0.0, rho=1.0, Z0=0.0
    )

    R, Z = miller.get_flux_surface(theta)

    assert np.isclose(np.min(R), -1.0)
    assert np.isclose(np.max(R), 1.0)
    assert np.isclose(np.min(Z), -10.0)
    assert np.isclose(np.max(Z), 10.0)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=1.0, Rmaj=0.0, rho=1.0, Z0=0.0
    )

    R, Z = miller.get_flux_surface(theta)

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

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    R, Z = miller.get_flux_surface(theta)

    assert np.isclose(R[0], -1.0)
    assert np.isclose(Z[0], 0.0)
    assert np.isclose(R[length // 4], 0.0)
    assert np.isclose(Z[length // 4], -4.0)
    assert np.isclose(R[length // 2], 3.0)
    assert np.isclose(Z[length // 2], 0.0)
    assert np.isclose(R[length * 3 // 4], 0.0)
    assert np.isclose(Z[length * 3 // 4], 4.0)


def test_default_bunit_over_b0():
    miller = LocalGeometryMillerTurnbull()

    assert np.isclose(miller.get_bunit_over_b0(), 1.0140827407220696)


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

    miller = generate_miller(theta, dict=parameters)
    assert np.allclose(
        miller.get_grad_r(theta=theta),
        expected(theta),
    )


def test_load_from_eq():
    """Golden answer test"""

    eq = read_equilibrium(template_dir / "test.geqdsk", "GEQDSK")
    miller = LocalGeometryMillerTurnbull()
    miller.from_global_eq(eq, 0.5)

    assert miller["local_geometry"] == "MillerTurnbull"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "btccw": -1,
        "bunit_over_b0": 3.57683215058002,
        "delta": 0.4623178370292059,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.874010706550275,
        "Fpsi": 6.096777229999999,
        "ipccw": -1,
        "kappa": 3.0302699173285554,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "s_delta": 0.3097805640684271,
        "s_kappa": -0.21353796880440293,
        "s_zeta": 0.052864547553089176,
        "shat": 0.7706147138551124,
        "shift": -0.5744560594220514,
        "zeta": 0.07019426799850659,
    }
    for key, value in expected.items():
        actual = miller[key]
        err_string = dedent(
            f"""\
            {key}
            actual: {actual}
            expected: {value}
            abs_err: {actual - value}
            rel_err: {(actual - value) / np.nextafter(value, np.inf)}"
            """
        )
        # Accurate to 0.5%. May need to update golden answer values
        assert np.isclose(actual, value, rtol=5e-3), err_string

    miller.R, miller.Z = miller.get_flux_surface(miller.theta, normalised=False)

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
                "dpsidr": 1.0,
                "Rmaj": 1.0,
            },
            lambda theta: 1 / (1 + 0.5 * np.cos(theta)),
        ),
        (
            {
                "kappa": 1.0,
                "delta": 0.0,
                "s_kappa": 1.0,
                "s_delta": 0.0,
                "shift": 0.0,
                "dpsidr": 3.0,
                "Rmaj": 2.5,
            },
            lambda theta: 3 / ((2.5 + 0.5 * np.cos(theta)) * (np.sin(theta) ** 2 + 1)),
        ),
        (
            {
                "kappa": 2.0,
                "delta": 0.5,
                "s_kappa": 0.5,
                "s_delta": 0.2,
                "shift": 0.1,
                "dpsidr": 0.3,
                "Rmaj": 2.5,
            },
            lambda theta: 0.3
            * np.sqrt(
                0.25
                * (0.523598775598299 * np.cos(theta) + 1.0) ** 2
                * np.sin(theta + 0.523598775598299 * np.sin(theta)) ** 2
                + np.cos(theta) ** 2
            )
            / (
                (0.5 * np.cos(theta + 0.523598775598299 * np.sin(theta)) + 2.5)
                * (
                    (0.585398163397448 * np.cos(theta) + 0.5)
                    * np.sin(theta)
                    * np.sin(theta + 0.523598775598299 * np.sin(theta))
                    + 0.1 * np.cos(theta)
                    + np.cos(0.523598775598299 * np.sin(theta))
                )
            ),
        ),
    ],
)
def test_b_poloidal(parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 65
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(theta, dict=parameters)

    assert np.allclose(
        miller.b_poloidal_eq,
        expected(theta),
    )
