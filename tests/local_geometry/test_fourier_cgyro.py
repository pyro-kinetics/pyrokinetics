from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryFourierCGYRO

from pyrokinetics.equilibrium import read_equilibrium

import numpy as np
import pytest

atol = 1e-2
rtol = 1e-3


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    n_moments = 16

    aR = np.array([0.0, 1.0, *[0.0] * (n_moments - 2)])
    bR = np.array([*[0.0] * n_moments])

    aZ = np.array([*[0.0] * n_moments])
    bZ = np.array([0.0, 1.0, *[0.0] * (n_moments - 2)])

    lg = LocalGeometryFourierCGYRO(
        {"aR": aR, "aZ": aZ, "bR": bR, "bZ": bZ, "a_minor": 1.0}
    )
    R, Z = lg.get_flux_surface(theta)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 501
    theta = np.linspace(0.0, 2 * np.pi, length)
    n_moments = 16

    R0 = 3.0
    elongation = 5.0

    aR = np.array([R0, 1.0, *[0.0] * (n_moments - 2)])
    bR = np.array([*[0.0] * n_moments])

    aZ = np.array([*[0.0] * n_moments])
    bZ = np.array([0.0, elongation, *[0.0] * (n_moments - 2)])

    lg = LocalGeometryFourierCGYRO(
        {"aR": aR, "aZ": aZ, "bR": bR, "bZ": bZ, "a_minor": 1.0}
    )
    R, Z = lg.get_flux_surface(theta)
    assert np.isclose(np.min(R), 2.0)
    assert np.isclose(np.max(R), 4.0)
    assert np.isclose(np.min(Z), -5.0)
    assert np.isclose(np.max(Z), 5.0)


def test_flux_surface_triangularity(generate_miller):
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=0.5, Rmaj=3.0, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller)

    R, Z = fourier.get_flux_surface(fourier.theta_eq)

    assert np.isclose(np.min(R), 2.0, atol=atol)
    assert np.isclose(np.max(R), 4.0, atol=atol)
    assert np.isclose(np.min(Z), -1.0, atol=atol)
    assert np.isclose(np.max(Z), 1.0, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 2.5, atol=atol)
    assert np.isclose(Z[top_corner], 1.0, atol=atol)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 2.5, atol=atol)
    assert np.isclose(Z[bottom_corner], -1.0, atol=atol)


def test_flux_surface_long_triangularity(generate_miller):
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller)

    high_res_theta = np.linspace(-np.pi, np.pi, length)
    R, Z = fourier.get_flux_surface(high_res_theta)

    assert np.isclose(np.min(R), -1.0, atol=atol)
    assert np.isclose(np.max(R), 3.0, atol=atol)
    assert np.isclose(np.min(Z), -4.0, atol=atol)
    assert np.isclose(np.max(Z), 4.0, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 0.01, atol=atol)

    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 0.01, atol=atol)


def test_default_bunit_over_b0(generate_miller):
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)
    miller = generate_miller(theta)

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller)

    assert np.isclose(fourier.get_bunit_over_b0(), 1.014082493337769)


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
def test_grad_r(generate_miller, parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 129
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(theta, dict=parameters)

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller)

    assert np.allclose(
        fourier.get_grad_r(theta=fourier.theta_eq),
        expected(theta),
        atol=atol,
    )


def test_load_from_eq():
    """Golden answer test"""

    eq = read_equilibrium(template_dir / "test.geqdsk", "GEQDSK")

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_global_eq(eq, 0.5)

    assert fourier["local_geometry"] == "FourierCGYRO"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "bt_ccw": 1,
        "bunit_over_b0": 3.563738638472842,
        "dpsidr": 1.874010706550275,
        "Fpsi": 6.096777229999999,
        "ip_ccw": 1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "aR": [
            2.63000039e00,
            1.13517381e00,
            1.59736559e-01,
            -1.42502624e-01,
            -1.41101558e-02,
            4.91382630e-02,
            -4.06944953e-03,
            -2.14594402e-02,
            6.35321799e-03,
            9.94019739e-03,
            -5.64566344e-03,
            -4.65236826e-03,
            4.21627257e-03,
            1.79676255e-03,
            -3.20186240e-03,
            -6.93915030e-04,
        ],
        "aZ": [
            -1.62968327e-04,
            -8.10492998e-05,
            -1.83443032e-04,
            -2.04958841e-04,
            -1.69517064e-04,
            -1.63278600e-04,
            -1.83572188e-04,
            -1.83845361e-04,
            -1.70872792e-04,
            -1.72718236e-04,
            -1.80700753e-04,
            -1.77588553e-04,
            -1.72311671e-04,
            -1.75151800e-04,
            -1.77878474e-04,
            -1.74787011e-04,
        ],
        "bR": [
            0.00000000e00,
            -5.08192363e-06,
            -2.01490392e-05,
            1.39563866e-05,
            4.59695977e-06,
            -9.86024844e-06,
            1.14069964e-06,
            6.08694077e-06,
            -2.17655169e-06,
            -3.70214948e-06,
            2.47459372e-06,
            2.12321479e-06,
            -2.17140701e-06,
            -9.73297353e-07,
            1.92301694e-06,
            3.82788065e-07,
        ],
        "bZ": [
            0.00000000e00,
            2.72966806e00,
            -3.26930543e-02,
            -2.43195872e-01,
            3.46224235e-02,
            6.89928834e-02,
            -2.32712041e-02,
            -2.59334287e-02,
            1.53390824e-02,
            1.05324565e-02,
            -1.00591785e-02,
            -4.03521446e-03,
            6.60286142e-03,
            1.20562097e-03,
            -4.21479621e-03,
            1.02520181e-04,
        ],
    }
    for key, value in expected.items():
        assert np.allclose(
            fourier[key], value
        ), f"{key} difference: {fourier[key] - value}"

    fourier.R, fourier.Z = fourier.get_flux_surface(fourier.theta_eq, normalised=False)

    assert np.isclose(min(fourier.R), 1.746538630605064)
    assert np.isclose(max(fourier.R), 3.8000199956457803)
    assert np.isclose(min(fourier.Z), -3.1074326938899426)
    assert np.isclose(max(fourier.Z), 3.107261707275496)
    assert all(fourier.theta <= 2 * np.pi)
    assert all(fourier.theta >= 0)


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
def test_b_poloidal(generate_miller, parameters, expected):
    """Analytic answers for this test generated using sympy"""
    length = 129
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(theta, dict=parameters)

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller)

    assert np.allclose(
        fourier.get_b_poloidal(fourier.theta_eq),
        expected(theta),
        atol=atol,
    )
