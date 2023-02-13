from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryFourierCGYRO

from test_miller import generate_miller
from pyrokinetics.equilibrium import Equilibrium

import numpy as np
import pytest

atol = 1e-2
rtol = 1e-3


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    n_moments = 4

    aR = np.array([0.0, 1.0, *[0.0] * (n_moments - 2)])
    bR = np.array([*[0.0] * n_moments])

    aZ = np.array([*[0.0] * n_moments])
    bZ = np.array([0.0, 1.0, *[0.0] * (n_moments - 2)])

    lg = LocalGeometryFourierCGYRO(
        {"aR": aR, "aZ": aZ, "bR": bR, "bZ": bZ, "n_moments": n_moments, "a_minor": 1.0}
    )
    R, Z = lg.get_flux_surface(theta)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 501
    theta = np.linspace(0.0, 2 * np.pi, length)
    n_moments = 4

    R0 = 3.0
    elongation = 5.0

    aR = np.array([R0, 1.0, *[0.0] * (n_moments - 2)])
    bR = np.array([*[0.0] * n_moments])

    aZ = np.array([*[0.0] * n_moments])
    bZ = np.array([0.0, elongation, *[0.0] * (n_moments - 2)])

    lg = LocalGeometryFourierCGYRO(
        {"aR": aR, "aZ": aZ, "bR": bR, "bZ": bZ, "n_moments": n_moments, "a_minor": 1.0}
    )
    R, Z = lg.get_flux_surface(theta)
    assert np.isclose(np.min(R), 2.0)
    assert np.isclose(np.max(R), 4.0)
    assert np.isclose(np.min(Z), -5.0)
    assert np.isclose(np.max(Z), 5.0)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=0.5, Rmaj=3.0, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller, n_moments=32)

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


def test_flux_surface_long_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_local_geometry(miller, n_moments=32)

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


def test_default_bunit_over_b0():
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
def test_grad_r(parameters, expected):
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

    eq = Equilibrium(template_dir / "test.geqdsk", "GEQDSK")

    fourier = LocalGeometryFourierCGYRO()
    fourier.from_global_eq(eq, 0.5)

    assert fourier["local_geometry"] == "FourierCGYRO"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "btccw": -1,
        "bunit_over_b0": 3.5636109344363525,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.874010706550275,
        "f_psi": 6.096777229999999,
        "ipccw": -1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "aR": [
            2.63000269e00,
            1.13516761e00,
            1.59721651e-01,
            -1.42506661e-01,
            -1.41027815e-02,
            4.91518821e-02,
            -4.04861912e-03,
            -2.14356485e-02,
            6.38168354e-03,
            9.98771000e-03,
            -5.57547180e-03,
            -4.57271559e-03,
            4.30030825e-03,
            1.89524547e-03,
            -3.07752150e-03,
            -5.40034607e-04,
        ],
        "aZ": [
            -0.00043876,
            -0.00063314,
            -0.00073462,
            -0.00075529,
            -0.00071919,
            -0.00071207,
            -0.00073096,
            -0.00072953,
            -0.00071483,
            -0.0007148,
            -0.00072052,
            -0.00071486,
            -0.00070689,
            -0.00070685,
            -0.00070642,
            -0.00069995,
        ],
        "bR": [
            0.00000000e00,
            -4.77374698e-06,
            -1.98636168e-05,
            1.43008200e-05,
            5.15800134e-06,
            -9.17280218e-06,
            1.89780939e-06,
            6.93357697e-06,
            -1.16917651e-06,
            -2.43341187e-06,
            3.81951599e-06,
            3.43717276e-06,
            -6.64550043e-07,
            7.60047209e-07,
            3.68999730e-06,
            2.26742847e-06,
        ],
        "bZ": [
            0.00000000e00,
            2.72968011e00,
            -3.26691911e-02,
            -2.43168711e-01,
            3.46520251e-02,
            6.90384599e-02,
            -2.32067826e-02,
            -2.58642997e-02,
            1.54083337e-02,
            1.06122034e-02,
            -9.96457669e-03,
            -3.93017987e-03,
            6.71682358e-03,
            1.32713191e-03,
            -4.09028014e-03,
            2.33627151e-04,
        ],
    }
    for key, value in expected.items():
        assert np.allclose(
            fourier[key], value
        ), f"{key} difference: {fourier[key] - value}"

    fourier.R, fourier.Z = fourier.get_flux_surface(fourier.theta_eq, normalised=False)

    assert np.isclose(min(fourier.R), 1.746454552038628)
    assert np.isclose(max(fourier.R), 3.800749327303827)
    assert np.isclose(min(fourier.Z), -3.1073950183509633)
    assert np.isclose(max(fourier.Z), 3.107097646545643)
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
def test_b_poloidal(parameters, expected):
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
