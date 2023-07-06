from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryFourierGENE

from test_miller import generate_miller
from pyrokinetics.equilibrium import read_equilibrium

import numpy as np
import pytest

atol = 1e-2
rtol = 1e-3


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    n_moments = 32

    cN = np.array([1.0, *[0.0] * (n_moments - 1)])

    sN = np.array([*[0.0] * n_moments])

    lg = LocalGeometryFourierGENE(
        {
            "cN": cN,
            "sN": sN,
            "a_minor": 1.0,
            "Rmaj": 0.0,
            "Z0": 0.0,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.allclose(R ** 2 + Z ** 2, np.ones(length))


def test_flux_surface_elongation():
    length = 129
    theta = np.linspace(0.0, 2 * np.pi, length)

    Rmaj = 3.0
    elongation = 5.0
    miller = generate_miller(
        theta=theta, kappa=elongation, delta=0.0, Rmaj=Rmaj, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    R, Z = fourier.get_flux_surface(theta)

    assert np.isclose(np.min(R), 2.0, atol=atol)
    assert np.isclose(np.max(R), 4.0, atol=atol)
    assert np.isclose(np.min(Z), -5.0, atol=atol)
    assert np.isclose(np.max(Z), 5.0, atol=atol)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=1.0, delta=0.5, Rmaj=3.0, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
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


def test_flux_surface_long_triangularity():
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    high_res_theta = np.linspace(-np.pi, np.pi, length)
    R, Z = fourier.get_flux_surface(high_res_theta)

    assert np.isclose(np.min(R), -1.0, atol=atol)
    assert np.isclose(np.max(R), 3.0, atol=atol)
    assert np.isclose(np.min(Z), -4.0, atol=atol)
    assert np.isclose(np.max(Z), 4.0, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 0.0, atol=atol)

    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 0.0, atol=atol)


def test_default_bunit_over_b0():
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)
    miller = generate_miller(theta)

    fourier = LocalGeometryFourierGENE()
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

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    assert np.allclose(
        fourier.get_grad_r(theta=fourier.theta_eq),
        expected(theta),
        atol=atol,
    )


def test_load_from_eq():
    """Golden answer test"""

    eq = read_equilibrium(template_dir / "test.geqdsk", "GEQDSK")

    fourier = LocalGeometryFourierGENE()
    fourier.from_global_eq(eq, 0.5)

    assert fourier["local_geometry"] == "FourierGENE"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "bt_ccw": 1,
        "bunit_over_b0": 3.5688826501910373,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.874010706550275,
        "Fpsi": 6.096777229999999,
        "ip_ccw": 1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "shift": 0.18409275837942818,
        "dZ0dr": -0.00939035703062135,
        "cN": [
            1.10827623e00,
            -5.30195594e-02,
            -5.73146297e-01,
            1.06053809e-01,
            2.01208245e-01,
            -9.13165947e-02,
            -6.42359294e-02,
            6.13266100e-02,
            1.12528646e-02,
            -3.45849942e-02,
            6.62203349e-03,
            1.60792526e-02,
            -9.81697677e-03,
            -5.30272581e-03,
            7.68178186e-03,
            -2.02630146e-05,
            -4.64347808e-03,
            1.79327331e-03,
            2.02415118e-03,
            -1.98798740e-03,
            -5.06056953e-04,
            1.38667639e-03,
            -3.25603489e-04,
            -8.53108926e-04,
            4.22477892e-04,
            2.14166033e-04,
            -5.39729148e-04,
            -9.69026453e-05,
            2.17492554e-04,
            -1.94840528e-04,
            -2.28742844e-04,
            4.41312850e-05,
        ],
        "sN": [
            0.00000000e00,
            -4.32975896e-06,
            9.03004175e-05,
            -2.34144020e-05,
            -6.47799889e-05,
            3.64033670e-05,
            2.94531152e-05,
            -3.64123777e-05,
            -7.34591313e-06,
            2.42953644e-05,
            -7.60372114e-06,
            -1.57993781e-05,
            8.49253786e-06,
            3.59872685e-06,
            -1.10889325e-05,
            -1.76773820e-06,
            4.19216834e-06,
            -5.10700437e-06,
            -5.66699265e-06,
            8.16710401e-07,
            -1.68437420e-06,
            -5.47790459e-06,
            -2.45904134e-06,
            -1.21640368e-06,
            -4.03232065e-06,
            -3.97966506e-06,
            -2.31837290e-06,
            -3.21063594e-06,
            -4.12636128e-06,
            -3.44535012e-06,
            -3.31754195e-06,
            -4.09954249e-06,
        ],
    }

    for key, value in expected.items():
        assert np.allclose(
            fourier[key],
            value,
            rtol=1e-3,
        ), f"{key} difference: {fourier[key] - value}"

    fourier.R, fourier.Z = fourier.get_flux_surface(fourier.theta_eq, normalised=False)
    assert np.isclose(min(fourier.R), 1.7476563059555796)
    assert np.isclose(max(fourier.R), 3.8023514986250713)
    assert np.isclose(min(fourier.Z), -3.112945604763297)
    assert np.isclose(max(fourier.Z), 3.112868609690877)
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

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller)

    assert np.allclose(
        fourier.get_b_poloidal(fourier.theta_eq),
        expected(theta),
        atol=atol,
    )
