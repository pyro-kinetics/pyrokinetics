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

    n_moments = 4

    cN = np.array([1.0, *[0.0] * (n_moments - 1)])

    sN = np.array([*[0.0] * n_moments])

    lg = LocalGeometryFourierGENE(
        {
            "cN": cN,
            "sN": sN,
            "n_moments": n_moments,
            "a_minor": 1.0,
            "Rmaj": 0.0,
            "Z0": 0.0,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.allclose(R**2 + Z**2, np.ones(length))


def test_flux_surface_elongation():
    length = 129
    theta = np.linspace(0.0, 2 * np.pi, length)
    n_moments = 32

    Rmaj = 3.0
    elongation = 5.0
    miller = generate_miller(
        theta=theta, kappa=elongation, delta=0.0, Rmaj=Rmaj, rho=1.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller, n_moments=n_moments)

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
    theta = np.linspace(0, 2 * np.pi, length)

    miller = generate_miller(
        theta=theta, kappa=2.0, delta=0.5, Rmaj=1.0, rho=2.0, Z0=0.0
    )

    fourier = LocalGeometryFourierGENE()
    fourier.from_local_geometry(miller, n_moments=32)

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
        "btccw": -1,
        "bunit_over_b0": 3.568591624957067,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.874010706550275,
        "f_psi": 6.096777229999999,
        "ipccw": -1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "shift": 0.2907564549980965,
        "dZ0dr": 0.03149573070972188,
        "cN": [
            1.10828849e00,
            -5.29653593e-02,
            -5.73081114e-01,
            1.06089723e-01,
            2.01250237e-01,
            -9.12586680e-02,
            -6.41831638e-02,
            6.13728025e-02,
            1.13011992e-02,
            -3.45219914e-02,
            6.68082380e-03,
            1.61223868e-02,
            -9.75653873e-03,
            -5.23536181e-03,
            7.73815776e-03,
            5.19363866e-05,
            -4.57407797e-03,
            1.84460891e-03,
            2.10397415e-03,
            -1.89173576e-03,
            -4.38709155e-04,
            1.45044121e-03,
            -2.29600734e-04,
            -7.43946090e-04,
            5.06035088e-04,
            2.90350076e-04,
            -4.22404979e-04,
            3.05762949e-05,
            3.06289572e-04,
            -8.83927170e-05,
            -8.38358786e-05,
            1.71918515e-04,
        ],
        "sN": [
            0.00000000e00,
            -6.13663760e-06,
            8.64355838e-05,
            -2.88625946e-05,
            -7.18985285e-05,
            2.71217202e-05,
            1.86179769e-05,
            -4.90720816e-05,
            -2.17295389e-05,
            8.73297115e-06,
            -2.51398427e-05,
            -3.51027938e-05,
            -1.20961226e-05,
            -1.80682064e-05,
            -3.41125255e-05,
            -2.65469279e-05,
            -2.17179100e-05,
            -3.18071269e-05,
            -3.33163723e-05,
            -2.81654118e-05,
            -3.14333019e-05,
            -3.57964424e-05,
            -3.37748754e-05,
            -3.30859231e-05,
            -3.62159348e-05,
            -3.65713678e-05,
            -3.55339302e-05,
            -3.66837464e-05,
            -3.75944590e-05,
            -3.70154050e-05,
            -3.69655795e-05,
            -3.75547240e-05,
        ],
    }

    for key, value in expected.items():
        assert np.allclose(
            fourier[key], value
        ), f"{key} difference: {fourier[key] - value}"

    fourier.R, fourier.Z = fourier.get_flux_surface(fourier.theta_eq, normalised=False)
    assert np.isclose(min(fourier.R), 1.7475723675522976)
    assert np.isclose(max(fourier.R), 3.804153646207261)
    assert np.isclose(min(fourier.Z), -3.1127956445053906)
    assert np.isclose(max(fourier.Z), 3.1126709695924224)
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
