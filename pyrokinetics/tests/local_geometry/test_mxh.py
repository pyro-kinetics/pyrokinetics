from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMXH

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

    sym_coeff = np.array([1.0, *[0.0] * (n_moments - 1)])

    asym_coeff = np.array([*[0.0] * n_moments])

    lg = LocalGeometryMXH(
        {
            "Rmaj": 0.0,
            "Z0": 0.0,
            "kappa": 1.0,
            "rho": 1.0,
            "a_minor": 1.0,
            "sym_coeff": sym_coeff,
            "asym_coeff": asym_coeff,
            "n_moments": n_moments,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.allclose(R ** 2 + Z ** 2, np.ones(length))


def test_flux_surface_elongation():
    length = 129
    theta = np.linspace(0.0, 2 * np.pi, length)
    n_moments = 4

    Rmaj = 3.0
    elongation = 5.0

    sym_coeff = np.array([1.0, *[0.0] * (n_moments - 1)])

    asym_coeff = np.array([*[0.0] * n_moments])

    lg = LocalGeometryMXH(
        {
            "Rmaj": Rmaj,
            "Z0": 0.0,
            "kappa": elongation,
            "rho": 1.0,
            "a_minor": 1.0,
            "sym_coeff": sym_coeff,
            "asym_coeff": asym_coeff,
            "n_moments": n_moments,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.isclose(np.min(R), 2.0, atol=atol)
    assert np.isclose(np.max(R), 4.0, atol=atol)
    assert np.isclose(np.min(Z), -5.0, atol=atol)
    assert np.isclose(np.max(Z), 5.0, atol=atol)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)
    n_moments = 4

    Rmaj = 0.0
    elongation = 1.0
    delta = 1.0
    rho = 1.0
    tri = np.arcsin(delta)

    sym_coeff = np.array([1.0, tri, *[0.0] * (n_moments - 2)])

    asym_coeff = np.array([*[0.0] * n_moments])

    lg = LocalGeometryMXH(
        {
            "Rmaj": Rmaj,
            "Z0": 0.0,
            "kappa": elongation,
            "rho": rho,
            "a_minor": rho,
            "sym_coeff": sym_coeff,
            "asym_coeff": asym_coeff,
            "n_moments": n_moments,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.isclose(np.min(R), -1.0, atol=atol)
    assert np.isclose(np.max(R), 1.0, atol=atol)
    assert np.isclose(np.min(Z), -1.0, atol=atol)
    assert np.isclose(np.max(Z), 1.0, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], -1.0, atol=atol)
    assert np.isclose(Z[top_corner], 1.0, atol=atol)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], -1.0, atol=atol)
    assert np.isclose(Z[bottom_corner], -1.0, atol=atol)


def test_flux_surface_long_triangularity():
    length = 257
    theta = np.linspace(0, 2 * np.pi, length)
    n_moments = 4

    Rmaj = 1.0
    elongation = 2.0
    delta = 0.5
    rho = 2.0
    tri = np.arcsin(delta)

    sym_coeff = np.array([1.0, tri, *[0.0] * (n_moments - 2)])

    asym_coeff = np.array([*[0.0] * n_moments])

    lg = LocalGeometryMXH(
        {
            "Rmaj": Rmaj,
            "Z0": 0.0,
            "kappa": elongation,
            "rho": rho,
            "a_minor": rho,
            "sym_coeff": sym_coeff,
            "asym_coeff": asym_coeff,
            "n_moments": n_moments,
        }
    )

    R, Z = lg.get_flux_surface(theta)

    assert np.isclose(np.min(R), -1.0, atol=atol)
    assert np.isclose(np.max(R), 3.0, atol=atol)
    assert np.isclose(np.min(Z), -4.0, atol=atol)
    assert np.isclose(np.max(Z), 4.0, atol=atol)

    top_corner = np.argmax(Z)
    assert np.isclose(R[top_corner], 0.0, atol=atol)
    assert np.isclose(Z[top_corner], 4.0, atol=atol)
    bottom_corner = np.argmin(Z)
    assert np.isclose(R[bottom_corner], 0.0, atol=atol)
    assert np.isclose(Z[bottom_corner], -4.0, atol=atol)


def test_default_bunit_over_b0():

    length = 257
    theta = np.linspace(0, 2 * np.pi, length)
    miller = generate_miller(theta)

    mxh = LocalGeometryMXH()
    mxh.from_local_geometry(miller)

    assert np.isclose(mxh.get_bunit_over_b0(), 1.014082493337769)


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

    mxh = LocalGeometryMXH()
    mxh.from_local_geometry(miller)

    assert np.allclose(
        mxh.get_grad_r(theta=mxh.theta_eq),
        expected(theta),
        atol=atol,
    )


def test_load_from_eq():
    """Golden answer test"""

    eq = Equilibrium(template_dir / "test.geqdsk", "GEQDSK")

    mxh = LocalGeometryMXH()
    mxh.from_global_eq(eq, 0.5)

    assert mxh["local_geometry"] == "MXH"

    expected = {
        "B0": 2.197104321877944,
        "Rmaj": 1.8498509607744338,
        "a_minor": 1.5000747773827081,
        "beta_prime": -0.9189081293324618,
        "btccw": -1,
        "bunit_over_b0": 3.5724247491369705,
        "dpressure_drho": -1764954.8121591895,
        "dpsidr": 1.874010706550275,
        "f_psi": 6.096777229999999,
        "ipccw": -1,
        "pressure": 575341.528,
        "q": 4.29996157,
        "r_minor": 1.0272473396800734,
        "rho": 0.6847974215474699,
        "shat": 0.7706147138551124,
        "kappa": 3.0302699173285554,
        "shift": -0.5766642393151462,
        "s_kappa": -0.3006247311719641,
        "dZ0dr": 2.1507304057489042e-05,
        "sym_coeff": [0.0, 0.45963629, -0.06939354, 0.00051382],
        "asym_coeff": [-6.85677163e-05, 1.04062899e-04, 2.58449844e-05, 4.47232369e-05],
    }

    for key, value in expected.items():
         assert np.allclose(
            mxh[key], value
         ), f"{key} difference: {mxh[key] - value}"

    mxh.R, mxh.Z = mxh.get_flux_surface(mxh.theta_eq, normalised=False)
    assert np.isclose(min(mxh.R), 1.7476674490324815)
    assert np.isclose(max(mxh.R), 3.8021620986302636)
    assert np.isclose(min(mxh.Z), -3.112902507930995)
    assert np.isclose(max(mxh.Z), 3.1127709142456346)
    assert all(mxh.theta <= 2 * np.pi)
    assert all(mxh.theta >= 0)


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

    mxh = LocalGeometryMXH()
    mxh.from_local_geometry(miller)

    assert np.allclose(
        mxh.get_b_poloidal(mxh.theta_eq),
        expected(theta),
        atol=atol,
    )
