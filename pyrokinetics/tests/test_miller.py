from pyrokinetics.miller import Miller, grad_r, flux_surface

import numpy as np
import pytest


def test_basic_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = Miller()
    R, Z = miller.miller_RZ(theta, kappa=1.0, delta=0.0, Rcen=0.0, rmin=1.0)

    assert np.allclose(R ** 2 + Z ** 2, np.ones(length))


def test_basic_elongation():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = Miller()
    R, Z = miller.miller_RZ(theta, kappa=10.0, delta=0.0, Rcen=0.0, rmin=1.0)

    assert np.isclose(np.min(R), -1.0)
    assert np.isclose(np.max(R), 1.0)
    assert np.isclose(np.min(Z), -10.0)
    assert np.isclose(np.max(Z), 10.0)


def test_basic_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = Miller()
    R, Z = miller.miller_RZ(theta, kappa=1.0, delta=1.0, Rcen=0.0, rmin=1.0)

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


def test_basic_long_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    miller = Miller()
    R, Z = miller.miller_RZ(theta, kappa=2.0, delta=0.5, Rcen=1.0, rmin=2.0)

    assert np.isclose(R[0], -1.0)
    assert np.isclose(Z[0], 0.0)
    assert np.isclose(R[length // 4], 0.0)
    assert np.isclose(Z[length // 4], -4.0)
    assert np.isclose(R[length // 2], 3.0)
    assert np.isclose(Z[length // 2], 0.0)
    assert np.isclose(R[length * 3 // 4], 0.0)
    assert np.isclose(Z[length * 3 // 4], 4.0)


def test_flux_surface_circle():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=1.0, delta=0.0, Rcen=0.0, rmin=1.0)

    assert np.allclose(R ** 2 + Z ** 2, np.ones(length))


def test_flux_surface_elongation():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=10.0, delta=0.0, Rcen=0.0, rmin=1.0)

    assert np.isclose(np.min(R), -1.0)
    assert np.isclose(np.max(R), 1.0)
    assert np.isclose(np.min(Z), -10.0)
    assert np.isclose(np.max(Z), 10.0)


def test_flux_surface_triangularity():
    length = 257
    theta = np.linspace(-np.pi, np.pi, length)

    R, Z = flux_surface(theta=theta, kappa=1.0, delta=1.0, Rcen=0.0, rmin=1.0)

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

    R, Z = flux_surface(theta=theta, kappa=2.0, delta=0.5, Rcen=1.0, rmin=2.0)

    assert np.isclose(R[0], -1.0)
    assert np.isclose(Z[0], 0.0)
    assert np.isclose(R[length // 4], 0.0)
    assert np.isclose(Z[length // 4], -4.0)
    assert np.isclose(R[length // 2], 3.0)
    assert np.isclose(Z[length // 2], 0.0)
    assert np.isclose(R[length * 3 // 4], 0.0)
    assert np.isclose(Z[length * 3 // 4], 4.0)


def test_default_bunit_over_b0():
    miller = Miller()
    assert np.isclose(miller.get_bunit_over_b0(), 1.0481789952353437)


@pytest.mark.parametrize(
    ["parameters", "expected"],
    [
        ((1.0, 0.0, 0.0, 0.0, 0.0), lambda theta: np.ones(theta.shape)),
        ((1, 0.0, 1.0, 0.0, 0.0), lambda theta: 1.0 / (np.sin(theta) ** 2 + 1)),
        (
            (2.0, 0.5, 0.5, 0.2, 0.1),
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
    miller = Miller()
    length = 65
    theta = np.linspace(-np.pi, np.pi, length)
    miller.kappa = parameters[0]
    miller.delta = parameters[1]
    assert np.allclose(
        miller.get_grad_r(parameters[2:], theta),
        expected(theta),
    )


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
def test_grad_r_free_function(parameters, expected):
    length = 65
    theta = np.linspace(-np.pi, np.pi, length)
    assert np.allclose(
        grad_r(**parameters, theta=theta),
        expected(theta),
    )
