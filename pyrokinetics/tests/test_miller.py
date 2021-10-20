from pyrokinetics.miller import Miller

import numpy as np


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


def test_default_bunit_over_b0():
    miller = Miller()
    assert np.isclose(miller.get_bunit_over_b0(), 1.0481789952353437)
