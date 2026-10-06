"""Independent checks of the optional legacy reduced collision projection."""

from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyrokinetics.hkbm_solver.exact import ExactSolver
from pyrokinetics.hkbm_solver.exact_collisions import LegacyCollisions, periodic_probes
from pyrokinetics.hkbm_solver.gene_io import Deck


def test_periodic_probes_against_ode():
    """Check mean response and constant-drive moments without analytic moments."""
    x = np.array([-0.2 + 0.4j, -0.4 - 0.1j, -0.15 + 0.2j, -0.3 + 0.6j])
    dt = np.array([0.4, 0.8, 0.3, 0.6])
    row = np.array([0.8, 1.1, 0.7, 1.3])
    col = np.array([0.2 - 0.5j, -0.1 - 0.3j, 0.3 - 0.4j, 0.1 - 0.6j])
    nodes_a, nodes_b = np.array([0, 1, 2, 1]), np.array([1, 2, 1, 0])
    W = np.zeros((2, 4, 3))
    W[0, np.arange(4), nodes_a] = 1
    W[1, np.arange(4), nodes_b] = 1
    h, U, V = periodic_probes(
        x[None, None], dt[None, None], [row[None, None]], [col[None, None]], [W]
    )
    vector = np.array([0.2 + 0.3j, -0.5 + 0.1j, 0.7 - 0.2j])

    def response(unit):
        def sweep(incoming):
            state = np.r_[incoming, np.zeros(4, complex)]
            for j in range(4):

                def rhs(t, y):
                    test = (1 - t) * W[0, j] + t * W[1, j]
                    source = -1j if unit else col[j] * (test @ vector)
                    return np.r_[
                        x[j] * y[0] + dt[j] * source,
                        dt[j] / dt.sum() * y[0],
                        row[j] * test * y[0],
                    ]

                state = solve_ivp(rhs, (0, 1), state, rtol=1e-11, atol=1e-13).y[:, -1]
            return state

        incoming = sweep(0j)[0] / (1 - np.exp(x.sum()))
        return sweep(incoming)

    driven, unit = response(False), response(True)
    np.testing.assert_allclose(U[0][0, 0] @ vector, driven[1], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(h[0, 0], unit[1], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(V[0][0, 0], unit[2:], rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("nu", [0.0, 1e-12, 0.04, 100.0])
def test_projected_lorentz_against_direct_coupled_system(nu):
    """Low-rank correction equals a direct pitch-coupled solve, including psi."""
    operator = LegacyCollisions.__new__(LegacyCollisions)
    operator.lo = np.array([2.0, 1.0])
    operator.up = np.array([1.0, 0.0])
    operator.diag = operator.lo + operator.up
    operator.nu = np.array([nu])
    L = (
        np.diag(operator.diag)
        - np.diag(operator.lo[1:], -1)
        - np.diag(operator.up[:-1], 1)
    )
    frequency = -0.3 + 0.12j
    d = frequency - np.array([-0.1, 0.04])
    source = np.array([[0.2 + 0.1j, -0.7j], [0.4, 0.2j]])
    Q = np.array([[0.1j, 0.3], [0.2 - 0.1j, -0.1]])
    moment = np.array([[0.7, 0.2], [-0.4, 0.6]])
    U = source / d[:, None]
    V = moment / d[None, :]
    operator.batches = {
        0: dict(h=(1 / d)[None], U=(U - Q)[None], V=V[None], present=np.ones(2, bool))
    }
    actual = moment @ U
    operator.apply(actual)
    expected = moment @ np.linalg.solve(
        np.diag(d) + 1j * nu * L, source + 1j * nu * L @ Q
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def deck():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Deck(Path(__file__).parent / "data/step_ky0.2/parameters")


def test_zero_collision_limit_preserves_full_matrix():
    d = deck()
    options = dict(npt=8, nturns=0, nE=3, nlp=3, nlt=2, nq=4, nbs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = ExactSolver.from_deck(d, 0.3, **options)
        legacy = ExactSolver.from_deck(
            d, 0.3, collision_model="legacy", coll=0, **options
        )
        # Split pitch cells very near the passing boundary can have large
        # diffusion coefficients: choose a rate small relative to that operator.
        tiny = ExactSolver.from_deck(
            d, 0.3, collision_model="legacy", coll=1e-18, **options
        )
    omega = -0.2 + 0.1j
    base = free.matrix(omega, "twisting")[0]
    np.testing.assert_array_equal(legacy.matrix(omega, "twisting")[0], base)
    np.testing.assert_allclose(
        tiny.matrix(omega, "twisting")[0], base, rtol=1e-8, atol=1e-9
    )


def test_legacy_operator_metadata_and_parity_guard():
    with pytest.warns(UserWarning, match="Legacy reduced collisions"):
        solver = ExactSolver.from_deck(
            deck(),
            0.3,
            collision_model="legacy",
            npt=8,
            nturns=0,
            nE=3,
            nlp=3,
            nlt=2,
            nq=4,
            nbs=2,
        )
    assert solver.coll > 0
    assert solver.legacy.kappa0 > 0
    assert np.all(solver.legacy.nu > 0)
    with pytest.raises(ValueError, match="twisting parity only"):
        solver.matrix(-0.2 + 0.1j, "tearing")
    matrix, _ = solver.matrix(-0.2 + 0.1j, "twisting")
    assert np.all(np.isfinite(matrix))


def test_nonzero_coll_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="requires collision_model"):
        ExactSolver(None, [], 0.3, 0.09, coll=0.001)
