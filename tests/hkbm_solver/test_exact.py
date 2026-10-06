"""Independent checks of the full collisionless GK discretisation."""

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from pyrokinetics.hkbm_solver.exact import ExactSolver, Species, _phis


@pytest.mark.parametrize("x", [0j, 1e-10j, -0.2 + 0.1j, -0.6 + 2j, -20 + 8j])
def test_exponential_moments(x):
    """Integrate the exponential independently of the recurrence used by the solver."""
    for k, actual in enumerate(_phis(np.array([x])), start=1):

        def integrand(t):
            return np.exp(x * (1 - t)) * t ** (k - 1) / math.factorial(k - 1)

        expected = (
            quad(lambda t: integrand(t).real, 0, 1, epsabs=1e-13)[0]
            + 1j * quad(lambda t: integrand(t).imag, 0, 1, epsabs=1e-13)[0]
        )
        assert actual[0] == pytest.approx(expected, rel=2e-12, abs=2e-14)


def test_exponential_moments_large_damping():
    """Very slow particles must not overflow a series evaluated outside its small-x branch."""
    x = np.array([-1e100 + 2e99j])
    with np.errstate(over="raise", invalid="raise"):
        values = _phis(x)
    for k, value in enumerate(values, start=1):
        assert value[0] == pytest.approx(
            -1 / (x[0] * math.factorial(k - 1)), rel=1e-14, abs=0
        )


def test_boundary_stagnation_is_not_a_root():
    """The growing-half-plane search must not call a damped root a growing mode."""
    solver = ExactSolver.__new__(ExactSolver)

    def dispersion(omega, parity):
        value = omega - (0.2 - 0.1j)
        solver._last_residual = abs(value)
        return value

    solver.D = dispersion
    solver.result = lambda omega, parity, conv, it, sec: dict(
        omega=omega, converged=conv
    )
    result = solver.find_root(0.2 + 0.01j, maxit=60)
    assert not result["converged"]


@pytest.mark.parametrize("omega", [1.0, 1 - 0.1j, np.nan + 1j])
def test_incoming_orbit_domain(omega):
    solver = ExactSolver.__new__(ExactSolver)
    with pytest.raises(ValueError, match=r"Im\(omega\) > 0"):
        solver.D(omega)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("block", [1, 3])
def test_passing_kernel_against_initial_value_problem(reverse, block):
    """The assembled Galerkin response agrees with direct integration along an orbit.

    This checks node/cell interpolation, both streaming directions, the local
    response and propagation between cells without using the analytic moments.
    """
    solver = ExactSolver.__new__(ExactSolver)
    solver.fields = ("phi", "apar")
    solver.ftype = {"phi": "node", "apar": "cell"}
    solver.off = {"phi": 0, "apar": 4}
    solver.nunk = 7
    solver.block = block
    cells = np.arange(3)[::-1] if reverse else np.arange(3)
    na, nb = (cells + 1, cells) if reverse else (cells, cells + 1)
    x = np.array(
        [
            [-0.2 + 0.5j, -1.1 + 0.1j, -0.4 - 0.3j],
            [-0.3 + 0.2j, -0.1 - 0.6j, -2.0 + 0.4j],
        ]
    )
    dt = np.array([[0.3, 0.7, 0.2], [0.5, 0.4, 0.6]])
    rows = [
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]]),
        np.array([[0.4, -0.2, 0.8], [-0.6, 0.1, 0.3]]),
    ]
    cols = [
        np.array([[0.4j, 0.2, -0.7j], [0.6, 0.5j, 0.3]]),
        np.array([[0.2, -0.5j, 0.9], [0.4j, 0.7, -0.3j]]),
    ]
    vector = np.array([0.4, 0.2j, -0.3, 0.7j, 0.8, -0.4j, -0.2])
    matrix = np.zeros((7, 7), complex)
    solver._path_kernel(matrix, x, dt, rows, cols, cells, na, nb, np.arange(7))
    expected = np.zeros(7, complex)
    for v in range(2):
        incoming = 0j
        for m, cell in enumerate(cells):

            def rhs(t, state):
                hats = np.array([1 - t, t])
                phi = hats @ vector[[na[m], nb[m]]]
                source = cols[0][v, m] * phi + cols[1][v, m] * vector[4 + cell]
                return np.array(
                    [
                        x[v, m] * state[0] + dt[v, m] * source,
                        rows[0][v, m] * hats[0] * state[0],
                        rows[0][v, m] * hats[1] * state[0],
                        rows[1][v, m] * state[0],
                    ]
                )

            end = solve_ivp(
                rhs, (0, 1), [incoming, 0j, 0j, 0j], rtol=1e-11, atol=1e-13
            ).y[:, -1]
            incoming = end[0]
            expected[na[m]] += end[1]
            expected[nb[m]] += end[2]
            expected[4 + cell] += end[3]
    np.testing.assert_allclose(matrix @ vector, expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("growth", [0.2, 2000.0])
def test_trapped_kernel_against_periodic_initial_value_problem(growth):
    """Compare a complete bounce to an independently integrated periodic orbit.

    Both bounce points are inside cells, so the test also exercises the
    interpolation that is absent from the passing-orbit check.
    """
    solver = ExactSolver.__new__(ExactSolver)
    solver.fields, solver.ftype, solver.off = ("phi",), {"phi": "node"}, {"phi": 0}
    solver.E, solver.wE, solver.wlt = np.array([1.0]), np.array([0.7]), np.array([0.4])
    solver.ky, solver.beta, solver.drift_sign = 0.3, 0.09, 1.0
    tau, dK = np.array([[0.6, 0.8]]), np.array([[0.1, -0.2]])
    gyro = np.array([[[0.8, 0.6]]])
    solver.trp = [
        dict(
            k1=np.array([0]),
            Mc=2,
            il=0,
            ns=2,
            tau=tau,
            dK=dK,
            gyro=[(gyro, gyro, gyro)],
            segc=np.array([0, 1]),
            pcl=np.array([0, 1, 1]),
            alpha=np.array([0.25, 0.0, 0.7]),
        )
    ]
    species = Species("electron", -1, 1, 1, 1, 1.2, 0.7)
    omega = 0.4 + growth * 1j
    matrix = np.zeros((3, 3), complex)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solver._add_trapped(matrix, species, 0, omega, np.arange(3))
    # Point-to-node interpolation at the two turning points and intervening edge.
    interpolation = np.array([[0.75, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.3, 0.7]])
    vector = np.array([0.4 + 0.2j, -0.8, 0.3j])
    segments = [(0, 0, 1), (1, 1, 2), (1, 2, 1), (0, 1, 0)]
    expected = np.zeros(3, complex)
    x = 1j * (omega * tau[0] + solver.ky * dK[0]) / np.sqrt(2)
    omega_star = solver.ky * (species.omn - species.omt / 2)
    coefficient = -1j * (omega - omega_star)

    def sweep(initial, moments):
        state = np.array([initial, 0j, 0j, 0j])
        for segment, a, b in segments:

            def rhs(t, y):
                hats = (1 - t) * interpolation[a] + t * interpolation[b]
                source = coefficient * gyro[0, 0, segment] * (hats @ vector)
                moment = (
                    -0.5
                    / np.sqrt(np.pi)
                    * solver.wE[0]
                    * solver.wlt[0]
                    * tau[0, segment]
                    * gyro[0, 0, segment]
                )
                return np.r_[
                    x[segment] * y[0] + tau[0, segment] / np.sqrt(2) * source,
                    moment * hats * y[0],
                ]

            state = solve_ivp(rhs, (0, 1), state, rtol=1e-11, atol=1e-13).y[:, -1]
        return state[1:] if moments else state[0]

    incoming = sweep(0j, False) / (1 - np.exp(2 * x.sum()))
    expected = sweep(incoming, True)
    np.testing.assert_allclose(matrix @ vector, expected, rtol=2e-10, atol=2e-12)


def test_benchmark_grid_and_resume(tmp_path):
    from pyrokinetics.hkbm_solver.exact_benchmark import (
        completed_records,
        read_reference,
    )

    table = tmp_path / "reference.csv"
    table.write_text(
        "beta,ky,growth_rate,mode_frequency,Ctear\n"
        "0.09,0.2848826,0.1,0.2,0.0\n"
        "0.14,0.2848826,0.0066,-0.518,0.9\n"
    )
    grid = read_reference(table)
    assert grid[1]["ky"] == 0.2848826
    assert grid[1]["mode_frequency"] == -0.518
    assert grid[1]["growth_rate_tolerance"] is None
    output = tmp_path / "results.jsonl"
    output.write_text(
        '{"reference":{"index":1},"parity":"tearing",' '"configuration":{"id":"old"}}\n'
    )
    assert completed_records(output, "old") == {(1, "tearing")}
    with pytest.raises(ValueError, match="different inputs"):
        completed_records(output, "new")


def test_field_weighted_wave_numbers_integrate_the_elements():
    """T3D's averages use integrals of the actual field basis, including endpoint cells."""
    from types import SimpleNamespace

    solver = ExactSolver.__new__(ExactSolver)
    solver.N, solver.npt, solver.nturns, solver.ky = 2, 2, 0, 0.3
    solver.edges, solver.theta = np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5])
    solver.eg = SimpleNamespace(
        at=lambda z: dict(J=1 + 0.2 * z, B=1 + z**2, gyy=1 + 2 * z**2)
    )
    solver._cells(6)
    phi, apar = np.array([1 + 0.2j, -0.5 + 0.3j, 0.4j]), np.array([0.5j, 1.0])
    solver.eigenvector = lambda: dict(phi=phi, apar=apar)
    solver._last_residual = 1e-12
    result = solver.result(0.2 + 0.1j, "twisting", True, 4, 1.0)

    def density(z):
        field = np.interp(z, solver.edges, phi)
        return (1 + 0.2 * z) * abs(field) ** 2

    norm = quad(density, -1, 1, points=[0], epsabs=1e-13)[0]
    moment = quad(
        lambda z: density(z) * 0.3**2 * (1 + 2 * z**2), -1, 1, points=[0], epsabs=1e-13
    )[0]
    assert result["kperp2_phi"] == pytest.approx(moment / norm, rel=1e-12)
    norm_a = sum(
        abs(a) ** 2 * quad(lambda z: 1 + 0.2 * z, left, right)[0]
        for a, left, right in zip(apar, solver.edges[:-1], solver.edges[1:])
    )
    moment_a = sum(
        abs(a) ** 2
        * quad(lambda z: (1 + 0.2 * z) * 0.3**2 * (1 + 2 * z**2), left, right)[0]
        for a, left, right in zip(apar, solver.edges[:-1], solver.edges[1:])
    )
    assert result["kperp2_apar"] == pytest.approx(moment_a / norm_a, rel=1e-12)


@pytest.mark.slow
def test_collisionless_step_root_and_field_refinement():
    """GENE NC_fB1_ky0.3: omega=0.1923, gamma=0.1366 (includes hyp_z=-1).

    Numerical convergence of our collisionless response is checked separately
    from the imperfectly matched GENE numerical operator.
    """
    from pathlib import Path

    from pyrokinetics.hkbm_solver.gene_io import Deck

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        deck = Deck(Path(__file__).parent / "data" / "step_ky0.2" / "parameters")
    deck.nml["general"]["coll"] = 0.0
    roots = []
    for npt in (32, 64):
        solver = ExactSolver.from_deck(
            deck, 0.3, npt=npt, nturns=1, nE=8, nlp=8, nlt=4, nq=6
        )
        root = solver.find_root(-0.19 + 0.14j, "twisting", timeout=120)
        assert root["converged"]
        assert root["relative_residual"] < 1e-7
        assert root["edge_phi"] < 0.01
        roots.append(root)
    assert roots[-1]["omega_gene"] == pytest.approx(0.1923, rel=0.04)
    assert roots[-1]["gamma"] == pytest.approx(0.1366, rel=0.15)
    assert roots[0]["gamma"] == pytest.approx(roots[1]["gamma"], rel=0.02)
