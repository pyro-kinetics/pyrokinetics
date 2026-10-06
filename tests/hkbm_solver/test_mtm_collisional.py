"""Independent checks of the experimental collisional C&S-shaped response.

These establish numerical identities, not agreement with STEP MTM growth.
"""

import warnings
from pathlib import Path

import f90nml
import numpy as np
import pytest
from numpy.polynomial.legendre import legvander
from scipy.sparse import csr_matrix, eye, kron

from pyrokinetics.hkbm_solver.gene_io import Deck
from pyrokinetics.hkbm_solver.mtm_collisional import (
    CollisionalMTMSolver,
    pitch_operators,
    theta_derivatives,
)
from pyrokinetics.hkbm_solver.solver import PARAMS, step_geo


@pytest.mark.parametrize("nxi", [8, 16, 32])
def test_lorentz_legendre_conservation_dissipation(nxi):
    xi, w, derivative, C = pitch_operators(nxi)
    ell = np.arange(nxi)
    V = legvander(xi, nxi - 1)
    np.testing.assert_allclose(C @ V, V * (-ell * (ell + 1) / 2), atol=2e-10)
    np.testing.assert_allclose(w @ C, 0.0, atol=2e-11)
    np.testing.assert_allclose(w[:, None] * C, C.T * w, atol=2e-11)
    symmetric = np.sqrt(w[:, None] / w) * C
    assert np.linalg.eigvalsh(symmetric).max() < 1e-10
    np.testing.assert_allclose(derivative @ xi**3, 3 * xi**2, atol=1e-11)


def test_upwind_derivative_and_reflection():
    x = np.linspace(-2, 2, 17)
    positive, negative = theta_derivatives(x.size, x[1] - x[0])
    np.testing.assert_allclose((positive @ x**2)[2:], 2 * x[2:], atol=1e-13)
    np.testing.assert_allclose((negative @ x**2)[:-2], 2 * x[:-2], atol=1e-13)
    np.testing.assert_allclose(positive.toarray(), -negative.toarray()[::-1, ::-1])
    positive, negative = theta_derivatives(x.size, x[1] - x[0], order=3)
    np.testing.assert_allclose((positive @ x**3)[2:-1], 3 * x[2:-1] ** 2, atol=1e-13)
    np.testing.assert_allclose((negative @ x**3)[1:-2], 3 * x[1:-2] ** 2, atol=1e-13)
    k = np.linspace(0, np.pi, 100)
    symbol = np.exp(-2j * k) / 6 - np.exp(-1j * k) + 0.5 + np.exp(1j * k) / 3
    assert symbol.real.min() >= -1e-14  # no negative numerical dissipation


@pytest.mark.parametrize("coll", [0.0, 1e-4, 0.01])
def test_uniform_plasma_density_and_inductive_conductivity(coll):
    """No streaming/drift: g_l=Q/(omega+i*nu*l(l+1)/2) chi_l.

    Density (l=0) is NOT Krook damped, but parallel flow (l=1) is. The
    energy-integrated inductive current is an independent analytic formula.
    """
    s = CollisionalMTMSolver(
        step_geo(),
        PARAMS,
        0.3,
        coll=coll,
        npt=8,
        nturns=0,
        nE=8,
        nxi=8,
        collision_flr=False,
    )
    _, _, _, C = pitch_operators(s.nxi)
    s.bases = [-nu * kron(eye(s.N), csr_matrix(C)).tocsc() for nu in s.nu]
    s.A[:] = 1.0
    s.j0 = [np.ones((s.N, s.nxi)) for _ in s.E]
    omega = 0.5 + 0.1j
    phi = np.linspace(-0.3, 0.3, s.N).astype(complex)
    s.factor(omega)
    density, current, error, _ = s.response(phi, 1.0, kinetic_residual=True)
    Q = omega - np.array(s.sources)
    expect_n = np.sum(s.wE * Q / omega) * phi
    expect_j = -np.sum(s.wE * Q / (omega + 1j * s.nu) * (s.vte**2 * s.E / 3))
    np.testing.assert_allclose(density[1:-1], expect_n[1:-1], atol=2e-11)
    np.testing.assert_allclose(current[1:-1], expect_j, rtol=2e-12)
    assert error < 1e-11


def test_no_collision_operator_at_zero_rate_and_correct_flr_term():
    kw = dict(npt=8, nturns=0, nE=4, nxi=8)
    a = CollisionalMTMSolver(step_geo(), PARAMS, 0.3, coll=0.0, **kw)
    b = CollisionalMTMSolver(
        step_geo(), PARAMS, 0.3, coll=0.0, collision_flr=False, **kw
    )
    for first, second in zip(a.bases, b.bases):
        np.testing.assert_array_equal(first.toarray(), second.toarray())
    c = CollisionalMTMSolver(step_geo(), PARAMS, 0.3, coll=1e-4, **kw)
    for E, nu, initial, collided in zip(c.E, c.nu, a.bases, c.bases):
        diff = collided - initial + nu * c.lorentz
        a02 = 2 * c.me * E * c.kperp2 / c.B**2
        expected = (nu * a02[:, None] * (1 + c.xi**2) / 4).ravel()
        np.testing.assert_allclose(diff.diagonal(), expected, atol=1e-11)
        assert expected.min() >= 0


def test_mirror_streams_along_constant_energy_mu_characteristics():
    """lambda=(1-xi^2)/B is invariant along S; residual falls with theta step."""
    errors = []
    for npt in (16, 32, 64):
        s = CollisionalMTMSolver(
            step_geo(), PARAMS, 0.3, npt=npt, nturns=0, nE=2, nxi=8
        )
        invariant = (1 - s.xi**2) / s.B[:, None]
        residual = (s.stream @ invariant.ravel()).reshape(s.N, s.nxi)
        errors.append(np.linalg.norm(residual[2:-2]) / np.sqrt(residual[2:-2].size))
    assert errors[2] < 0.4 * errors[1] < 0.16 * errors[0]


def test_response_tearing_parity_and_inner_residual():
    s = CollisionalMTMSolver(
        step_geo(), PARAMS, 0.3, coll=1e-4, npt=16, nturns=1, nE=6, nxi=12
    )
    value = s.evaluate(0.5 + 0.1j)
    assert np.isfinite(value)
    assert s.last["qn_residual"] < 1e-7
    assert s.last["kinetic_residual"] < 1e-10
    np.testing.assert_allclose(s.last["phi"], -s.last["phi"][::-1], atol=1e-12)
    np.testing.assert_allclose(
        s.last["current"], s.last["current"][::-1], rtol=1e-7, atol=1e-8
    )


def test_even_apar_basis_orthogonality_and_eliminated_projections():
    s = CollisionalMTMSolver(
        step_geo(),
        PARAMS,
        0.3,
        coll=1e-4,
        npt=32,
        nturns=1,
        nE=6,
        nxi=12,
        nA=4,
        theta_order=3,
    )
    gram = (s.A_basis * (s.wtheta * s.kperp2)) @ s.A_basis.T / s.ampere_norm
    np.testing.assert_allclose(gram, np.eye(s.nA), atol=1e-12)
    np.testing.assert_allclose(s.A_basis, s.A_basis[:, ::-1], atol=1e-12)
    value = s.evaluate(0.5 + 0.1j)
    residual = s.kperp2 * s.last["apar"] - s.beta / 2 * s.last["current"]
    projections = (s.A_basis * s.wtheta) @ residual / s.ampere_norm
    np.testing.assert_allclose(projections[1:], 0.0, atol=1e-7)
    assert projections[0] == pytest.approx(value, abs=1e-7)
    assert s.last["kinetic_residual"] < 1e-10


@pytest.mark.parametrize(
    "kw",
    [{"coll": -1}, {"nxi": 7}, {"npt": 9}, {"nE": 1}, {"frequency_model": "krook"}],
)
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        CollisionalMTMSolver(step_geo(), PARAMS, 0.3, **kw)


def test_no_landau_continuation_claim():
    s = CollisionalMTMSolver(step_geo(), PARAMS, 0.3, npt=8, nturns=0, nE=2, nxi=8)
    for w in (0.5, 0.5 - 0.01j, complex(float("nan"), 0.1)):
        with pytest.raises(ValueError):
            s.factor(w)


def test_step_collisional_root_is_labelled_experimental(tmp_path):
    """Numerical regression only: this SHORT-domain root is NOT a GENE benchmark.

    It must expose both its algebraic convergence and its failed domain check.
    The collisional root is obtained with the physical drift sign, not a flip.
    """
    nml = f90nml.read(Path(__file__).parent / "data/step_ky0.2/parameters")
    nml["general"]["beta"] = 0.14
    nml["box"]["kymin"] = 0.2848826
    path = tmp_path / "parameters"
    nml.write(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = Deck(path)
    with pytest.warns(UserWarning, match="Experimental"):
        s = CollisionalMTMSolver.from_deck(
            deck, 0.2848826, npt=16, nturns=4, nE=8, nxi=12
        )
    r = s.solve(0.52 + 0.01j)
    assert r["status"] == "growing_root"
    assert r["gamma_deck"] == pytest.approx(0.005973737657, rel=1e-5)
    assert r["omega_gene_deck"] == pytest.approx(-0.5210463074, rel=1e-5)
    assert r["relative_residual"] < 1e-7
    assert r["kinetic_residual"] < 1e-10
    assert r["edge_phi"] > 0.1
    assert r["validated"] is False
    assert r["collision_physics_match"] is False
    assert r["drift_sign"] == 1
