"""Quasilinear flux weights of the hKBM solver (pyrokinetics.hkbm_solver.quasilinear).

Fast tests: properties of the weights of one STEP root (k_y rho_s = 0.3; one solve, ~10 s):
sign, exact ambipolarity of the phi and dB_par particle fluxes (quasineutrality and pressure
balance hold in the Galerkin sense, and phi, dB_par lie in the test space), approximate
ambipolarity of the A_par channel, invariance under the eigenvector's normalisation, and
agreement with GENE's quasilinear ratios for the same deck (nrg fluxes over the Jacobian-
weighted <|phi|^2> of the linear run spec_fB1_ky0.3: Q_i/<|phi|^2> = 2.681, Q_e/Q_i = 0.741,
EM fractions 0.325 (ions) and 0.475 (electrons), Gamma/Q_i = 0.208).

Slow tests: run_linear on a Pyro object (toroidal n, theta0 symmetry) and a regression of
fluxes() on the STEP deck."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics.hkbm_solver import quasilinear as QL
from pyrokinetics.hkbm_solver.gene_io import Deck

DECK = Path(__file__).parent / "data" / "step_ky0.2" / "parameters"

GENE_KY03 = dict(Qi=2.681, QeQi=0.741, Qi_em=0.325, Qe_em=0.475, G_Qi=0.208)


@pytest.fixture(scope="module")
def root():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = Deck(DECK)
        r = deck.solve(0.3)
    assert r["converged"]
    return r


@pytest.fixture(scope="module")
def w(root):
    return QL.weights(root["solver"], root)


def test_weights_sign_and_ambipolarity(w):
    for k in ("Q_i", "Q_e", "Gamma_i"):
        assert w[k]["total"] > 0
    assert w["ambipolarity"]["phi"] < 1e-8
    assert w["ambipolarity"]["bpar"] < 1e-8
    assert w["ambipolarity"]["apar"] < 0.6  # weak (projected) vorticity equation
    for s in ("Q_i", "Q_e", "Gamma_i", "Gamma_e"):
        d = w[s]
        assert np.isclose(d["em"], d["apar"] + d["bpar"])
        assert np.isclose(d["total"], d["es"] + d["em"])


def test_weights_normalisation_invariant(root, w):
    f = 0.37 * np.exp(1.1j)
    r2 = dict(root)
    for k in ("phi", "bpar", "psi"):
        r2[k] = root[k] * f
    r2["coef"] = {k: c * f for k, c in root["coef"].items()}
    w2 = QL.weights(root["solver"], r2)
    for s in ("Q_i", "Q_e", "Gamma_i"):
        assert np.isclose(w2[s]["total"], w[s]["total"], rtol=1e-10)
    assert np.isclose(w2["phi2"], w["phi2"] * abs(f) ** 2)


def test_weights_against_gene(w):
    qi, qe = w["Q_i"]["total"], w["Q_e"]["total"]
    assert qi == pytest.approx(GENE_KY03["Qi"], rel=0.2)
    assert qe / qi == pytest.approx(GENE_KY03["QeQi"], rel=0.15)
    assert w["Q_i"]["em"] / qi == pytest.approx(GENE_KY03["Qi_em"], rel=0.25)
    assert w["Q_e"]["em"] / qe == pytest.approx(GENE_KY03["Qe_em"], rel=0.25)
    assert w["Gamma_i"]["total"] / qi == pytest.approx(GENE_KY03["G_Qi"], rel=0.15)


def test_saturate_mixing_length():
    m = dict(
        converged=True,
        ky_rho_s=0.2,
        gamma_solver=0.1,
        weights_solver=dict(
            kperp2=0.5,
            **{
                s: dict(phi=1.0, apar=0.0, bpar=0.5, es=1.0, em=0.5, total=1.5)
                for s in ("Q_i", "Q_e", "Gamma_i", "Gamma_e")
            },
        ),
    )
    m2 = dict(m, ky_rho_s=0.4)
    m3 = dict(m, ky_rho_s=0.6, converged=False, weights_solver=None)
    out = QL.saturate([m, m2, m3], C=2.0)
    amp = 2.0 * (0.1 / 0.5) ** 2
    assert out["Q_i_total"] == pytest.approx(1.5 * amp * 0.2 + 0.5 * 1.5 * amp * 0.2)


@pytest.mark.slow
def test_run_linear_pyro_theta0():
    from pyrokinetics import Pyro

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyro = Pyro(gk_file=DECK)
        lg = pyro.local_geometry
        # a Pyro read from a GENE deck with dpdx_pm = -1 has beta_prime = 0: set it
        lg.beta_prime = -0.4985523 * lg.beta_prime.units
        ms = QL.run_linear(pyro, n=[40], theta0=[0.0, 0.1, -0.1], rho_star=0.0026)
    m0, mp_, mm = ms
    assert m0["hkbm_like"] and m0["ky"] == pytest.approx(
        40 * 0.0026 / 0.5438691, rel=1e-4
    )
    assert m0["gamma"] == pytest.approx(0.0654, rel=0.05)
    assert mp_["gamma"] == pytest.approx(mm["gamma"], rel=1e-3)
    assert mp_["omega"] == pytest.approx(mm["omega"], rel=1e-3)
    for k in ("phi", "apar", "bpar", "kperp2", "jacobian", "theta"):
        assert m0[k].shape == m0["theta"].shape
    assert np.isclose(np.abs(m0["phi"]).max(), 1.0)


@pytest.mark.slow
def test_fluxes_step_regression():
    r = QL.fluxes(DECK, ky=[0.1, 0.2, 0.3, 0.4], C=1.0)
    assert r["converged"].all() and r["hkbm_like"].all()
    # values of the first version (6 Oct 2026); C = 1
    assert r["Q_i"] == pytest.approx(0.7533, rel=0.02)
    assert r["Q_e"] == pytest.approx(0.6654, rel=0.02)
    assert r["Gamma"] == pytest.approx(0.1758, rel=0.02)
