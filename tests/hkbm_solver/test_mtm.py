"""Tests of the microtearing branch (mtm.py, Chandran & Schekochihin 2024) against the limits and
numbers of the paper, on the Patel et al. (2022) Table 2 surface (fixture deck) and the STEP deck.
"""

from pathlib import Path

import numpy as np
import pytest

from pyrokinetics.hkbm_solver import mtm
from pyrokinetics.hkbm_solver import solver as S
from pyrokinetics.hkbm_solver.gene_io import Deck

DATA = Path(__file__).parent / "data"
PATEL = DATA / "patel2022" / "parameters"
FAST = dict(nE=12, nlam=12, nlam_tr=8, npt=16)


@pytest.fixture(scope="module")
def patel():
    with pytest.warns(UserWarning):
        return Deck(PATEL)


def _kfac(d):
    g = d.geo
    i0 = int(np.argmin(abs(g.z)))
    return np.sqrt(g.gyy[i0]) * np.sqrt(2 * d.params["me"]) / g.B[i0]


def test_patel_geometry(patel):
    """C&S section 3: beta_e averaged around the flux-surface contour 0.125 (ours 0.119 with
    Patel's Table 2 Miller parameters), |B| between ~2.0 and 3.4 T (C&S Fig. 2; B0 = 2.16 T), and
    omega_0 = 0.0046 v_Te/a at k_wedge rho_e|theta=0 = 0.0022 (C&S Fig. 3, n = 25)."""
    g = patel.geo
    R, Z = np.asarray(patel.miller["R"]), np.asarray(patel.miller["Z"])
    dl = np.hypot(np.roll(R, -1) - R, np.roll(Z, -1) - Z)
    bav = np.sum(0.15 / g.B**2 * dl) / dl.sum()
    assert abs(bav - 0.125) < 0.01
    assert 1.9 < 2.16 * g.Bmin < 2.1 and 3.3 < 2.16 * g.Bmax < 3.6
    ky = 0.0022 / _kfac(patel)
    vte = np.sqrt(2 / patel.params["me"])
    M = mtm.MTMSolver(g, patel.params, ky, **FAST)
    assert M.omega0 / vte == pytest.approx(0.0046, rel=0.03)


def test_turn_extension_continuous():
    """Turn j of the extended ballooning angle: k_perp^2 and dK_y/dtheta continuous at the turn
    boundaries (the sign of the theta0 shift)."""
    M = mtm.MTMSolver(S.step_geo(), S.PARAMS, 0.3, nturns=3, trapped=False, **FAST)
    f = M.fine
    nz = M.geo.z.size
    dz = M.dz
    for b in range(1, 6):
        i = b * nz  # first point of the next turn
        gl = 2 * f["gyy"][i - 1] - f["gyy"][i - 2]
        assert f["gyy"][i] == pytest.approx(gl, rel=0.02)
        dl = (3 * f["Ky"][i - 1] - 4 * f["Ky"][i - 2] + f["Ky"][i - 3]) / (2 * dz)
        dr = (-3 * f["Ky"][i] + 4 * f["Ky"][i + 1] - f["Ky"][i + 2]) / (2 * dz)
        assert dr == pytest.approx(dl, rel=0.02, abs=0.02)


def test_phi_term_off_is_damped(patel):
    """C&S section 4: without the dPhi term, gamma = -sqrt(pi) v_Te/L < 0 and omega_r = omega_0."""
    ky = 0.0088 / _kfac(patel)
    M = mtm.MTMSolver(patel.geo, patel.params, ky, phi_term=False, **FAST)
    r = M.solve(im_floor=-1.0)
    assert r["converged"]
    assert r["gamma"] == pytest.approx(-np.sqrt(np.pi) * M.vte_over_L, rel=1e-6)
    assert r["omega"].real == pytest.approx(M.omega0, rel=1e-6)


def test_eta_e_zero(patel):
    """C&S section 2.10: eta_e = 0 -> Gamma = W = dPhi = 0 at omega = omega_*e, so
    D(omega_*e) = -i sqrt(pi) v_Te/L (omega = omega_*e for k rho_e << beta_e)."""
    p = dict(patel.params, omte=0.0)
    ky = 0.0088 / _kfac(patel)
    M = mtm.MTMSolver(patel.geo, p, ky, **FAST)
    w = M.ky * p["omn"]
    D = M.D(complex(w, 1e-9))
    assert abs(D.real) < 1e-10 * w
    assert D.imag == pytest.approx(np.sqrt(np.pi) * M.vte_over_L, rel=1e-6)


def test_cold_ion_limit(patel):
    """C&S (2.55-2.57): for tau << 1, dPhi = tau Gamma, so at omega = omega_0 the dPhi term of
    (2.39) equals tau int J Gamma^2 (the closed form); and Re D(omega_0) = 0 to O(tau) (omega_r =
    omega_0)."""
    ky = 0.0088 / _kfac(patel)
    p = dict(patel.params, Ti=0.01)
    M = mtm.MTMSolver(patel.geo, p, ky, trapped=True, **FAST)
    w0 = complex(M.omega0)
    D = M.D(w0)
    pref = 1j * np.sqrt(np.pi) * w0**2 * M.Bmax / (2 * M.vte)
    S_full = (D - (w0 - M.omega0) - 1j * np.sqrt(np.pi) * M.vte_over_L) / pref
    S_closed = p["Ti"] * np.sum(M.c["J"] * M.Gamma**2) * M.dth
    assert abs(S_full / S_closed - 1) < 0.03
    wc = M.cold_ion()
    assert wc.imag == pytest.approx(
        -np.sqrt(np.pi) * M.vte_over_L
        + (pref * S_closed).real * 0
        - np.sqrt(np.pi)
        * p["Ti"]
        * M.omega0**2
        * M.Bmax
        / (2 * M.vte)
        * np.sum(M.c["J"] * (M.Gamma**2).real)
        * M.dth,
        rel=1e-9,
    )


def test_ordering_threshold():
    """C&S (2.53): the current-limiting term v_Te/L grows as (k rho_e)^2/beta_e."""
    p = dict(S.PARAMS, beta=0.12)
    a = mtm.MTMSolver(S.step_geo(), p, 0.2, trapped=False, nturns=8, **FAST)
    b = mtm.MTMSolver(S.step_geo(), p, 0.4, trapped=False, nturns=8, **FAST)
    c = mtm.MTMSolver(
        S.step_geo(), dict(p, beta=0.24), 0.4, trapped=False, nturns=8, **FAST
    )
    assert b.vte_over_L / a.vte_over_L == pytest.approx(4.0, rel=1e-6)
    assert b.vte_over_L / c.vte_over_L == pytest.approx(2.0, rel=1e-6)


def test_run_linear_mtm_branch():
    """run_linear(modes=('mtm',)) on a GENE deck: records tagged branch/parity, no hKBM solve, a
    quick verdict at a STEP point where the collisionless C&S root is damped."""
    from pyrokinetics.hkbm_solver.quasilinear import run_linear

    ms = run_linear(
        DATA / "step_ky0.2" / "parameters" if (DATA / "step_ky0.2").exists() else PATEL,
        ky=[0.3],
        modes=("mtm",),
        mtm_kw=FAST,
        timeout=20,
    )
    assert len(ms) == 1
    m = ms[0]
    assert m["branch"] == "mtm" and m["parity"] == "tearing"
    assert m["seconds"] < 20


@pytest.mark.parametrize(
    "n, paper", [(25, (0.0068, 0.0012)), (100, (0.026, 0.0046)), (400, (0.088, 0.0150))]
)
def test_fig3_reversed_drift(patel, n, paper):
    """C&S Fig. 3 (omega_r, gamma in v_Te/a, read off the figure) is reproduced only with the
    electron magnetic drift reversed relative to GENE's convention (drift_sign = -1): omega_r
    within 10 %, gamma within 40 %.  Removing the pressure term from the drift (dpdx_pm = 0),
    beta' altogether, or the radial (K_x) drift does not reproduce it (no growing root; see the
    mtm work folder, MTM_PROGRESS.md)."""
    vte = np.sqrt(2 / patel.params["me"])
    ky = 0.0022 * n / 25 / _kfac(patel)
    r = mtm.find_root(patel.geo, patel.params, ky, drift_sign=-1.0)
    assert r["growing"]
    assert r["omega"].real / vte == pytest.approx(paper[0], rel=0.1)
    assert r["gamma"] / vte == pytest.approx(paper[1], rel=0.4)


def test_fig3_collapse_reversed_drift(patel):
    """C&S Fig. 3: gamma drops sharply between n = 800 and 1060 (k rho_e -> beta_e, (2.53));
    paper 0.0125 -> 0.0022 v_Te/a, reversed-drift solver 0.0145 -> 0.005."""
    g = []
    for n in (800, 1060):
        ky = 0.0022 * n / 25 / _kfac(patel)
        g.append(mtm.find_root(patel.geo, patel.params, ky, drift_sign=-1.0)["gamma"])
    assert g[1] < 0.5 * g[0]


def test_fig3_gene_sign_damped(patel):
    """With GENE's drift sign the collisionless C&S MTM on the Patel surface is damped (no
    growing root; the low-resolution verdict), omega_r within 10 % of omega_0."""
    ky = 0.0088 / _kfac(patel)
    r = mtm.find_root(patel.geo, patel.params, ky)
    assert r["no_root_verdict"] and not r["growing"]
    assert r["omega"].real == pytest.approx(r["omega0"], rel=0.1)


def test_step_fig16_point_no_collisionless_mtm(tmp_path):
    """STEP, k_y 0.285, beta_e 0.14 (Kennedy et al. 2023 Fig. 16: GENE's tearing-parity mode,
    omega -0.518, gamma +0.0066 c_s/a at nominal collisionality).  With the physical drift sign the
    collisionless C&S branch has no growing root (fast verdict) and its frequency sits within 2 %
    of GENE's; the reversed drift would give gamma ~ 0.07 (10x GENE)."""
    import re

    t = (DATA / "step_ky0.2" / "parameters").read_text()
    t = re.sub(r"^(\s*)kymin\s*=.*$", r"\1kymin = 0.2848826", t, count=1, flags=re.M)
    t = re.sub(r"^(\s*)beta\s*=.*$", r"\1beta = 0.14", t, count=1, flags=re.M)
    (tmp_path / "parameters").write_text(t)
    with pytest.warns(UserWarning):
        d = Deck(tmp_path / "parameters")
    r = mtm.solve_deck(d, 0.2848826)
    assert r["no_root_verdict"] and not r["growing"]
    assert r["omega"] == pytest.approx(-0.518, rel=0.02)
    assert r["seconds"] < 10
