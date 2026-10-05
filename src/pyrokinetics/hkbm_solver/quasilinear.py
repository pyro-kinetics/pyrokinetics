"""
Quasilinear fluxes from the hKBM eigenmode, and a mixing-length saturation rule.

weights(solver, result)
    Particle and heat fluxes of ions and electrons per <|phi|^2>, split into the phi (E x B),
    A_par (flutter) and dB_par (grad-B drift of mu dB_par) channels, in GENE's definitions and
    normalisation (the columns Gamma_es, Gamma_em, Q_es, Q_em of GENE's nrg file: em = A_par +
    dB_par).  Derivation in PHYSICS.md ("Quasilinear fluxes").

fluxes(source, ky=None, ...)
    Solve the eigenproblem on a k_y grid (continuation from the previous k_y, one worker process
    per root with a timeout) and return the saturated fluxes Q_i, Q_e, Gamma in gyro-Bohm units
    of the deck (GENE normalisation), with the per-k_y growth rates, frequencies and weights.

Units inside: the solver's (T_ref = T_e, n_ref = n_e, m_ref = m_i, L_ref, B_ref; k_y in 1/rho_s,
frequencies in c_s/L_ref, fluxes in n_e T_e c_s rho_s^2/L_ref^2 and n_e c_s rho_s^2/L_ref^2).
"""

import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma

from . import kernels as Z

SQPI = np.sqrt(np.pi)
# <E^k> over a Maxwellian (E = v^2/v_T^2): Gamma(k + 3/2)/Gamma(3/2)
_M = np.array([Gamma(k + 1.5) / Gamma(1.5) for k in range(6)])
# <(v_perp/v_T)^2 E^k> = (2/3) <E^(k+1)>
_N = np.array([2.0 / 3.0 * _M[k + 1] for k in range(5)])

# Mixing-length calibration (fluxes(): rule='mixing_length'), fitted to the stella STEP
# nonlinear (q, beta_e) scan; see QL_FLUX.md in the analysis folder.  Total heat flux
# Q_i + Q_e = C_ML int dk_y [Q_i + Q_e]/<|phi|^2> (gamma/<k_perp^2>)^2, GENE gyro-Bohm units.
C_ML = 1.0
KY_DEFAULT = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)


def _periodic_spline(theta, f):
    tp = np.append(theta, np.pi)
    fr = CubicSpline(tp, np.append(f.real, f.real[0]), bc_type="periodic")
    fi = CubicSpline(tp, np.append(f.imag, f.imag[0]), bc_type="periodic")
    return fr, fi


def _electron_moments(Sv, omega, c):
    """Velocity moments of the electron non-adiabatic h_e on the solver's theta grid.

    c: basis coefficients (phi, bpar, psi).  Returns dict of arrays (k = 0, 1):
    n[k] = int d3v E^k h_e, m[k] = int d3v mu E^k h_e, d[k] = int d3v E^k omega_de h_e.
    """
    geo, sw = Sv.geo, Sv.sw
    ky, ae, be = Sv.ky, Sv.ae, Sv.be
    lam = Sv.lam
    cphi, cb, cpsi = c["phi"], c["bpar"], c["psi"]
    # bounce-averaged source of the trapped response (solver.matrix, src())
    C0 = cphi @ Sv.bar_phi - cpsi @ Sv.bar_psi
    C1 = -sw["vl_e"] * lam * (cb @ Sv.bar_b) - (ky / omega) * (cpsi @ Sv.bar_Dpsi)
    if Sv.coll > 0:
        H = Sv._eH(omega, C0[None], C1[None])[:, 0, :]  # (nE, nlam)
        K = [np.einsum("e,el->l", Sv.eW * Sv.eE**j, H) for j in range(4)]
    else:
        M = [Z.Mint(n, omega, ky * Sv.Omega) for n in range(6)]
        K = [Sv._K(M, j, C0, C1, omega - ae, be) for j in range(4)]
    psi = c["psi_theta"]
    a_, b_ = 1 - ae / omega, be / omega
    out = dict(n=[], m=[], d=[])
    for k in (0, 1):
        n_tr = -(Sv.G0 @ K[k]) / SQPI
        m_tr = -(Sv.Gl @ K[k + 1]) / SQPI
        d_tr = -(Sv.Gd @ K[k + 1]) / SQPI
        n_ps = -psi * (a_ * _M[k] - b_ * _M[k + 1])
        m_ps = -psi * (a_ * _N[k] - b_ * _N[k + 1]) / geo.B
        d_ps = (
            ky
            * (2.0 / 3.0)
            * (Sv.Kg_e + geo.Kc)
            / geo.B
            * psi
            * (a_ * _M[k + 1] - b_ * _M[k + 2])
        )
        out["n"].append(n_tr + n_ps)
        out["m"].append(m_tr + m_ps)
        out["d"].append(d_tr + d_ps)
    return out


def _parallel_moment(geo, R):
    """Gamma(theta) = int d3v v_par E^k h from the moment equation i B d_l(Gamma/B) = R(theta),
    dl = J B dtheta, Gamma(0) = 0 (twisting parity): Gamma = -i B int_0^theta J R dtheta'.
    """
    th = geo.theta
    f = geo.J * R
    cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(th))])
    i0 = int(np.argmin(np.abs(th)))
    return -1j * geo.B * (cum - cum[i0])


def weights(Sv, res):
    """Quasilinear flux weights of one eigenmode.

    Sv: solver.Solver (main model: ions='stream', bpar='field' or 'off', psi=True); res: its
    result at the root (Solver.solve / result, with 'coef').  Returns a dict, solver units:
      'Gamma_i', 'Gamma_e', 'Q_i', 'Q_e': dict(phi=, apar=, bpar=, es=, em=, total=) per <|phi|^2>
         (GENE's definitions: factor 2 for +-k_y, flux-surface average with the Jacobian over the
         central turn, em = apar + bpar);
      'phi2' (<|phi|^2> of res's normalisation), 'kperp2' (<k_perp^2>_phi = int k_y^2 g^yy |phi|^2 J
      / int |phi|^2 J), 'ambipolarity' (relative mismatch of e Gamma_i vs e Gamma_e per channel).
    """
    assert Sv.ions == "stream" and Sv.use_psi and Sv.bpar in ("field", "off")
    geo, p = Sv.geo, Sv.p
    omega = complex(res["omega"])
    ky = Sv.ky
    th, J, B = geo.theta, geo.J, geo.B
    dth = geo.dth
    coef = res["coef"]
    c = dict(
        phi=coef["phi"],
        bpar=coef.get("bpar", np.zeros(Sv.Bb.shape[0], complex)),
        psi=coef["psi"],
    )
    phi, bpar, psi = res["phi"], res["bpar"], res["psi"]
    sr, si = _periodic_spline(th, psi)
    dpsi = sr(th, 1) + 1j * si(th, 1)
    apar = -1j / omega * dpsi / (J * B)

    def apar_at(z):
        return -1j / omega * (sr(z, 1) + 1j * si(z, 1)) / (geo.g.sJ(z) * geo.g.sB(z))

    norm = np.sum(J) * dth
    phi2 = float(np.sum(np.abs(phi) ** 2 * J) * dth / norm)
    kp2 = ky**2 * geo.gyy
    kperp2 = float(np.sum(kp2 * np.abs(phi) ** 2 * J) / np.sum(np.abs(phi) ** 2 * J))

    def integ(mom, field):  # int J dtheta conj(mom) field
        return np.sum(J * np.conj(mom) * field) * dth

    Ti = p["Ti"]
    # ---- ions (charge +1, n = 1, T = Ti): exact orbit integration
    im = Sv.stream.flux_moments(omega, dict(c, dapar=apar_at))
    # ---- electrons (charge -1, n = 1, T = 1)
    em = _electron_moments(Sv, omega, dict(c, psi_theta=psi))
    ae, be, vle = Sv.ae, Sv.be, Sv.sw["vl_e"]
    Pe, Be, Ae = [], [], []
    for k in (0, 1):
        S_k = (omega - ae) * _M[k] - be * _M[k + 1]
        T_k = (omega - ae) * _N[k] - be * _N[k + 1]
        R = -S_k * phi + vle * T_k * bpar / B - omega * em["n"][k] + em["d"][k]
        G = _parallel_moment(geo, R)
        Pe.append(integ(em["n"][k], phi))
        Be.append(integ(em["m"][k], bpar))
        Ae.append(integ(G, apar))

    def chan(P, Bm, A, k, T_over_q, Tfac):
        f = 2 * Tfac / norm
        out = dict(
            phi=-f * np.real(1j * ky * P[k]),
            apar=f * np.real(1j * ky * A[k]),
            bpar=-f * T_over_q * np.real(1j * ky * Bm[k]),
        )
        out["es"] = out["phi"]
        out["em"] = out["apar"] + out["bpar"]
        out["total"] = out["es"] + out["em"]
        return {key: float(v) / phi2 for key, v in out.items()}

    w = dict(
        Gamma_i=chan(im["P"], im["B"], im["A"], 0, Ti, 1.0),
        Q_i=chan(im["P"], im["B"], im["A"], 1, Ti, Ti),
        Gamma_e=chan(Pe, Be, Ae, 0, -1.0, 1.0),
        Q_e=chan(Pe, Be, Ae, 1, -1.0, 1.0),
        phi2=phi2,
        kperp2=kperp2,
    )
    amb = {}
    for ch in ("phi", "apar", "bpar"):
        gi, ge = w["Gamma_i"][ch], w["Gamma_e"][ch]
        amb[ch] = abs(gi - ge) / max(abs(gi), abs(ge), 1e-300)
    w["ambipolarity"] = amb
    return w


# ---------------------------------------------------------------------------- one root per process
def _solve_one(path, ky_ref, seed, scan, out):
    """Worker: solve one k_y of the deck at path and compute its weights (plain dict to out)."""
    warnings.simplefilter("ignore")
    from .gene_io import Deck

    t0 = time.time()
    try:
        deck = Deck(path)
        r = deck.solve(ky_ref, omega0=seed, scan=scan)
        ok = bool(r["converged"] and r["gamma"] > 0 and np.isfinite(r["omega"]))
        d = dict(
            ky_ref=ky_ref,
            ky=r["ky_solver"],
            omega=float(r["omega_gene"]),
            gamma=float(r["gamma"]),
            omega_ref=float(r["omega_ref"]),
            gamma_ref=float(r["gamma_ref"]),
            converged=ok,
            weights=weights(r["solver"], r) if ok else None,
        )
    except Exception as e:  # reported, the k_y counts as no root
        d = dict(ky_ref=ky_ref, converged=False, error=repr(e))
    d["seconds"] = time.time() - t0
    out.put(d)


def _run_root(ctx, path, ky_ref, seed, scan, timeout):
    q = ctx.Queue()
    pr = ctx.Process(target=_solve_one, args=(str(path), ky_ref, seed, scan, q))
    pr.start()
    try:
        d = q.get(timeout=timeout)
    except Exception:
        d = dict(ky_ref=ky_ref, converged=False, error="timeout after %g s" % timeout)
    pr.join(5)
    if pr.is_alive():
        pr.terminate()
        pr.join()
    return d


def _chain(ctx, path, kys, omega0, scan, timeout, verbose):
    """Roots along a list of k_y with continuation: each seeded by the previous converged root
    (scaled with k_y), the solver's default seeds otherwise."""
    out, prev = [], None
    for n, ky in enumerate(kys):
        if prev is not None and prev["converged"]:
            seed = complex(prev["omega_ref"], prev["gamma_ref"]) * ky / prev["ky_ref"]
        else:
            seed = omega0 if n == 0 else None
        d = _run_root(ctx, path, ky, seed, scan, timeout)
        if verbose:
            print(
                "  k_y %-7.4g gamma %+.5f omega %+.5f %s (%.0f s)"
                % (
                    ky,
                    d.get("gamma_ref", np.nan),
                    d.get("omega_ref", np.nan),
                    "ok" if d["converged"] else d.get("error", "no growing root"),
                    d.get("seconds", np.nan),
                ),
                flush=True,
            )
        out.append(d)
        prev = d
    return out


def _deck_path(source, workdir):
    """A GENE parameters path for source (path, run directory or pyrokinetics Pyro object)."""
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        return p / "parameters" if p.is_dir() else p
    # a Pyro object: write its GENE deck (pyrokinetics converts from any code)
    p = Path(workdir) / "parameters"
    source.write_gk_file(p, gk_code="GENE")
    return p


def saturate(modes, C=None, rule="mixing_length", species_T=None):
    """Saturated fluxes from per-k_y modes (list of dicts with ky [rho_s], gamma, converged,
    weights), solver units.  rule 'mixing_length': <|phi|^2>(k_y) = C (gamma/<k_perp^2>)^2,
    integrated over k_y with the trapezoid rule (stable or unconverged k_y contribute 0).
    """
    if rule != "mixing_length":
        raise ValueError("unknown saturation rule %r" % rule)
    C = C_ML if C is None else C
    ky = np.array([m["ky"] if "ky" in m else np.nan for m in modes], float)
    keys = [
        (s, ch)
        for s in ("Q_i", "Q_e", "Gamma_i", "Gamma_e")
        for ch in ("es", "apar", "bpar", "em", "total")
    ]
    vals = {k: np.zeros(len(modes)) for k in keys}
    amp = np.zeros(len(modes))
    for j, m in enumerate(modes):
        if not m["converged"] or m["weights"] is None or m["gamma"] <= 0:
            continue
        w = m["weights"]
        amp[j] = C * (m["gamma"] / w["kperp2"]) ** 2
        for s, ch in keys:
            vals[(s, ch)][j] = w[s][ch] * amp[j]
    good = np.isfinite(ky)
    order = np.argsort(ky[good])
    x = ky[good][order]

    def integ(y):
        return float(np.trapz(y[good][order], x)) if x.size > 1 else 0.0

    out = {"%s_%s" % k: integ(v) for k, v in vals.items()}
    out["phi2"] = amp
    return out


def fluxes(
    source,
    ky=None,
    C=None,
    rule="mixing_length",
    parallel=1,
    timeout=300.0,
    omega0=None,
    scan="auto",
    verbose=False,
):
    """Quasilinear hKBM fluxes for a flux surface, for use as a transport (e.g. T3D) flux model.

    source  GENE parameters file, a run directory containing one, or a pyrokinetics Pyro object
            (any code; written as a GENE deck by pyrokinetics).  Must be a deck the solver accepts
            (two species, Miller/MXH, no flow; see README).
    ky      k_y values in the deck's 1/rho_ref (default: KY_DEFAULT in k_y rho_s, converted).
    C       saturation constant (default C_ML, the stella STEP calibration).
    parallel  number of worker processes: the k_y list is cut into this many contiguous chains,
            each followed by continuation; 1 = one chain (best root following).
    timeout seconds allowed per root (the worker process is killed after it; counts as no root).
    omega0  seed for the first k_y of each chain (GENE sign omega + i gamma, deck units).

    Returns dict (deck's GENE normalisation: Q in n_ref T_ref c_ref rho_ref^2/L_ref^2, Gamma in
    n_ref c_ref rho_ref^2/L_ref^2):
      Q_i, Q_e, Gamma (= Gamma_i = Gamma_e), and the splits Q_i_es, Q_i_em, Q_i_apar, Q_i_bpar,
      Q_e_..., Gamma_es, Gamma_em; ky (deck units), ky_rho_s, gamma, omega (deck units, GENE sign),
      converged (per k_y), weights (per k_y: weights() dicts, solver units, None without a root),
      phi2 (saturated <|phi|^2> per k_y, solver units), C, rule, seconds."""
    import tempfile

    from .gene_io import Deck

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = _deck_path(source, tmp)
        deck = Deck(path)
        rs = deck.units["rho_s_over_rho_ref"]
        kys = (
            [k / rs for k in KY_DEFAULT]
            if ky is None
            else [float(k) for k in np.atleast_1d(ky)]
        )
        kys = sorted(kys)
        ctx = mp.get_context("spawn")
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")
        nch = max(1, min(int(parallel), len(kys)))
        chains = [list(c) for c in np.array_split(np.array(kys), nch) if len(c)]
        if nch == 1:
            modes = _chain(ctx, path, chains[0], omega0, scan, timeout, verbose)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(nch) as ex:
                futs = [
                    ex.submit(_chain, ctx, path, c, omega0, scan, timeout, verbose)
                    for c in chains
                ]
                modes = [m for f in futs for m in f.result()]
    for m in modes:
        m.setdefault("ky", m["ky_ref"] * rs)
        m.setdefault("gamma", np.nan)
        m.setdefault("weights", None)
    sat = saturate(modes, C=C, rule=rule)
    u = deck.units
    fQ = u["n_e"] * u["T_e"] ** 2.5 * u["m_i"] ** 0.5
    fG = u["n_e"] * u["T_e"] ** 1.5 * u["m_i"] ** 0.5
    res = dict(
        Q_i=sat["Q_i_total"] * fQ,
        Q_e=sat["Q_e_total"] * fQ,
        Gamma=sat["Gamma_i_total"] * fG,
        Gamma_es=sat["Gamma_i_es"] * fG,
        Gamma_em=sat["Gamma_i_em"] * fG,
        ky=np.array([m["ky_ref"] for m in modes]),
        ky_rho_s=np.array([m["ky"] for m in modes]),
        gamma=np.array([m.get("gamma_ref", np.nan) for m in modes]),
        omega=np.array([m.get("omega_ref", np.nan) for m in modes]),
        converged=np.array([bool(m["converged"]) for m in modes]),
        weights=[m["weights"] for m in modes],
        errors=[m.get("error") for m in modes],
        phi2=sat["phi2"],
        C=C_ML if C is None else C,
        rule=rule,
        seconds=time.time() - t0,
    )
    for s in ("Q_i", "Q_e"):
        for ch in ("es", "em", "apar", "bpar"):
            res["%s_%s" % (s, ch)] = sat["%s_%s" % (s, ch)] * fQ
    return res
