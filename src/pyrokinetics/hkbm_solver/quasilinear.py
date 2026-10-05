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
# k_y rho_s range over which the solver's hKBM was validated against GENE (STEP)
HKBM_KY_RANGE = (0.05, 0.6)


_trapz = getattr(np, "trapezoid", None) or np.trapz


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
    dl = J B dtheta: Gamma = -i B [int_-pi^theta J R dtheta' - c], c fixed by Gamma(-pi)/B(-pi) =
    -Gamma(pi)/B(pi) (the mean of the two one-sided conditions; for a twisting-parity mode this
    is exactly Gamma(0) = 0)."""
    th = geo.theta
    f = geo.J * R
    cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(th))])
    end = cum[-1] + 0.5 * (f[-1] + f[0]) * (np.pi - th[-1])  # up to theta = pi
    return -1j * geo.B * (cum - 0.5 * end)


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


# ---------------------------------------------------------------------------- linear runs
def _apar_theta(geo, psi, omega):
    """A_par = -(i/omega) d_l psi on the solver's theta grid (dl = J B dtheta)."""
    sr, si = _periodic_spline(geo.theta, psi)
    return -1j / omega * (sr(geo.theta, 1) + 1j * si(geo.theta, 1)) / (geo.J * geo.B)


def _source_deck(source, workdir):
    """GENE parameters path for source: a parameters file, a run directory, or a pyrokinetics
    Pyro object (written as a GENE deck; flow, flow shear and Z_eff, which the solver does not
    have, are set to zero / ignored with a warning)."""
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        return p / "parameters" if p.is_dir() else p
    import f90nml

    p = Path(workdir) / "parameters"
    source.write_gk_file(p, gk_code="GENE")
    nml = f90nml.read(p)
    changed = []
    for grp in ("general", "external_contr"):
        if grp in nml:
            for k in list(nml[grp].keys()):
                if k.lower() in ("exbrate", "pfsrate", "omega0_tor") and nml[grp][k]:
                    changed.append("%s = %s" % (k, nml[grp][k]))
                    nml[grp][k] = 0.0
    for sp in nml["species"] if isinstance(nml["species"], list) else [nml["species"]]:
        for k in list(sp.keys()):
            if k.lower() in ("omegator",) and sp[k]:
                changed.append("%s = %s" % (k, sp[k]))
                sp[k] = 0.0
    if changed:
        warnings.warn(
            "hKBM solver: no flow or flow shear in the model, set to zero: "
            + ", ".join(changed)
        )
    zeff = float(nml["general"].get("zeff", 1.0) or 1.0) if "general" in nml else 1.0
    if abs(zeff - 1.0) > 1e-6:
        warnings.warn(
            "hKBM solver: Z_eff = %g ignored (one ion species; nu_ei with Z = 1)" % zeff
        )
    geo = nml["geometry"]
    if float(geo.get("dpdx_pm", -2) or 0) == 0 and float(geo.get("amhd", 0) or 0) == 0:
        warnings.warn(
            "hKBM solver: the deck written from the Pyro object has beta' = 0 (dpdx_pm = amhd "
            "= 0); the hKBM depends strongly on beta' (pyro.local_geometry.beta_prime).  A Pyro "
            "read from a GENE deck with dpdx_pm = -1 and no reference B0 gets beta_prime = 0."
        )
    nml.write(p, force=True)
    return p


def _rho_star(source, deck, rho_star):
    if rho_star is not None:
        return float(rho_star)
    try:  # a Pyro object with physical reference values
        nr = source.norms.gene
        return float((nr.rhoref / nr.lref).to("dimensionless").magnitude)
    except Exception:
        pass
    rs = float(deck.nml["geometry"].get("rhostar", -1) or -1)
    if rs > 0:
        return rs
    raise ValueError(
        "toroidal mode numbers n need rho_star = rho_ref/L_ref (argument rho_star, Pyro "
        "reference values, or rhostar in the GENE deck)"
    )


def _flags(r, ky_s):
    om, ga = r["omega_gene"], r["gamma"]
    f = dict(
        converged=bool(r["converged"]),
        growing=bool(np.isfinite(ga) and ga > 0),
        ion_direction=bool(om > 0),
        ballooning=bool(r.get("frac_pi2", 0.0) >= 0.5),
        not_alfvenic=bool(abs(om) < 0.8),
        ky_in_range=bool(HKBM_KY_RANGE[0] <= ky_s <= HKBM_KY_RANGE[1]),
    )
    return f, all(f.values())


def _record(deck, r, ky_ref, theta0, n=None, fields=True):
    """One mode in the deck's GENE normalisation (see run_linear)."""
    u = deck.units
    rs, cs, Te = u["rho_s_over_rho_ref"], u["c_s_over_c_ref"], u["T_e"]
    ky_s = ky_ref * rs
    rec = dict(
        ky=ky_ref,
        ky_rho_s=ky_s,
        n=n,
        theta0=theta0,
        omega=np.nan,
        gamma=np.nan,
        converged=False,
        hkbm_like=False,
        checks=None,
        weights=None,
        weights_solver=None,
        error=r.get("error") if isinstance(r, dict) else None,
    )
    if r is None or "omega" not in r:
        return rec
    checks, ok = _flags(r, ky_s)
    rec.update(
        omega=float(r["omega_gene"]) * cs,
        gamma=float(r["gamma"]) * cs,
        gamma_solver=float(r["gamma"]),
        converged=checks["converged"] and checks["growing"],
        hkbm_like=ok,
        checks=checks,
        seconds=r.get("seconds"),
    )
    if not rec["converged"]:
        return rec
    Sv = r["solver"]
    geo = Sv.geo
    w = weights(Sv, r)
    rec["weights_solver"] = w
    fQ = u["n_e"] * Te**2.5 * u["m_i"] ** 0.5
    fG = u["n_e"] * Te**1.5 * u["m_i"] ** 0.5
    fphi = (Te * rs) ** 2  # <|phi_deck|^2> / <|phi_solver|^2>
    wd = {}
    for key, f in (("Q_i", fQ), ("Q_e", fQ), ("Gamma_i", fG), ("Gamma_e", fG)):
        wd[key] = {ch: v * f / fphi for ch, v in w[key].items()}
    qt = wd["Q_i"]["total"] + wd["Q_e"]["total"]
    wd["shares"] = dict(
        Q_i=wd["Q_i"]["total"] / qt,
        Q_e=wd["Q_e"]["total"] / qt,
        Gamma=wd["Gamma_i"]["total"] / qt,
    )
    wd["kperp2"] = w["kperp2"] / rs**2
    wd["ambipolarity"] = w["ambipolarity"]
    rec["weights"] = wd
    rec["kperp2_avg"] = wd["kperp2"]
    if fields:
        omega = complex(r["omega"])
        phi, bpar = r["phi"], r["bpar"]
        apar = _apar_theta(geo, r["psi"], omega)
        k = int(np.argmax(np.abs(phi)))
        p0 = (
            phi[k] * Te * rs
        )  # deck phi at its maximum: normalise to max |phi| = 1, real
        rec.update(
            theta=geo.theta.copy(),
            phi=phi / phi[k],
            apar=apar * rs**2 / p0,
            bpar=bpar * rs / p0,
            kperp2=Sv.ky**2 * geo.gyy / rs**2,
            jacobian=geo.J.copy(),
            bmag=geo.B.copy(),
        )
    return rec


def run_linear(
    source,
    ky=None,
    n=None,
    theta0=(0.0,),
    rho_star=None,
    timeout=60.0,
    omega0=None,
    scan="auto",
    fields=True,
    verbose=False,
):
    """Linear hKBM modes on a (k_y, theta0) grid in one process: the raw input of a
    quasilinear transport model (e.g. T3D's GS2-QL machinery).

    source   pyrokinetics Pyro object (local Miller/MXH, electrons + one ion species; any code,
             written as a GENE deck by pyrokinetics), a GENE parameters file or its run directory.
             Flow shear, toroidal flow and Z_eff are not in the model: set to zero / ignored
             with a warning.
    ky       k_y rho_ref values (deck normalisation), or
    n        toroidal mode numbers: k_y rho_ref = n rho_star/|C_y| (GENE; rho_star = rho_ref/L_ref
             from rho_star=, the Pyro reference values or the deck's rhostar).
    theta0   ballooning angles (k_x = shat k_y theta0); theta0 != 0 uses a basis without parity
             (twice the cost).  Validated against GENE only at theta0 = 0.
    timeout  seconds per root (the search stops and the mode is returned with converged False).
    omega0   seed for the first root (GENE sign omega + i gamma, deck units).  Roots are followed
             in k_y outward from k_y rho_s ~ 0.2 and in theta0 from the previous theta0.

    Returns a list (theta0 outer, k_y inner, in the order given) of dicts, deck units (GENE
    normalisation, sign: omega > 0 = ion diamagnetic direction):
      ky, ky_rho_s, n, theta0, gamma, omega, converged (growing root found), hkbm_like (all
      checks passed), checks (converged, growing, ion_direction, ballooning: >= half of |phi|^2
      within |theta| < pi/2, not_alfvenic: |omega| < 0.8 c_s/L_ref, ky_in_range: k_y rho_s in
      HKBM_KY_RANGE), error, seconds;
      with a root: theta (ballooning angle, central turn), phi, apar, bpar (max |phi| = 1),
      kperp2(theta) [1/rho_ref^2], jacobian(theta) (GENE's J; dl = J B dtheta), bmag(theta)
      (B/B_ref), kperp2_avg = int kperp2 |phi|^2 J / int |phi|^2 J, and weights: Q_i, Q_e,
      Gamma_i, Gamma_e (= Gamma_i) per <|phi|^2> (GENE nrg definitions, channels phi, apar,
      bpar, es, em, total), shares (Q_i, Q_e, Gamma over Q_i + Q_e), ambipolarity;
      weights_solver: the same in the solver's units (T_e, n_e, m_i, rho_s)."""
    import tempfile

    from .gene_io import Deck

    with tempfile.TemporaryDirectory() as tmp:
        path = _source_deck(source, tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deck = Deck(path)
    rs = deck.units["rho_s_over_rho_ref"]
    if n is not None:
        rst = _rho_star(source, deck, rho_star)
        ns = [float(x) for x in np.atleast_1d(n)]
        kys = [x * rst / abs(float(deck.geo.h["Cy"])) for x in ns]
    else:
        kys = (
            [k / rs for k in KY_DEFAULT]
            if ky is None
            else [float(k) for k in np.atleast_1d(ky)]
        )
        ns = [None] * len(kys)
    th0s = [float(t) for t in np.atleast_1d(theta0)]
    order = np.argsort(kys)
    i0 = int(order[np.argmin(np.abs(np.log(np.asarray(kys)[order] * rs / 0.2)))])
    pos = list(order).index(i0)
    walk = list(order[pos:]) + list(order[:pos][::-1])
    out = {}
    for it, th0 in enumerate(th0s):
        prev = None
        for j in walk:
            kyj = kys[j]
            seed = None
            if it > 0 and out[(it - 1, j)]["converged"]:
                o = out[(it - 1, j)]
                seed = complex(o["omega"], o["gamma"])
            elif j == order[pos - 1] if pos > 0 else False:
                o = out[(it, i0)]
                if o["converged"]:
                    seed = complex(o["omega"], o["gamma"]) * kyj / o["ky"]
            elif prev is not None and prev["converged"]:
                seed = complex(prev["omega"], prev["gamma"]) * kyj / prev["ky"]
            elif j == i0 and it == 0:
                seed = omega0
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = deck.solve(
                        kyj, omega0=seed, scan=scan, theta0=th0, timeout=timeout
                    )
                r["seconds"] = time.time() - t0
                rec = _record(deck, r, kyj, th0, ns[j], fields)
            except TimeoutError:
                rec = _record(
                    deck, dict(error="timeout after %g s" % timeout), kyj, th0, ns[j]
                )
            except Exception as e:  # reported per mode; T3D drops or falls back
                rec = _record(deck, dict(error=repr(e)), kyj, th0, ns[j])
            rec.setdefault("seconds", time.time() - t0)
            out[(it, j)] = rec
            prev = rec
            if verbose:
                print(
                    "  k_y %-8.4g theta0 %-5.3g gamma %+.5f omega %+.5f %s%s (%.1f s)"
                    % (
                        kyj,
                        th0,
                        rec["gamma"],
                        rec["omega"],
                        "hKBM" if rec["hkbm_like"] else "",
                        (
                            ""
                            if rec["converged"]
                            else " no root " + str(rec["error"] or "")
                        ),
                        rec["seconds"],
                    ),
                    flush=True,
                )
    return [out[(it, j)] for it in range(len(th0s)) for j in range(len(kys))]


# ---------------------------------------------------------------------------- saturation
def saturate(modes, C=None, rule="mixing_length"):
    """Saturated fluxes (solver units) from run_linear modes at one theta0.  rule
    'mixing_length': <|phi|^2>(k_y) = C (gamma/<k_perp^2>)^2 (solver units: rho_s, c_s/L_ref),
    trapezoid integral over k_y rho_s; modes without a growing root contribute 0."""
    if rule != "mixing_length":
        raise ValueError("unknown saturation rule %r" % rule)
    C = C_ML if C is None else C
    ky = np.array([m["ky_rho_s"] for m in modes], float)
    keys = [
        (s, ch)
        for s in ("Q_i", "Q_e", "Gamma_i", "Gamma_e")
        for ch in ("phi", "apar", "bpar", "es", "em", "total")
    ]
    vals = {k: np.zeros(len(modes)) for k in keys}
    amp = np.zeros(len(modes))
    for j, m in enumerate(modes):
        w = m.get("weights_solver")
        if not m["converged"] or w is None:
            continue
        amp[j] = C * (m["gamma_solver"] / w["kperp2"]) ** 2
        for s, ch in keys:
            vals[(s, ch)][j] = w[s][ch] * amp[j]
    o = np.argsort(ky)
    out = {
        "%s_%s" % k: (float(_trapz(v[o], ky[o])) if ky.size > 1 else 0.0)
        for k, v in vals.items()
    }
    out["phi2"] = amp
    return out


def _chain_job(path, kys, omega0, scan, timeout, verbose):
    warnings.simplefilter("ignore")
    return run_linear(
        path,
        ky=kys,
        omega0=omega0,
        scan=scan,
        timeout=timeout,
        fields=False,
        verbose=verbose,
    )


def fluxes(
    source,
    ky=None,
    C=None,
    rule="mixing_length",
    parallel=1,
    timeout=60.0,
    omega0=None,
    scan="auto",
    verbose=False,
):
    """Quasilinear hKBM fluxes of a flux surface with the solver's own saturation rule (a
    comparison line for transport models that apply their own rule to run_linear output).

    source, ky, timeout, omega0, scan: as run_linear (theta0 = 0).  C: saturation constant
    (default C_ML, calibrated on the stella STEP (q, beta_e) scan).  parallel: number of worker
    processes; the k_y list is cut into this many contiguous chains, each followed by
    continuation (1 = in this process, best root following).

    Returns dict, deck's GENE normalisation (Q in n_ref T_ref c_ref rho_ref^2/L_ref^2, Gamma in
    n_ref c_ref rho_ref^2/L_ref^2): Q_i, Q_e, Gamma (= Gamma_i = Gamma_e), the splits
    Q_i_es/_em/_apar/_bpar, Q_e_..., Gamma_es, Gamma_em; ky, ky_rho_s, gamma, omega (deck
    units, GENE sign), converged, hkbm_like (per k_y), modes (run_linear records), phi2
    (saturated <|phi|^2> per k_y, solver units), C, rule, seconds."""
    import tempfile

    from .gene_io import Deck

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = _source_deck(source, tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deck = Deck(path)
        rs = deck.units["rho_s_over_rho_ref"]
        kys = sorted(
            [k / rs for k in KY_DEFAULT]
            if ky is None
            else [float(k) for k in np.atleast_1d(ky)]
        )
        nch = max(1, min(int(parallel), len(kys)))
        if nch == 1:
            modes = _chain_job(path, kys, omega0, scan, timeout, verbose)
        else:
            from concurrent.futures import ProcessPoolExecutor

            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(v, "1")
            chains = [list(c) for c in np.array_split(np.array(kys), nch)]
            with ProcessPoolExecutor(nch, mp_context=mp.get_context("spawn")) as ex:
                futs = [
                    ex.submit(_chain_job, str(path), c, omega0, scan, timeout, verbose)
                    for c in chains
                ]
                modes = [m for f in futs for m in f.result()]
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
        ky=np.array([m["ky"] for m in modes]),
        ky_rho_s=np.array([m["ky_rho_s"] for m in modes]),
        gamma=np.array([m["gamma"] for m in modes]),
        omega=np.array([m["omega"] for m in modes]),
        converged=np.array([m["converged"] for m in modes]),
        hkbm_like=np.array([m["hkbm_like"] for m in modes]),
        modes=modes,
        phi2=sat["phi2"],
        C=C_ML if C is None else C,
        rule=rule,
        seconds=time.time() - t0,
    )
    for s in ("Q_i", "Q_e"):
        for ch in ("es", "em", "apar", "bpar"):
            res["%s_%s" % (s, ch)] = sat["%s_%s" % (s, ch)] * fQ
    return res
