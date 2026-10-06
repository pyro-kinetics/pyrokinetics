"""Experimental Lorentz-MTM adapter for the existing T3D linear-mode contract.

Opt in through run_linear(..., modes=("hkbm", "mtm"),
mtm_kw={"backend": "lorentz_ei"}). No saturation rule or MTM species flux
shares are inferred here. A growing root is not a validated transport model.
All fields and field-weighted widths cover the SAME full ballooning domain.
"""

import tempfile
import time

import numpy as np

from .gene_io import Deck
from .mtm_collisional import CollisionalMTMSolver
from .quasilinear import KY_DEFAULT, _rho_star, _source_deck


def _field_metrics(solver, result):
    """Trapezoidal J|field|^2 widths, in solver rho_s units, without rescaling."""
    kp = {}
    for field in ("phi", "apar"):
        weight = solver.wtheta * np.abs(result[field]) ** 2
        norm = float(np.sum(weight))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"nonfinite or zero {field} norm")
        kp[field] = float(np.sum(weight * solver.kperp2) / norm)
    amplitude = float(np.max(np.abs(result["apar"])) / np.max(np.abs(result["phi"])))
    return kp, amplitude


def solve_deck(
    deck,
    ky_ref,
    omega0=None,
    theta0=0.0,
    coll="deck",
    timeout=90.0,
    maxit=20,
    tol=1e-7,
    max_side_turns=32,
    verbose=False,
    **options,
):
    """Collisional MTM, with GENE-sign seed/output in the deck's units.

    theta0=0 only. Default: the exploratory Figure-16 fast profile, nominal
    deck collisions and at most 32 side turns (65 total periods). Explicit
    nturns overrides this heuristic. There is no boundary-matching claim.
    omega_solver/gamma_solver and kperp2 use c_s/L_ref and rho_s; kperp2_ref
    uses rho_ref. Raw phi and apar retain the numerical solver's common scale.
    """
    if not np.isfinite([ky_ref, theta0, timeout, tol]).all() or ky_ref <= 0:
        raise ValueError("finite positive ky, timeout and tolerance required")
    if theta0 != 0:
        raise ValueError("Lorentz-MTM currently supports theta0=0 only")
    if timeout <= 0 or tol <= 0 or maxit < 1 or int(maxit) != maxit:
        raise ValueError("positive timeout, tolerance and integer maxit required")
    if int(max_side_turns) != max_side_turns or max_side_turns < 0:
        raise ValueError("max_side_turns must be a nonnegative integer")
    units = deck.units
    rs, cs = units["rho_s_over_rho_ref"], units["c_s_over_c_ref"]
    ky_s = float(ky_ref) * rs
    requested_turns = max(8, int(np.ceil(32 * 0.2848826 / ky_s)))
    explicit_domain = "nturns" in options
    options = dict(options)
    for key, value in dict(
        npt=24,
        nE=12,
        nxi=16,
        theta_order=3,
        nturns=min(int(max_side_turns), requested_turns),
    ).items():
        options.setdefault(key, value)
    options["coll"] = deck.solver_kw["coll"] if coll == "deck" else float(coll)
    seed = None
    if omega0 is not None:
        value = complex(omega0)
        seed = complex(-value.real, value.imag) / cs
        if not np.isfinite([seed.real, seed.imag]).all() or seed.imag <= 0:
            raise ValueError("omega0 must be a finite growing GENE-sign seed")
    start = time.monotonic()
    solver = CollisionalMTMSolver.from_deck(deck, ky_ref, **options)
    result = solver.solve(seed, timeout=timeout, maxit=maxit, tol=tol, verbose=verbose)
    kp, amplitude = _field_metrics(solver, result)
    omega_solver = complex(result["omega"])
    growing = bool(
        result["status"] == "growing_root"
        and result["converged"]
        and np.isfinite(omega_solver)
        and omega_solver.imag > 0
    )
    domain_ok = not result["domain_warning"]
    kpe = float(
        np.sqrt(solver.kperp2[solver.mid] * 2 * solver.me) / solver.B[solver.mid]
    )
    result.update(
        ky=float(ky_ref),
        ky_rho_s=ky_s,
        theta0=0.0,
        branch="mtm",
        backend="lorentz_ei",
        parity="tearing",
        omega_solver=omega_solver,
        gamma_solver=omega_solver.imag if growing else np.nan,
        omega=-omega_solver.real * cs if growing else np.nan,
        gamma=omega_solver.imag * cs if growing else np.nan,
        converged=growing,
        growing=growing,
        no_root_verdict=False,
        seconds=time.monotonic() - start,
        solver=solver,
        apar_theta=result["theta"].copy(),
        kperp2=kp,
        kperp2_ref={key: value / rs**2 for key, value in kp.items()},
        apar_over_phi=amplitude,
        domain_capped=bool(not explicit_domain and requested_turns > max_side_turns),
        full_domain_fields=True,
        omitted_fields=("bpar",),
        validity=dict(
            edge_phi=result["edge_phi"],
            edge_g=result["edge_g"],
            beta_e=solver.beta,
            nturns=solver.nturns,
            kperp_rho_e=kpe,
            kperp_rho_e_over_beta_e=kpe / solver.beta,
            nu_ei_over_omega=solver.nu_ei / max(abs(omega_solver), 1e-30),
            domain_warning=result["domain_warning"],
            apar_shape_warning=result["apar_shape_warning"],
            ampere_core_residual=result["ampere_core_residual"],
        ),
        transport=dict(
            validated=False,
            flux_shares_available=False,
            domain_resolved=bool(growing and domain_ok),
            layer_fallback_required=bool(not growing or not domain_ok),
            kperp2_layer=kp["phi"],
            kperp2_apar=kp["apar"],
            kr2_geometric=(
                float(np.sqrt(kp["phi"] * kp["apar"]))
                if growing and domain_ok
                else None
            ),
            units="rho_s^-2",
            rule="caller-owned; not a flux prediction",
        ),
    )
    return result


def _record(deck, result, ky, n=None, fields=True):
    """T3D/QL record: normalised fields in deck units, widths in both units."""
    units = deck.units
    rs, te = units["rho_s_over_rho_ref"], units["T_e"]
    rec = dict(
        ky=float(ky),
        ky_rho_s=float(ky) * rs,
        n=n,
        theta0=result.get("theta0", 0.0),
        branch="mtm",
        parity="tearing",
        backend="lorentz_ei",
        omega=np.nan,
        gamma=np.nan,
        gamma_solver=np.nan,
        converged=False,
        growing=False,
        hkbm_like=False,
        mtm_like=False,
        checks={},
        weights=None,
        weights_solver=None,
        no_root_verdict=False,
        validated=False,
        resolution_converged=False,
        collision_physics_match=False,
        error=result.get("error"),
        status=result.get("status", "error"),
        seconds=result.get("seconds"),
        omitted_fields=("bpar",),
    )
    for key in (
        "validity",
        "domain_warning",
        "apar_shape_warning",
        "domain_capped",
        "transport",
        "resolution",
        "relative_residual",
        "qn_residual",
        "kinetic_residual",
        "ampere_core_residual",
        "collision_parameter",
        "frequency_model",
        "collision_flr",
        "drift_sign",
        "full_domain_fields",
    ):
        if key in result:
            rec[key] = result[key]
    if not result.get("converged") or not result.get("growing"):
        return rec
    checks = dict(
        converged=True,
        growing=True,
        electron_direction=bool(result["omega_solver"].real > 0),
        phi_decays=bool(result["validity"]["edge_phi"] < 0.01),
        distribution_decays=bool(result["validity"]["edge_g"] < 0.01),
    )
    rec.update(
        omega=float(result["omega"]),
        gamma=float(result["gamma"]),
        gamma_solver=float(result["gamma_solver"]),
        converged=True,
        growing=True,
        checks=checks,
        mtm_like=all(checks.values()),
        kperp2_avg=result["kperp2_ref"]["phi"],
        kperp2_ref=dict(result["kperp2_ref"]),
    )
    kp = result["kperp2"]
    amp = dict(phi=1.0, apar=result["apar_over_phi"])
    rec["giacomin"] = dict(
        kperp2=dict(kp),
        amplitude=amp,
        Lambda_hat=result["gamma_solver"] * sum(amp[k] / kp[k] for k in amp),
        Q_i_over_Q=None,
        Q_e_over_Q=None,
        Gamma_over_Q=None,
    )
    if fields:
        solver = result["solver"]
        phi = result["phi"]
        p0 = phi[int(np.argmax(np.abs(phi)))]
        rec.update(
            theta=result["theta"].copy(),
            apar_theta=result["theta"].copy(),
            phi=phi / p0,
            apar=result["apar"] * rs / (p0 * te),
            kperp2=solver.kperp2.copy() / rs**2,
            jacobian=solver.J.copy(),
            bmag=solver.B.copy(),
        )
    return rec


def _warm_seed(warm, ky):
    candidates = [
        m
        for m in (warm or [])
        if m.get("backend") == "lorentz_ei"
        and m.get("converged")
        and m.get("theta0", 0) == 0
        and m.get("ky", 0) > 0
        and np.isfinite(m.get("gamma", np.nan))
        and np.isfinite(m.get("omega", np.nan))
        and m["gamma"] > 0
    ]
    if not candidates:
        return None
    mode = min(candidates, key=lambda m: abs(np.log(m["ky"] / ky)))
    if abs(np.log(mode["ky"] / ky)) > 0.5:
        return None
    return complex(mode["omega"], mode["gamma"]) * ky / mode["ky"]


def run_linear_mtm(
    source,
    ky=None,
    n=None,
    theta0=(0.0,),
    rho_star=None,
    timeout=None,
    omega0=None,
    fields=True,
    verbose=False,
    warm=None,
    mtm_kw=None,
):
    """Only the Lorentz-MTM branch; never runs hKBM or creates its own pool.

    Nonzero theta0 returns an explicit unsupported record. Warm roots are
    restricted to this backend. Failure returns NaN frequencies, not stability.
    mtm_like means growing/electron-direction/decaying, NOT model validation;
    the separate A-shape and resolution flags must be retained by the caller.
    """
    if ky is not None and n is not None:
        raise ValueError("choose ky or toroidal n, not both")
    with tempfile.TemporaryDirectory() as tmp:
        deck = Deck(_source_deck(source, tmp))
    rs = deck.units["rho_s_over_rho_ref"]
    if n is not None:
        ns = [float(value) for value in np.atleast_1d(n)]
        rst = _rho_star(source, deck, rho_star)
        kys = [value * rst / abs(float(deck.geo.h["Cy"])) for value in ns]
    else:
        kys = (
            [value / rs for value in KY_DEFAULT]
            if ky is None
            else list(np.atleast_1d(ky).astype(float))
        )
        ns = [None] * len(kys)
    if not kys or not np.isfinite(kys).all() or min(kys) <= 0:
        raise ValueError("need finite positive wavenumbers")
    th0s = [float(value) for value in np.atleast_1d(theta0)]
    if not th0s or not np.isfinite(th0s).all():
        raise ValueError("need finite theta0 values")
    options = dict(mtm_kw or {})
    explicit_seed = options.pop("omega0", omega0)
    search_timeout = options.pop("timeout", timeout if timeout is not None else 90.0)
    out = {}
    for it, th0 in enumerate(th0s):
        previous = None
        for j in np.argsort(kys):
            kyj = kys[j]
            start = time.monotonic()
            try:
                if th0 != 0:
                    raise ValueError("Lorentz-MTM currently supports theta0=0 only")
                seed = _warm_seed(warm, kyj)
                if seed is None and previous and previous["converged"]:
                    seed = (
                        complex(previous["omega"], previous["gamma"])
                        * kyj
                        / previous["ky"]
                    )
                if seed is None:
                    seed = explicit_seed
                result = solve_deck(
                    deck,
                    kyj,
                    omega0=seed,
                    timeout=search_timeout,
                    verbose=verbose,
                    **options,
                )
                rec = _record(deck, result, kyj, ns[j], fields)
            except Exception as error:
                rec = _record(
                    deck,
                    dict(
                        error=repr(error),
                        theta0=th0,
                        status="unsupported" if th0 != 0 else "error",
                    ),
                    kyj,
                    ns[j],
                    fields,
                )
            rec["seconds"] = time.monotonic() - start
            out[it, j] = rec
            previous = rec
    return [out[it, j] for it in range(len(th0s)) for j in range(len(kys))]
