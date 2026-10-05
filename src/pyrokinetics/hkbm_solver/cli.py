"""Command line entry point ``hkbm-solve``: run the hKBM solver on a GENE parameters file."""

import argparse
import sys
import warnings


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hkbm-solve",
        description="Solve the local linear hKBM eigenproblem for a GENE parameters file and write GENE "
        "output files (parameters, omega, field, nrg, miller) next to it.",
    )
    ap.add_argument(
        "run",
        nargs="?",
        help="GENE run directory containing 'parameters', or the parameters file itself",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory (default: the run directory)",
    )
    ap.add_argument(
        "--omega0",
        type=complex,
        default=None,
        help="seed for the first k_y, GENE sign and deck units: omega+gammaj (e.g. 0.1+0.08j)",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--scan",
        dest="scan",
        action="store_const",
        const=True,
        default="auto",
        help="always add a coarse search of the complex-omega plane to the seeds",
    )
    g.add_argument(
        "--no-scan",
        dest="scan",
        action="store_const",
        const=False,
        help="never search (default: search only when no seed gives a growing root)",
    )
    ap.add_argument(
        "--fluxes",
        action="store_true",
        help="quasilinear fluxes instead of GENE output files: solve on a k_y grid, print the "
        "per-k_y roots and weights and the mixing-length fluxes (gyro-Bohm, deck normalisation), "
        "write ql_fluxes.json to the output directory",
    )
    ap.add_argument(
        "--ky",
        type=float,
        nargs="+",
        default=None,
        help="with --fluxes: k_y rho_ref values (default: k_y rho_s 0.05 ... 0.6)",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="with --fluxes: worker processes over k_y (default 1)",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run the solver's internal checks and exit",
    )
    a = ap.parse_args(argv)
    if a.selftest:
        from .solver import selftest

        return 0 if selftest() else 1
    if a.run is None:
        ap.error("the run directory (or parameters file) is required")
    from .gene_io import UnsupportedDeck, run

    if a.quiet:
        warnings.simplefilter("ignore")
    if a.fluxes:
        return _fluxes(a, UnsupportedDeck)
    try:
        res = run(
            a.run, outdir=a.outdir, omega0=a.omega0, verbose=not a.quiet, scan=a.scan
        )
    except UnsupportedDeck as e:
        print("hkbm-solve: unsupported GENE deck: %s" % e, file=sys.stderr)
        return 2
    bad = [r for r in res if not r["converged"]]
    for r in res:
        print(
            "k_y %-8.4g gamma %+.6f omega %+.6f %s"
            % (
                r["ky"],
                r["gamma"],
                r["omega"],
                "" if r["converged"] else "NOT CONVERGED",
            )
        )
    return 1 if bad else 0


def _fluxes(a, UnsupportedDeck):
    import json
    from pathlib import Path

    import numpy as np

    from .quasilinear import fluxes

    p = Path(a.run)
    p = p / "parameters" if p.is_dir() else p
    try:
        r = fluxes(
            p,
            ky=a.ky,
            parallel=a.nproc,
            omega0=a.omega0,
            scan=a.scan,
            verbose=not a.quiet,
        )
    except UnsupportedDeck as e:
        print("hkbm-solve: unsupported GENE deck: %s" % e, file=sys.stderr)
        return 2
    print(
        "%8s %10s %10s %6s %9s %9s %9s %9s"
        % (
            "k_y",
            "gamma",
            "omega",
            "hKBM",
            "Q_i/phi2",
            "Q_e/phi2",
            "G/phi2",
            "<kperp2>",
        )
    )
    for m in r["modes"]:
        w = m["weights"]
        print(
            "%8.4g %+10.5f %+10.5f %6s %9.4g %9.4g %9.4g %9.4g"
            % (
                m["ky"],
                m["gamma"],
                m["omega"],
                "yes" if m["hkbm_like"] else "no",
                w["Q_i"]["total"] if w else np.nan,
                w["Q_e"]["total"] if w else np.nan,
                w["Gamma_i"]["total"] if w else np.nan,
                m.get("kperp2_avg", np.nan),
            )
        )
    print(
        "mixing length (C = %g): Q_i %.4g  Q_e %.4g  Gamma %.4g  (gyro-Bohm, deck normalisation; %.0f s)"
        % (r["C"], r["Q_i"], r["Q_e"], r["Gamma"], r["seconds"])
    )
    out = Path(a.outdir) if a.outdir else p.parent
    keep = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in r.items()
        if k != "modes"
    }
    keep["weights"] = [m["weights"] for m in r["modes"]]
    (out / "ql_fluxes.json").write_text(json.dumps(keep, indent=1, default=float))
    return 0 if r["converged"].any() else 1


if __name__ == "__main__":
    sys.exit(main())
