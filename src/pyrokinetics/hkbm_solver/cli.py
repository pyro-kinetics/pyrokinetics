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


if __name__ == "__main__":
    sys.exit(main())
