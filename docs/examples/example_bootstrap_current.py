r"""
Plot the bootstrap current density predicted by the Redl (2021) and
Sauter (1999) neoclassical models across a range of flux surfaces, and repeat
the scan with :math:`Z_{eff}` scaled up to show how sensitive each model is
to it.

The bootstrap current is a local quantity, so it needs a flux surface geometry
as well as the kinetic profiles. Some formats (TRANSP, GACODE, IMAS) carry both
in one file; for the rest (JETTO, pFile, SCENE, ITERDB, ELITEINP) supply the
equilibrium separately with ``--eq-file``.

When the kinetics file is a JETTO JSP, the run's own bootstrap current is
overlaid from the ``JPBS`` profile (:math:`\langle J_{bs} \cdot B \rangle / B_0`).
(``JZBS`` is the *toroidal* bootstrap density and pairs with ``Jphi_bs_eff``
instead, so it is not the right key to compare against here.)

``JPBS`` is normalised to :math:`B_0` while the models are naturally
:math:`\langle J_{bs} \cdot B \rangle / \langle B^2 \rangle^{1/2}`, so the two
differ by :math:`\langle B^2 \rangle^{1/2} / B_0`. Give a JETTO JSS file (or let
one be picked up alongside the JSP) and every curve is put on
:math:`\langle J_{bs} \cdot B \rangle / B_0` using the run's own ``B0``, making the
comparison exact. Without it the plot falls back to
:math:`\langle B^2 \rangle^{1/2}` and the axis label says so.

Note the ``--zeff-scale`` scan overrides the cached ``zeff`` scalar that the
neoclassical coefficients read, and nothing else. It is a sensitivity test of
the model coefficients, not a self-consistent change of plasma composition:
real impurity seeding would also dilute the main ions and change the ion
collisionality.

Examples
--------
Use the bundled TRANSP template, which holds equilibrium and kinetics::

    python example_bootstrap_current.py

A JETTO profile file with a separate equilibrium::

    python example_bootstrap_current.py jetto.jsp --eq-file jetto.eqdsk_out
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import Redl2021, Sauter1999

MODELS = {"Redl (2021)": Redl2021, "Sauter (1999)": Sauter1999}


def jetto_bootstrap(kinetics_file, time):
    """
    The JETTO run's own bootstrap current, from the JSP ``JPBS`` profile.

    Returns ``(psi_n, Jbs)`` with ``Jbs`` in MA/m^2, or ``None`` if the file
    carries no ``JPBS``.
    """
    from jetto_tools.binary import read_binary_file

    jsp = read_binary_file(kinetics_file)
    if "JPBS" not in jsp:
        return None

    # Match the time slice the kinetics reader picked
    if time is None:
        time_index = -1
    else:
        times = np.asarray(jsp["TIME"]).ravel()
        time_index = int(np.argmin(np.abs(times - time)))

    psi = np.asarray(jsp["PSI"])[time_index, :]
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])

    # JPBS lives on the cell boundary grid, put it on the centred grid that
    # PSI uses
    x = np.asarray(jsp["XVEC1"])[0, :]
    xb = np.asarray(jsp["XVEC2"])[0, :]
    jpbs = np.interp(x, xb, np.asarray(jsp["JPBS"])[time_index, :])

    return psi_n, np.abs(jpbs) / 1e6


def jetto_b0(jss_file, time):
    """
    Vacuum toroidal field ``B0`` [T] from a JETTO JSS file, unsigned.

    This is the field ``JPBS`` is normalised to. Returns ``None`` if the file
    carries no ``B0`` -- not all JSS files do.
    """
    from jetto_tools.binary import read_binary_file

    jss = read_binary_file(jss_file)
    if "B0" not in jss:
        return None

    b0 = np.asarray(jss["B0"]).ravel()
    if time is None or "TVEC1" not in jss:
        return float(np.abs(b0[-1]))

    times = np.asarray(jss["TVEC1"]).ravel()
    return float(np.abs(b0[int(np.argmin(np.abs(times - time)))]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kinetics_file",
        nargs="?",
        default=template_dir / "transp.cdf",
        help="Kinetic profiles file (default: the bundled TRANSP template)",
    )
    parser.add_argument(
        "--eq-file",
        default=None,
        help="Equilibrium file. Defaults to the kinetics file, which only works "
        "for formats that hold both (TRANSP, GACODE, IMAS)",
    )
    parser.add_argument(
        "--kinetics-type",
        default=None,
        help="Kinetics format. Inferred from the file if not given",
    )
    parser.add_argument(
        "--eq-type",
        default=None,
        help="Equilibrium format. Inferred from the file if not given",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Time to slice time-dependent files (TRANSP, JETTO) at",
    )
    parser.add_argument(
        "--local-geometry",
        default="MXH",
        help="Local geometry parameterisation used for the fit (default: MXH)",
    )
    parser.add_argument(
        "--psi-n",
        type=float,
        nargs=2,
        default=(0.05, 0.95),
        metavar=("MIN", "MAX"),
        help="Range of normalised poloidal flux to scan (default: 0.05 0.95). "
        "The models degenerate on axis and the geometry fit degrades near the "
        "separatrix, so avoid both ends",
    )
    parser.add_argument(
        "--n-surfaces",
        type=int,
        default=20,
        help="Number of flux surfaces in the scan (default: 20)",
    )
    parser.add_argument(
        "--zeff-scale",
        type=float,
        default=3.0,
        help="Factor to scale Zeff by for the second set of curves "
        "(default: 3.0). Pass 1.0 to plot the baseline only",
    )
    parser.add_argument(
        "--jss-file",
        default=None,
        help="JETTO JSS file supplying B0, so the models and JPBS share a "
        "normalisation. Defaults to a .jss sitting next to the JSP",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save the figure here instead of showing it interactively",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Fall back to the kinetics file for the equilibrium. Only pass --time on to
    # the equilibrium reader when it is reading that same combined file, since a
    # standalone equilibrium (a GEQDSK, say) is already a single time slice.
    eq_file = args.eq_file or args.kinetics_file
    kinetics_kwargs = {} if args.time is None else {"time": args.time}
    eq_kwargs = kinetics_kwargs if args.eq_file is None else {}

    pyro = Pyro(
        eq_file=eq_file,
        eq_type=args.eq_type,
        eq_kwargs=eq_kwargs,
        kinetics_file=args.kinetics_file,
        kinetics_type=args.kinetics_type,
        kinetics_kwargs=kinetics_kwargs,
    )

    # The JETTO run's own answer, where the kinetics file is a JSP
    jetto = None
    b0 = None
    if pyro.kinetics_type == "JETTO":
        try:
            jetto = jetto_bootstrap(args.kinetics_file, args.time)
        except Exception as exc:
            print(f"Could not read JPBS from {args.kinetics_file} ({exc})")

        # B0 puts the models on JPBS's normalisation. Fall back to a .jss
        # alongside the JSP when one was not named explicitly.
        jss_file = args.jss_file
        if jss_file is None:
            sibling = Path(args.kinetics_file).with_suffix(".jss")
            jss_file = sibling if sibling.is_file() else None
        if jss_file is not None:
            try:
                b0 = jetto_b0(jss_file, args.time)
            except Exception as exc:
                print(f"Could not read B0 from {jss_file} ({exc})")
            if b0 is None:
                print(f"No B0 in {jss_file}")
            else:
                print(f"Using B0 = {b0:.4g} T from {jss_file}")

    # Normalisation shared by every curve on the plot
    if b0 is None:
        norm_label = r"\langle B^2\rangle^{1/2}"
        if jetto is not None:
            print(
                "No B0 available: models are normalised to <B^2>^1/2 and JPBS to "
                "B0, so the curves differ by <B^2>^1/2 / B0. Pass --jss-file to "
                "remove this."
            )
    else:
        norm_label = "B_0"
        b0 = b0 * pyro.norms.units.tesla

    psi_ns = np.linspace(*args.psi_n, args.n_surfaces)

    scales = [1.0]
    if not np.isclose(args.zeff_scale, 1.0):
        scales.append(args.zeff_scale)

    # A surface where the geometry fit or the model fails is left as NaN rather
    # than dropped, so every curve stays on the same radial grid.
    columns = [(label, scale) for scale in scales for label in MODELS]
    jbs = {column: np.full_like(psi_ns, np.nan) for column in columns}
    zeffs = np.full_like(psi_ns, np.nan)

    for i, psi_n in enumerate(psi_ns):
        try:
            pyro.load_local(psi_n=psi_n, local_geometry=args.local_geometry)
        except Exception as exc:
            print(f"psi_n = {psi_n:.3f}: could not load local data ({exc})")
            continue

        # zeff is a cached scalar on LocalSpecies, set once by set_zeff() rather
        # than derived on demand, so overriding it here changes exactly what the
        # neoclassical coefficients see and leaves the densities alone.
        zeff_baseline = pyro.local_species.zeff
        zeffs[i] = zeff_baseline.m

        for label, scale in columns:
            pyro.local_species.zeff = scale * zeff_baseline
            try:
                model = MODELS[label](pyro)
                # Jbs is JbsdotB / <B^2>^1/2; divide the moment by B0 instead
                # when B0 is known, so the models match JPBS exactly
                value = model.Jbs if b0 is None else model.JbsdotB / b0
                jbs[(label, scale)][i] = value.to("MA / m**2").m
            except Exception as exc:
                print(f"psi_n = {psi_n:.3f}: {label} at Zeff x{scale:g} failed ({exc})")

        pyro.local_species.zeff = zeff_baseline

    headers = [f"{label.split()[0]} x{scale:g}" for label, scale in columns]
    print(
        f"\n{'psi_n':>6} {'Zeff':>6} | "
        + " ".join(f"{header:>14}" for header in headers)
    )
    print(f"{'':>6} {'':>6} | " + " ".join(f"{'MA/m^2':>14}" for _ in columns))
    print("-" * (16 + 15 * len(columns)))
    for i, psi_n in enumerate(psi_ns):
        print(
            f"{psi_n:6.3f} {zeffs[i]:6.3f} | "
            + " ".join(f"{jbs[column][i]:14.4g}" for column in columns)
        )

    # Report how much the scaled Zeff moved each model, averaged over the scan
    for scale in scales[1:]:
        print(f"\nMean |change| in Jbs from scaling Zeff by {scale:g}:")
        for label in MODELS:
            baseline = jbs[(label, 1.0)]
            scaled = jbs[(label, scale)]
            valid = np.isfinite(baseline) & np.isfinite(scaled)
            change = 100 * np.abs(scaled[valid] / baseline[valid] - 1)
            print(f"  {label:<15} {change.mean():6.2f} %  (max {change.max():6.2f} %)")

    if jetto is not None:
        plt.plot(
            jetto[0],
            jetto[1],
            lw=2,
            color="k",
            label=r"JETTO JPBS ($\langle J_{bs}\cdot B\rangle / B_0$)",
        )

    for label, scale in columns:
        colour = f"C{list(MODELS).index(label)}"
        style = "-" if scale == 1.0 else "--"
        suffix = "" if scale == 1.0 else rf", $Z_{{eff}} \times {scale:g}$"
        plt.plot(
            psi_ns,
            jbs[(label, scale)],
            lw=2,
            ls=style,
            marker="o",
            ms=4,
            color=colour,
            label=f"{label}{suffix}",
        )

    plt.title(rf"Bootstrap current $\langle J_{{bs}}\cdot B\rangle / {norm_label}$")
    plt.xlabel(r"$\psi_N$")
    plt.ylabel(r"$MA\,m^{-2}$")
    plt.grid()
    plt.legend()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
        print(f"\nFigure written to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
