from argparse import ArgumentParser, Namespace
from pathlib import Path
from textwrap import dedent

from pyrokinetics import Pyro
from pyrokinetics.units import ureg as units

description = (
    "Generate a gyrokinetics input file from an Equilibrium and Kinetics file."
)


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "target",
        type=str,
        help=dedent(
            f"""\
            The target gyrokinetics code. Options include
            {', '.join(Pyro().supported_gk_inputs)}.
            """
        ),
    )

    parser.add_argument(
        "--geometry",
        "-g",
        type=str,
        help=dedent(
            """\
            The type of flux surface geometry to convert to. Options currently include
            Miller (all), MillerTurnbull (GENE) and MXH (CGYRO, TGLF).
            """
        ),
    )

    parser.add_argument(
        "--equilibrium",
        "--eq",
        "-e",
        type=Path,
        help=dedent(
            f"""\
            Path to a plasma equilibrium file, which is used to overwrite the flux
            surface in 'input_file'. Users should also provide 'psi' to select which
            flux surface to use from the equilibrium. The supported equilibrium types
            are {', '.join(Pyro().supported_equilibrium_types)}.
            """
        ),
    )

    parser.add_argument(
        "--equilibrium_type",
        "--eq_type",
        type=str,
        help="The type of equilibrium file. If not provided, this is inferred.",
    )

    parser.add_argument(
        "--kinetics",
        "-k",
        type=Path,
        help=dedent(
            f"""\
            Path to a plasma kinetics file, which is used to overwrite the local species
            data in 'input_file'. Users should also provide 'psi' and 'a_minor' to
            select which flux surface to use, or provide 'psi' and 'equilibrium'. The
            supported kinetcs types are {', '.join(Pyro().supported_kinetics_types)}.
            """
        ),
    )

    parser.add_argument(
        "--kinetics_type",
        "--k_type",
        type=str,
        help="The type of kinetics file. If not provided, this is inferred.",
    )

    parser.add_argument(
        "--psi",
        "-p",
        type=float,
        help=dedent(
            """\
            The normalised poloidal flux function, used to index which flux surface to
            draw equilibrium/kinetics data from. Should be in the range [0,1], with 0
            being the magnetic axis, and 1 being the last closed flux surface.
            """
        ),
    )

    parser.add_argument(
        "--a_minor",
        "-a",
        type=float,
        help=dedent(
            """\
            The width of the last closed flux surface, in meters. Used to select a flux
            surface when providing kinetics data but no equilibrium. Otherwise, this
            argument is ignored.
            """
        ),
    )

    parser.add_argument(
        "--time",
        type=float,
        help=dedent(
            """\
            Time in seconds from which to take data from in equilibrium/kinetics files
            If not set then final time slice is taken by default
            Appropriate when using data with a time series
            """
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Name of the new gyrokinetics config file.",
    )

    parser.add_argument(
        "--template",
        "-t",
        type=Path,
        help="Template file to use for the new gyrokinetics config file.",
    )

    parser.add_argument(
        "--show_fit",
        type=int,
        help="Shows fit to local geometry when set to 1",
    )


def main(args: Namespace) -> None:
    # Handle illegal combinations of optional args
    if args.equilibrium is None or args.kinetics is None:
        raise ValueError("Must provide an equilibrium file and a kinetics file")

    if args.psi is None:
        raise ValueError("Must provide psi")

    if args.geometry is None:
        raise ValueError("A geometry type must be specified")

    if args.show_fit == 1:
        show_fit = True
    else:
        show_fit = False

    # Create an empty pyro object
    pyro = Pyro()

    # Load local geometry
    if args.equilibrium is not None:
        if args.equilibrium_type == "GEQDSK":
            pyro.load_global_eq(eq_file=args.equilibrium, eq_type=args.equilibrium_type)
        else:
            pyro.load_global_eq(
                eq_file=args.equilibrium, eq_type=args.equilibrium_type, time=args.time
            )
        pyro.load_local_geometry(
            psi_n=args.psi, local_geometry=args.geometry, show_fit=show_fit
        )

    # Load local species
    if args.kinetics is not None:
        pyro.load_global_kinetics(
            kinetics_file=args.kinetics,
            kinetics_type=args.kinetics_type,
            time=args.time,
        )
        pyro.load_local_species(
            psi_n=args.psi,
            a_minor=(args.a_minor * units.meter if args.equilibrium is None else None),
        )

    # Convert and write
    filename = f"input.{args.target}".lower() if args.output is None else args.output
    pyro.write_gk_file(filename, gk_code=args.target, template_file=args.template)
