from argparse import ArgumentParser, Namespace
from pathlib import Path
from textwrap import dedent

from pyrokinetics import Pyro

description = "Convert a gyrokinetics input file to a different code."


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "target",
        type=str,
        help="The target gyrokinetics code, e.g. 'GS2', 'CGYRO'.",
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="The gyrokinetics config file you wish to convert.",
    )

    parser.add_argument(
        "--input_type",
        type=str,
        help="The code type of the input file. If not provided, this will be inferred.",
    )

    parser.add_argument(
        "--equilibrium",
        "--eq",
        "-e",
        type=Path,
        help=dedent(
            """\
            Path to a plasma equilibrium file, which is used to overwrite the flux
            surface in 'input_file'. Users should also provide 'psi' to select which
            flux surface to use from the equilibrium.
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
            """\
            Path to a plasma kinetics file, which is used to overwrite the local species
            data in 'input_file'. Users should also provide 'psi' and 'a_minor' to
            select which flux surface to use, or provide 'psi' and 'equilibrium'.
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


def main(args: Namespace) -> None:
    # Handle illegal combinations of optional args
    if args.equilibrium is not None and args.psi is None:
        raise ValueError("If providing an equilibrium file, must also provide psi")

    if args.kinetics is not None and args.psi is None:
        raise ValueError("If providing a kinetics file, must also provide psi")

    if args.kinetics is not None and args.equilibrium is None and args.a_minor is None:
        raise ValueError(
            "If providing a kinetics file without an equilibrium, "
            "must also provide a_minor"
        )

    # Create a pyro object with just gk info
    pyro = Pyro(gk_file=args.input_file, gk_code=args.input_type)

    # Modify local geometry
    if args.equilibrium is not None:
        pyro.load_global_eq(eq_file=args.equilibrium, eq_type=args.equilibrium_type)
        pyro.load_local_geometry(psi_n=args.psi)

    # Modify local species
    if args.kinetics is not None:
        pyro.load_global_kinetics(
            kinetics_file=args.kinetics,
            kinetics_type=args.kinetics_type,
        )
        pyro.load_local_species(
            psi_n=args.psi,
            a_minor=(args.a_minor if args.equilibrium is None else None),
        )

    # Convert and write
    filename = f"input.{args.target}".lower() if args.output is None else args.output
    pyro.write_gk_file(filename, gk_code=args.target, template_file=args.template)
