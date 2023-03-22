import sys
from difflib import unified_diff

import pytest

import pyrokinetics as pk
from pyrokinetics.cli import entrypoint


_geometries = {
    "GS2": "Miller",
    "CGYRO": "MXH",
    "TGLF": "MXH",
    "GENE": "MillerTurnbull",
}

_long_opts = {
    "a": "a_minor",
    "e": "equilibrium",
    "g": "geometry",
    "k": "kinetics",
    "p": "psi",
    "o": "output",
}


def opt(name: str, long: bool):
    return f"--{_long_opts[name]}" if long else f"-{name}"


eq_type = "GEQDSK"
eq_file = pk.eq_templates[eq_type]
k_type = "JETTO"
k_file = pk.kinetics_templates[k_type]


@pytest.mark.parametrize(
    "gk_input,gk_output,eq,kinetics,switch_geometry,long_opts,explicit_types",
    [
        ("GENE", "GS2", None, None, False, False, False),
        ("GS2", "GENE", eq_file, k_file, True, False, False),
        ("GS2", "CGYRO", None, None, True, False, True),
        ("CGYRO", "GENE", None, k_file, False, True, False),
        ("CGYRO", "GS2", eq_file, None, True, True, True),
        ("CGYRO", "TGLF", None, None, False, False, False),
        ("TGLF", "GS2", eq_file, k_file, False, False, False),
    ],
)
def test_convert(
    gk_input,
    gk_output,
    eq,
    kinetics,
    switch_geometry,
    long_opts,
    explicit_types,
    tmp_path,
):
    # TODO Tests only a small subset of all possible combinations. The whole matrix
    #      can take up to 10 minutes!
    # TODO Not testing template options
    # TODO Not testing with TGLF while switching geomtry to MXH, as this functionality
    #      seems to be bugged at the time of writing.
    # Create directory to work in
    d = tmp_path / "test_convert"
    d.mkdir(exist_ok=True)

    # Get input file and output file names
    file_in = pk.gk_templates[gk_input]
    file_out = d / f"result.{gk_output.lower()}"
    file_out_cli = d / f"result_cli.{gk_output.lower()}"

    # Get input types, if needed
    if explicit_types:
        gk_type = gk_input
        eq_type = "GEQDSK"
        k_type = "JETTO"
    else:
        gk_type = None
        eq_type = None
        k_type = None

    # Perform conversion using Python API
    pyro = pk.Pyro(
        gk_file=file_in,
        gk_code=gk_type,
        eq_file=eq,
        eq_type=eq_type,
        kinetics_file=kinetics,
        kinetics_type=k_type,
    )
    if eq is not None:
        pyro.load_local_geometry(psi_n=0.9)
    if kinetics is not None:
        pyro.load_local_species(psi_n=0.9, a_minor=3.0 if eq is None else None)
    if switch_geometry:
        geometry = _geometries[gk_output]
        pyro.switch_local_geometry(geometry)
    pyro.write_gk_file(file_out, gk_code=gk_output)

    # Perform conversion using cli tool
    with pytest.MonkeyPatch.context() as m:
        argv = ["pyro", "convert", gk_output, str(file_in)]
        argv.extend([opt("o", long_opts), str(file_out_cli)])
        if explicit_types:
            argv.extend(["--input_type", str(gk_input)])
        if eq is not None:
            argv.extend([opt("e", long_opts), str(eq)])
            argv.extend([opt("p", long_opts), str(0.9)])
            if explicit_types:
                argv.extend(["--eq_type", "GEQDSK"])
        if kinetics is not None:
            argv.extend([opt("k", long_opts), str(kinetics)])
            if explicit_types:
                argv.extend(["--kinetics_type", "JETTO"])
            if eq is None:
                argv.extend([opt("p", long_opts), str(0.9)])
                argv.extend([opt("a", long_opts), str(3.0)])
        if switch_geometry:
            argv.extend([opt("g", long_opts), geometry])
        m.setattr(sys, "argv", argv)
        entrypoint()

    # Expect the files match
    # Is this condition too strict? Not sure how else to test without writing
    # individual tests for each combination of gk in, gk out, eq, kinetics, and geometry
    with open(file_out_cli) as f_results, open(file_out) as f_expected:
        results = f_results.readlines()
        expected = f_expected.readlines()
    diff = [*unified_diff(results, expected)]
    assert not diff
