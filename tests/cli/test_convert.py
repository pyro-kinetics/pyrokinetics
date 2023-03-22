import sys
from difflib import unified_diff
from itertools import product
from pathlib import Path
from typing import Optional

import pytest

import pyrokinetics as pk
from pyrokinetics.cli import entrypoint


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


def convert_with_python(
    gk_input: str,
    gk_output: str,
    output_dir: Path,
    eq_file: Optional[Path] = None,
    kinetics_file: Optional[Path] = None,
    local_geometry: Optional[str] = None,
    explicit_types: bool = False,
    base_case: bool = False,
) -> Path:
    """
    Performs conversion via Pyro object within Python
    """
    file_in = pk.gk_templates[gk_input]
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
        eq_file=eq_file,
        eq_type=eq_type,
        kinetics_file=kinetics_file,
        kinetics_type=k_type,
    )
    if eq_file is not None:
        pyro.load_local_geometry(psi_n=0.9)
    if kinetics_file is not None:
        pyro.load_local_species(psi_n=0.9, a_minor=3.0 if eq_file is None else None)
    if local_geometry is not None:
        pyro.switch_local_geometry(local_geometry)

    if base_case:
        file_out = output_dir / f"result_base.{gk_output.lower()}"
    else:
        file_out = output_dir / f"result.{gk_output.lower()}"
    pyro.write_gk_file(file_out, gk_code=gk_output)
    return file_out


def convert_with_cli(
    gk_input: str,
    gk_output: str,
    output_dir: Path,
    eq_file: Optional[Path] = None,
    kinetics_file: Optional[Path] = None,
    local_geometry: Optional[str] = None,
    long_opts: bool = False,
    explicit_types: bool = False,
) -> Path:
    """
    Performs conversion via CLI convert tool
    """
    file_in = pk.gk_templates[gk_input]
    file_out = output_dir / f"result_cli.{gk_output.lower()}"
    with pytest.MonkeyPatch.context() as m:
        argv = ["pyro", "convert", gk_output, str(file_in)]
        argv.extend([opt("o", long_opts), str(file_out)])
        if explicit_types:
            argv.extend(["--input_type", str(gk_input)])
        if eq_file is not None:
            argv.extend([opt("e", long_opts), str(eq_file)])
            argv.extend([opt("p", long_opts), str(0.9)])
            if explicit_types:
                argv.extend(["--eq_type", "GEQDSK"])
        if kinetics_file is not None:
            argv.extend([opt("k", long_opts), str(kinetics_file)])
            if explicit_types:
                argv.extend(["--kinetics_type", "JETTO"])
            if eq_file is None:
                argv.extend([opt("p", long_opts), str(0.9)])
                argv.extend([opt("a", long_opts), str(3.0)])
        if local_geometry is not None:
            argv.extend([opt("g", long_opts), local_geometry])
        m.setattr(sys, "argv", argv)
        entrypoint()
    return file_out


def files_match(result: Path, expected: Path, base_case: Optional[Path] = None):
    # Assert the files match
    # Is this condition too strict? Not sure how else to test without writing
    # individual tests for each combination of gk_input and gk_output
    with open(result) as f_result, open(expected) as f_expected:
        result_lines = f_result.readlines()
        expected_lines = f_expected.readlines()
    diff = [*unified_diff(result_lines, expected_lines)]
    assert not diff

    if base_case is not None:
        # Assert the files don't match the base case.
        # This should be used when testing with optional features, to ensure they
        # are having an affect on the results
        with open(base_case) as f_base:
            base_lines = f_base.readlines()
        base_diff = [*unified_diff(result_lines, base_lines)]
        assert base_diff


@pytest.mark.parametrize(
    "gk_input,gk_output",
    product(
        pk.Pyro().supported_gk_inputs,
        pk.Pyro().supported_gk_inputs,
    ),
)
def test_convert(gk_input, gk_output, tmp_path):
    # Create directory to work in
    d = tmp_path / "test_convert"
    d.mkdir(exist_ok=True)

    # Perform conversions, assert files match
    file_out = convert_with_python(gk_input, gk_output, output_dir=d)
    file_out_cli = convert_with_cli(gk_input, gk_output, output_dir=d)
    files_match(file_out, file_out_cli)


@pytest.mark.parametrize("eq", ["GEQDSK", "TRANSP"])
def test_convert_with_eq(eq, tmp_path):
    # Create directory to work in
    d = tmp_path / "test_convert_with_eq"
    d.mkdir(exist_ok=True)

    eq_file = pk.eq_templates[eq]

    # Perform conversions, assert files match
    file_out = convert_with_python("GS2", "CGYRO", output_dir=d, eq_file=eq_file)
    file_out_cli = convert_with_cli("GS2", "CGYRO", output_dir=d, eq_file=eq_file)
    file_out_base = convert_with_python("GS2", "CGYRO", output_dir=d, base_case=True)
    files_match(file_out, file_out_cli, base_case=file_out_base)


@pytest.mark.parametrize(
    "kinetics,long_opts",
    product(
        ("JETTO", "SCENE", "TRANSP"),
        (True, False),
    ),
)
def test_convert_with_kinetics(kinetics, long_opts, tmp_path):
    # Note: this tests over long_opts as it's the only test where a_minor is used
    # Create directory to work in
    d = tmp_path / "test_convert_with_kinetics"
    d.mkdir(exist_ok=True)

    kinetics_file = pk.kinetics_templates[kinetics]

    # Perform conversions, assert files match
    file_out = convert_with_python(
        "GS2", "CGYRO", output_dir=d, kinetics_file=kinetics_file
    )
    file_out_cli = convert_with_cli(
        "GS2", "CGYRO", output_dir=d, kinetics_file=kinetics_file, long_opts=long_opts
    )
    file_out_base = convert_with_python("GS2", "CGYRO", output_dir=d, base_case=True)
    files_match(file_out, file_out_cli, base_case=file_out_base)


@pytest.mark.parametrize("kinetics", ["JETTO", "SCENE", "TRANSP"])
def test_convert_with_kinetics_and_eq(kinetics, tmp_path):
    # Create directory to work in
    d = tmp_path / "test_convert_with_kinetics_and_eq"
    d.mkdir(exist_ok=True)

    eq_file = pk.eq_templates["GEQDSK"]
    kinetics_file = pk.kinetics_templates[kinetics]

    # Perform conversions, assert files match
    file_out = convert_with_python(
        "GS2", "CGYRO", output_dir=d, eq_file=eq_file, kinetics_file=kinetics_file
    )
    file_out_cli = convert_with_cli(
        "GS2", "CGYRO", output_dir=d, eq_file=eq_file, kinetics_file=kinetics_file
    )
    file_out_base = convert_with_python("GS2", "CGYRO", output_dir=d, base_case=True)
    files_match(file_out, file_out_cli, base_case=file_out_base)


def test_convert_with_geometry(tmp_path):
    # Create directory to work in
    d = tmp_path / "test_convert_with_geometry"
    d.mkdir(exist_ok=True)

    # Perform conversions, assert files match
    file_out = convert_with_python("GS2", "CGYRO", output_dir=d, local_geometry="MXH")
    file_out_cli = convert_with_cli("GS2", "CGYRO", output_dir=d, local_geometry="MXH")
    file_out_base = convert_with_python("GS2", "CGYRO", output_dir=d, base_case=True)
    files_match(file_out, file_out_cli, base_case=file_out_base)


@pytest.mark.parametrize("long_opts", (True, False))
def test_convert_all_opts(long_opts, tmp_path):
    # Create directory to work in
    d = tmp_path / "test_convert_all_opts"
    d.mkdir(exist_ok=True)

    eq_file = pk.eq_templates["GEQDSK"]
    kinetics_file = pk.kinetics_templates["JETTO"]

    # Perform conversions, assert files match
    file_out = convert_with_python(
        "GS2",
        "CGYRO",
        output_dir=d,
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        local_geometry="MXH",
        explicit_types=True,
    )
    file_out_cli = convert_with_cli(
        "GS2",
        "CGYRO",
        output_dir=d,
        eq_file=eq_file,
        kinetics_file=kinetics_file,
        local_geometry="MXH",
        long_opts=True,
        explicit_types=True,
    )
    file_out_base = convert_with_python("GS2", "CGYRO", output_dir=d, base_case=True)
    files_match(file_out, file_out_cli, base_case=file_out_base)
