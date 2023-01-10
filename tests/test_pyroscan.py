from pyrokinetics.pyroscan import PyroScan
from pyrokinetics.examples import example_SCENE
from pyrokinetics import Pyro

from pathlib import Path
import numpy as np


def assert_close_or_equal(attr, left_pyroscan, right_pyroscan):
    left = getattr(left_pyroscan, attr)
    right = getattr(right_pyroscan, attr)

    if isinstance(left, (str, list, type(None), dict, Path)):
        assert left == right
    else:
        assert np.allclose(left, right), f"{left} != {right}"


def test_compare_read_write_pyroscan(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    parameter_dict = {"ky": [0.1, 0.2, 0.3]}

    initial_pyroscan = PyroScan(pyro, parameter_dict=parameter_dict)

    initial_pyroscan.write(file_name="test_pyroscan.input", base_directory=tmp_path)

    pyroscan_json = tmp_path / "pyroscan.json"

    new_pyroscan = PyroScan(pyro, pyroscan_json=pyroscan_json)

    comparison_attrs = [
        "base_directory",
        "file_name",
        "p_prime_type",
        "parameter_dict",
        "parameter_map",
        "parameter_separator",
        "pyroscan_json",
        "run_directories",
        "value_fmt",
        "value_size",
    ]
    for attrs in comparison_attrs:
        assert_close_or_equal(attrs, initial_pyroscan, new_pyroscan)


def test_format_run_name():
    scan = PyroScan(Pyro(gk_code="GS2"), value_separator="|", parameter_separator="@")

    assert scan.format_single_run_name({"ky": 0.1, "nx": 55}) == "ky|0.10@nx|55.00"


def test_create_single_run():
    scan = PyroScan(
        Pyro(gk_code="GS2"),
        base_directory="some_dir",
        value_separator="|",
        parameter_separator="@",
    )

    run_parameters = {"ky": 0.1, "nx": 55}
    name, new_run = scan.create_single_run(run_parameters)

    assert name == scan.format_single_run_name(run_parameters)
    assert new_run.file_name == "input.in"
    assert new_run.run_directory == Path(f"./some_dir/{name}").absolute()
    assert new_run.run_parameters == run_parameters
