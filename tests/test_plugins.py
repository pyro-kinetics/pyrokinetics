from pathlib import Path

import pytest
from pyrokinetics_plugin_examples import make_plugin_example_file

from pyrokinetics.equilibrium import (
    Equilibrium,
    supported_equilibrium_types,
    read_equilibrium,
)
from pyrokinetics.kinetics import Kinetics, supported_kinetics_types, read_kinetics
from pyrokinetics.gk_code import (
    GKInput,
    GKOutput,
    supported_gk_input_types,
    supported_gk_output_types,
    read_gk_input,
    read_gk_output,
)
from pyrokinetics.normalisation import SimulationNormalisation


@pytest.fixture
def plugin_file(tmp_path: Path) -> Path:
    """
    The pyrokinetics-plugin-examples repository contains a number of different
    plugin implementations, including variants for Equilibrium, Kinetics,
    GKInput, and GKOutput. However, all share the same input file. This fixture
    creates a new copy of this universal input file.
    """
    d = tmp_path / "plugin_test_files"
    d.mkdir(parents=True, exist_ok=True)
    filename = d / "universal_input_file.txt"
    make_plugin_example_file(filename)
    return filename


def test_equilibrium_plugin(plugin_file: Path) -> None:
    assert "_test" in supported_equilibrium_types()
    assert isinstance(read_equilibrium(plugin_file), Equilibrium)


def test_kinetics_plugin(plugin_file: Path) -> None:
    assert "_test" in supported_kinetics_types()
    assert isinstance(read_kinetics(plugin_file), Kinetics)


def test_gk_input_plugin(plugin_file: Path) -> None:
    assert "_test" in supported_gk_input_types()
    assert isinstance(read_gk_input(plugin_file), GKInput)


def test_gk_output_plugin(plugin_file: Path) -> None:
    assert "_test" in supported_gk_output_types()
    norm = SimulationNormalisation("_test", convention="pyrokinetics")
    assert isinstance(read_gk_output(plugin_file, norm), GKOutput)
