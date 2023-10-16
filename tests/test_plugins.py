from pathlib import Path

import pytest

from pyrokinetics.equilibrium import (
    Equilibrium,
    read_equilibrium,
    supported_equilibrium_types,
)
from pyrokinetics.gk_code import (
    GKInput,
    GKOutput,
    read_gk_input,
    read_gk_output,
    supported_gk_input_types,
    supported_gk_output_types,
)
from pyrokinetics.kinetics import Kinetics, read_kinetics, supported_kinetics_types
from pyrokinetics.normalisation import SimulationNormalisation

# Skip tests if plugin library is missing
try:
    from pyrokinetics_plugin_examples import make_plugin_example_file
except ImportError:
    pytest.skip("Plugin test library not found.", allow_module_level=True)


@pytest.fixture
def plugin_file(tmp_path: Path) -> Path:
    """The pyrokinetics-plugin-examples repository contains a number of different plugin
    implementations, including variants for Equilibrium, Kinetics, GKInput, and
    GKOutput.

    However, all share the same input file. This fixture creates a new copy of this
    universal input file.
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
