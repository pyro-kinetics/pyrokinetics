from pathlib import Path

import pytest
from pyrokinetics_plugin_examples import make_plugin_example_file

from pyrokinetics import Equilibrium, supported_equilibrium_types, read_equilibrium


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
