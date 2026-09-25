import pytest

from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputNEO

template_file = template_dir / "input.neo"


@pytest.fixture
def neo():
    return GKInputNEO(template_file)


def test_add_flags(neo):
    neo.add_flags({"foo": "bar"})
    assert neo.data["FOO"] == "bar"


@pytest.mark.parametrize("key", ["N_ENERGY", "n_energy", "N_Energy"])
def test_add_flags_case_insensitive(tmp_path, neo, key):
    neo.add_flags({key: 6, "new_flag": 1})
    assert neo.data["N_ENERGY"] == 6
    assert neo.data["NEW_FLAG"] == 1
    assert [k for k in neo.data if k.upper() == "N_ENERGY"] == ["N_ENERGY"]

    filename = tmp_path / "input.neo"
    neo.write(filename)
    with open(filename) as f:
        lines = [line for line in f if line.upper().startswith("N_ENERGY")]
    assert lines == ["N_ENERGY = 6\n"]
