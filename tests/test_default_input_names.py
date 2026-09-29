import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.pyroscan import PyroScan


@pytest.mark.parametrize("code", ["GFTM", "TGLF"])
def test_default_file_name_is_lowercase(code, tmp_path):
    """Default deck name must match what the output readers look up (input.<code>)."""
    pyro = Pyro(gk_file=template_dir / "input.gs2")
    pyro.convert_gk_code(code)
    ps = PyroScan(pyro, {"ky": [0.1, 0.2]}, base_directory=tmp_path)
    ps.write()
    decks = sorted(
        p.relative_to(tmp_path) for p in tmp_path.rglob(f"input.{code.lower()}")
    )
    assert len(decks) == 2
    assert not list(tmp_path.rglob(f"input.{code}"))

    # Reader's required-file lookup finds the deck written by the scan
    reader = Pyro(gk_file=tmp_path / decks[0]).gk_input
    assert reader is not None


@pytest.mark.parametrize("code", ["GFTM", "TGLF"])
def test_old_uppercase_file_name_json_still_loads(code, tmp_path):
    """A pyroscan.json written before the rename carries its own file_name."""
    pyro = Pyro(gk_file=template_dir / "input.gs2")
    pyro.convert_gk_code(code)
    ps = PyroScan(pyro, {"ky": [0.1, 0.2]}, base_directory=tmp_path)
    ps.write(file_name=f"input.{code}")
    reloaded = PyroScan(
        pyro, pyroscan_json=tmp_path / "pyroscan.json", base_directory=tmp_path
    )
    assert reloaded.file_name == f"input.{code}"
