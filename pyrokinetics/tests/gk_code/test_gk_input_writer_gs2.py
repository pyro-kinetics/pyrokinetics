from pyrokinetics.gk_code import GKInputReaderGS2, GKInputWriterGS2
from pyrokinetics import template_dir
from pathlib import Path
import pytest

template_file = template_dir.joinpath("input.gs2")


@pytest.fixture
def gs2_reader():
    return GKInputReaderGS2(template_file)


def test_write(tmp_path, gs2_reader):
    """Ensure a gs2 file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gs2_reader.get_local_geometry()
    local_species = gs2_reader.get_local_species()
    numerics = gs2_reader.get_numerics()
    # Set output path
    filename = tmp_path / "input.in"
    # Write out a new input file
    GKInputWriterGS2().write( filename, local_geometry, local_species, numerics)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputReaderGS2().verify(filename)
    reader = GKInputReaderGS2(filename)
    new_local_geometry = reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time
