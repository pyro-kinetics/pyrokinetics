from pyrokinetics.gk_code import GKOutputReaderGS2
import pytest
import pathlib

@pytest.fixture
def reader():
    return GKOutputReaderGS2()

@pytest.fixture
def nc():
    return pathlib.Path(__file__).parent.parent / "test_files" / "test_fields_111_fluxes_111.out.nc"

def test_read(reader, nc):
    output = reader.read(nc)

