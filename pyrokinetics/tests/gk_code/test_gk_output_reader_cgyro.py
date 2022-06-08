from pyrokinetics.gk_code import GKOutputReaderCGYRO, GKInputCGYRO
from itertools import product, combinations
from pathlib import Path
import xarray as xr
import numpy as np
import pytest

#FIXME Currently not testing most of this class! So far only testing that 'verify' works
#      as intended, as going beyond this will require both a minimal set of test files
#      of size <<1MB, and a complex mocking setup.

@pytest.fixture(scope="module")
def cgyro_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_cgyro")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderCGYRO()


@pytest.fixture
def cgyro_output_dir(cgyro_tmp_path):
    mock_dir = cgyro_tmp_path / "mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderCGYRO._required_files(mock_dir)
    for required_file in required_files.values():
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def cgyro_output_dir_missing_file(cgyro_tmp_path):
    mock_dir = cgyro_tmp_path / "broken_mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderCGYRO._required_files(mock_dir)
    skip = True
    for required_file in required_files.values():
        if skip:
            skip = False
            continue
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def not_cgyro_dir(cgyro_tmp_path):
    filename = cgyro_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


def test_verify_cgyro_output(reader, cgyro_output_dir):
    # Expect exception to be raised if this fails
    reader.verify(cgyro_output_dir)


def test_verify_cgyro_missing_file(reader, cgyro_output_dir_missing_file):
    with pytest.raises(Exception):
        reader.verify(cgyro_output_dir_missing_file)


def test_verify_not_cgyro_dir(reader, not_cgyro_dir):
    with pytest.raises(Exception):
        reader.verify(not_cgyro_dir)
