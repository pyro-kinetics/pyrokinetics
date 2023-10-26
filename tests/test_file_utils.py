import pytest

from pyrokinetics.file_utils import (
    FileReader,
    ReadableFromFile,
)
from pyrokinetics.typing import PathLike


class MyReadable(ReadableFromFile):
    """Defines a minimal readable object"""

    def __init__(self, data: str):
        self.data = data


class MyReader(FileReader, file_type="MyReader", reads=MyReadable):
    """Defines a minimal reader object"""

    def read_from_file(self, filename: PathLike) -> MyReadable:
        with open(filename, "r") as f:
            return MyReadable(f.read())


@pytest.fixture
def example_input_file(tmp_path):
    d = tmp_path / "test_file_utils"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "example_input_file.txt"
    with open(path, "w") as f:
        f.write("hello world")
    return path


def test_supported_file_types():
    """Test that a Reader has been successfully registered"""
    assert "MyReader" in MyReadable.supported_file_types()


@pytest.mark.parametrize("file_type", ("MyReader", None))
def test_from_file(example_input_file, file_type):
    readable = MyReadable.from_file(example_input_file)
    assert readable.data == "hello world"
