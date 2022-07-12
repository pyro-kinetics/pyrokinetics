from pyrokinetics.gk_code import GKOutputReaderTGLF
from pyrokinetics import template_dir
from pathlib import Path
import numpy as np
import pytest

from pyrokinetics.tests.gk_code.utils import array_similar, get_golden_answer_data

# TODO mock output tests, similar to GS2


@pytest.fixture(scope="module")
def tglf_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_tglf")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderTGLF()


@pytest.fixture
def tglf_output_dir(tglf_tmp_path):
    mock_dir = tglf_tmp_path / "mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderTGLF._required_files(mock_dir)
    for required_file in required_files.values():
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def tglf_output_dir_missing_file(tglf_tmp_path):
    mock_dir = tglf_tmp_path / "broken_mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderTGLF._required_files(mock_dir)
    skip = True
    for required_file in required_files.values():
        if skip:
            skip = False
            continue
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def not_tglf_dir(tglf_tmp_path):
    filename = tglf_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


def test_verify_tglf_output(reader, tglf_output_dir):
    # Expect exception to be raised if this fails
    reader.verify(tglf_output_dir)


def test_verify_tglf_missing_file(reader, tglf_output_dir_missing_file):
    with pytest.raises(Exception):
        reader.verify(tglf_output_dir_missing_file)


def test_verify_not_tglf_dir(reader, not_tglf_dir):
    with pytest.raises(Exception):
        reader.verify(not_tglf_dir)


def test_infer_path_from_input_file_tglf():
    input_path = Path("dir/to/input.tglf")
    output_path = GKOutputReaderTGLF.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/")


# Golden answer tests
# Compares against results obtained using GKCode methods from commit 7d551eaa
# This data was gathered from templates/outputs/TGLF_linear

reference_data_commit_hash = "7d551eaa"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"tglf_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    ds = get_golden_answer_data(cdf_path)
    request.cls.reference_data = ds


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "TGLF_linear"
    request.cls.data = GKOutputReaderTGLF().read(path)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestTGLFGoldenAnswers:
    def test_coords(self):
        """
        Ensure that all reference coords are present in data
        """
        for c in self.reference_data.coords:
            dtype = self.reference_data[c].dtype
            if dtype == "float64" or dtype == "complex128":
                assert array_similar(self.reference_data[c], self.data[c])
            else:
                assert np.array_equal(self.reference_data[c], self.data[c])

    @pytest.mark.parametrize(
        "var",
        [
            "eigenfunctions",
            "growth_rate",
            "mode_frequency",
        ],
    )
    def test_data_vars(self, var):
        assert array_similar(self.reference_data[var], self.data[var])
