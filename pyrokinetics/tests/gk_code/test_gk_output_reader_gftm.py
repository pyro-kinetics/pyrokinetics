from pyrokinetics.gk_code import GKOutputReaderGFTM
from pyrokinetics import template_dir
from pathlib import Path
import numpy as np
import pytest

from pyrokinetics.tests.gk_code.utils import array_similar, get_golden_answer_data

# TODO mock output tests, similar to GS2


@pytest.fixture(scope="module")
def gftm_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gftm")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGFTM()


@pytest.fixture
def gftm_output_dir(gftm_tmp_path):
    mock_dir = gftm_tmp_path / "mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderGFTM._required_files(mock_dir)
    for required_file in required_files.values():
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def gftm_output_dir_missing_file(gftm_tmp_path):
    mock_dir = gftm_tmp_path / "broken_mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderGFTM._required_files(mock_dir)
    skip = True
    for required_file in required_files.values():
        if skip:
            skip = False
            continue
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def not_gftm_dir(gftm_tmp_path):
    filename = gftm_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


def test_verify_gftm_output(reader, gftm_output_dir):
    # Expect exception to be raised if this fails
    reader.verify(gftm_output_dir)


def test_verify_gftm_missing_file(reader, gftm_output_dir_missing_file):
    with pytest.raises(Exception):
        reader.verify(gftm_output_dir_missing_file)


def test_verify_not_gftm_dir(reader, not_gftm_dir):
    with pytest.raises(Exception):
        reader.verify(not_gftm_dir)


def test_infer_path_from_input_file_gftm():
    input_path = Path("dir/to/input.gftm")
    output_path = GKOutputReaderGFTM.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/")


# Golden answer tests
# Compares against results obtained using GKCode methods from commit 7d551eaa
# This data was gathered from templates/outputs/GFTM_linear

reference_data_commit_hash = "7d551eaa"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gftm_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    ds = get_golden_answer_data(cdf_path)
    request.cls.reference_data = ds


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GFTM_linear"
    request.cls.data = GKOutputReaderGFTM().read(path)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGFTMGoldenAnswers:
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
