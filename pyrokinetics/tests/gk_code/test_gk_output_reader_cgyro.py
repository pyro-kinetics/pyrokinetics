from pyrokinetics.gk_code import GKOutputReaderCGYRO
from pyrokinetics import template_dir
from pathlib import Path
import numpy as np
import pytest

from pyrokinetics.tests.gk_code.utils import array_similar, get_golden_answer_data

# TODO mock output tests, similar to GS2


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


def test_infer_path_from_input_file_cgyro():
    input_path = Path("dir/to/input.cgyro")
    output_path = GKOutputReaderCGYRO.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/")


# Golden answer tests
# Compares against results obtained using GKCode methods from commit 7d551eaa
# This data was gathered from templates/outputs/CGYRO_linear

reference_data_commit_hash = "7d551eaa"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"cgyro_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    ds = get_golden_answer_data(cdf_path)
    request.cls.reference_data = ds


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "CGYRO_linear"
    request.cls.data = GKOutputReaderCGYRO().read(path)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestCGYROGoldenAnswers:
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
            "fields",
            "fluxes",
            "eigenvalues",
            "eigenfunctions",
            "growth_rate",
            "growth_rate_tolerance",
            "mode_frequency",
        ],
    )
    def test_data_vars(self, var):
        assert array_similar(self.reference_data[var], self.data[var])
