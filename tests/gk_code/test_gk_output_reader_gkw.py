from pyrokinetics.gk_code import GKOutputReaderGKW
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics import template_dir
from pyrokinetics.normalisation import SimulationNormalisation as Normalisation
from pathlib import Path
import numpy as np
import pytest


# TODO mock output tests, similar to GS2


@pytest.fixture(scope="module")
def gkw_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gkw")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGKW()


@pytest.fixture
def gkw_output_dir(gkw_tmp_path):
    mock_dir = gkw_tmp_path / "mock_dir"
    mock_dir.mkdir()
    required_files = GKOutputReaderGKW._required_files(mock_dir)
    for required_file in required_files.values():
        with open(required_file.path, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def gkw_output_dir_missing_input(gkw_tmp_path):
    mock_dir = gkw_tmp_path / "broken_mock_dir"
    mock_dir.mkdir()
    for f in [mock_dir / f for f in ["geom.dat", "time.dat"]]:
        with open(f, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def empty_gkw_dir(gkw_tmp_path):
    mock_dir = gkw_tmp_path / "empty_dir"
    mock_dir.mkdir()
    return mock_dir


@pytest.fixture
def not_gkw_file(gkw_tmp_path):
    mock_dir = gkw_tmp_path / "nongkw_dir"
    mock_dir.mkdir()
    filename = mock_dir / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


def test_verify_gkw_output(reader, gkw_output_dir):
    # Expect exception to be raised if this fails
    reader.verify_file_type(gkw_output_dir)


def test_verify_gkw_missing_input(reader, gkw_output_dir_missing_input):
    with pytest.raises(Exception):
        reader.verify_file_type(gkw_output_dir_missing_input)


def test_verify_not_gkw_dir(reader, empty_gkw_dir):
    with pytest.raises(Exception):
        reader.verify_file_type(empty_gkw_dir)


def test_verify_not_gkw_file(reader, not_gkw_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_gkw_file)


# Golden answer tests
# This data was gathered from templates/outputs/GKW_linear

reference_data_commit_hash = "beb68100"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gkw_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GKW_linear"
    norm = Normalisation("test_gk_output_gkw")

    request.cls.data = GKOutputReaderGKW().read_from_file(
        path, norm=norm, output_convention="GKW"
    )


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGKWGoldenAnswers:
    def test_coords(self, array_similar):
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
            "phi",
            "particle",
            "momentum",
            "heat",
            "eigenvalues",
            "eigenfunctions",
            "growth_rate",
            "mode_frequency",
            "growth_rate_tolerance",
        ],
    )
    def test_data_vars(self, array_similar, var):
        assert array_similar(self.reference_data[var], self.data[var])

    @pytest.mark.parametrize(
        "attr",
        [
            "linear",
            "gk_code",
            "input_file",
            "attribute_units",
            "title",
        ],
    )
    def test_data_attrs(self, attr):
        if isinstance(getattr(self.reference_data, attr), float):
            assert np.isclose(
                getattr(self.reference_data, attr), getattr(self.data, attr)
            )
        else:
            assert getattr(self.reference_data, attr) == getattr(self.data, attr)
