from pyrokinetics.gk_code import GKOutputReaderCGYRO
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics import template_dir, Pyro
from pathlib import Path
import numpy as np
import pytest

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
    reader.verify_file_type(cgyro_output_dir)


def test_verify_cgyro_missing_file(reader, cgyro_output_dir_missing_file):
    with pytest.raises(Exception):
        reader.verify_file_type(cgyro_output_dir_missing_file)


def test_verify_not_cgyro_dir(reader, not_cgyro_dir):
    with pytest.raises(Exception):
        reader.verify_file_type(not_cgyro_dir)


def test_infer_path_from_input_file_cgyro():
    input_path = Path("dir/to/input.cgyro")
    output_path = GKOutputReaderCGYRO.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/")


# Golden answer tests
# This data was gathered from templates/outputs/CGYRO_linear

reference_data_commit_hash = "54f1d7d1"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"cgyro_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "CGYRO_linear"

    pyro = Pyro(gk_file=path / "input.cgyro", name="test_gk_output_cgyro")
    norm = pyro.norms

    request.cls.data = GKOutputReaderCGYRO().read_from_file(path, norm=norm)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestCGYROGoldenAnswers:
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
