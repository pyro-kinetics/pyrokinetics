from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.gk_code import GKOutputReaderTGLF
from pyrokinetics.gk_code.gk_output import GKOutput

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


def test_verify_file_type_tglf_output(reader, tglf_output_dir):
    # Expect exception to be raised if this fails
    reader.verify_file_type(tglf_output_dir)


def test_verify_tglf_missing_file(reader, tglf_output_dir_missing_file):
    with pytest.raises(Exception):
        reader.verify_file_type(tglf_output_dir_missing_file)


def test_verify_not_tglf_dir(reader, not_tglf_dir):
    with pytest.raises(Exception):
        reader.verify_file_type(not_tglf_dir)


def test_infer_path_from_input_file_tglf():
    input_path = Path("dir/to/input.tglf")
    output_path = GKOutputReaderTGLF.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/")


def test_read_tglf_transport():
    path = template_dir / "outputs" / "TGLF_transport"
    pyro = Pyro(gk_file=path / "input.tglf", name="test_gk_output_tglf_transport")
    pyro.load_gk_output()
    assert isinstance(pyro.gk_output, GKOutput)


# Golden answer tests
# This data was gathered from templates/outputs/TGLF_linear

reference_data_commit_hash = "24fd3e6d"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"tglf_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "TGLF_linear"
    pyro = Pyro(gk_file=path / "input.tglf", name="test_gk_output_tglf")
    norm = pyro.norms

    request.cls.data = GKOutputReaderTGLF().read_from_file(path, norm=norm)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestTGLFGoldenAnswers:
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
            "eigenfunctions",
            "growth_rate",
            "mode_frequency",
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


@pytest.mark.parametrize("load_fields", [True, False])
def test_amplitude(load_fields):

    path = template_dir / "outputs" / "TGLF_linear"

    pyro = Pyro(gk_file=path / "input.tglf")

    pyro.load_gk_output(load_fields=load_fields)
    eigenfunctions = pyro.gk_output.data["eigenfunctions"].isel(mode=0)
    field_squared = np.abs(eigenfunctions) ** 2

    amplitude = np.sqrt(
        field_squared.sum(dim="field").integrate(coord="theta") / (2 * np.pi)
    )

    assert np.isclose(amplitude, 1.0)
