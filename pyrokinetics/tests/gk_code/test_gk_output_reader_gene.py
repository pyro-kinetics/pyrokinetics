from pyrokinetics.gk_code import GKOutputReaderGENE
from pyrokinetics import template_dir
from pathlib import Path
import numpy as np
import pytest
import subprocess

from pyrokinetics.tests.gk_code.utils import array_similar, get_golden_answer_data

# TODO mock output tests, similar to GS2


@pytest.fixture(scope="module")
def gene_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gene")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGENE()


@pytest.fixture
def gene_output_dir(gene_tmp_path):
    mock_dir = gene_tmp_path / "mock_dir"
    mock_dir.mkdir()
    subprocess.run(
        ["cp", str(template_dir / "input.gene"), str(mock_dir / "parameters_0000")]
    )
    return mock_dir


@pytest.fixture
def gene_output_dir_missing_parameters(gene_tmp_path):
    mock_dir = gene_tmp_path / "broken_mock_dir"
    mock_dir.mkdir()
    for f in [mock_dir / f for f in ["nrg_0000", "field_0000"]]:
        with open(f, "w") as _:
            pass
    return mock_dir


@pytest.fixture
def empty_gene_dir(gene_tmp_path):
    mock_dir = gene_tmp_path / "empty_dir"
    mock_dir.mkdir()
    return mock_dir


@pytest.fixture
def not_gene_file(gene_tmp_path):
    mock_dir = gene_tmp_path / "nongene_dir"
    mock_dir.mkdir()
    filename = mock_dir / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


def test_verify_gene_output(reader, gene_output_dir):
    # Expect exception to be raised if this fails
    reader.verify(gene_output_dir)


def test_verify_gene_missing_parameters(reader, gene_output_dir_missing_parameters):
    with pytest.raises(Exception):
        reader.verify(gene_output_dir_missing_parameters)


def test_verify_not_gene_dir(reader, empty_gene_dir):
    with pytest.raises(Exception):
        reader.verify(empty_gene_dir)


def test_verify_not_gene_file(reader, not_gene_file):
    with pytest.raises(Exception):
        reader.verify(not_gene_file)


@pytest.mark.parametrize(
    "input_path",
    [
        Path("dir/to/parameters_0003"),
        Path("dir/to/nrg_0017"),
        Path("dir/to/input_file"),
    ],
)
def test_infer_path_from_input_file_gene(input_path):
    output_path = GKOutputReaderGENE.infer_path_from_input_file(input_path)
    # If the last four chars are digits, expect to find "parameters_####".
    # Otherwise, get the dir
    last_4_chars = str(input_path)[-4:]
    if last_4_chars.isdigit():
        assert output_path == Path(f"dir/to/parameters_{last_4_chars}")
    else:
        assert output_path == Path("dir/to/")


# Golden answer tests
# Compares against results obtained using GKCode methods from commit 7d551eaa
# Update: Commit 9eae331 accounts for last time step (7d551eaa-2nd last step)
# This data was gathered from templates/outputs/GENE_linear

reference_data_commit_hash = "9eae331"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gene_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    ds = get_golden_answer_data(cdf_path)
    request.cls.reference_data = ds


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GENE_linear" / "parameters_0001"
    request.cls.data = GKOutputReaderGENE().read(path)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGENEGoldenAnswers:
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
