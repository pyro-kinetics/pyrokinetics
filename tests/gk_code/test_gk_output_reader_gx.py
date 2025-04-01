from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyrokinetics import Pyro, template_dir
from pyrokinetics.gk_code import GKInputGX, GKOutputReaderGX
from pyrokinetics.gk_code.gk_output import GKOutput


@pytest.fixture(scope="module")
def gx_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gx")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGX()


@pytest.fixture
def gx_output_file():
    return template_dir / "outputs" / "GX_linear" / "gx.out.nc"


@pytest.fixture
def not_gx_output_file(gx_tmp_path):
    filename = gx_tmp_path / "not_gx.out.nc"
    x = xr.Dataset(coords={"x": [1, 2, 3]})
    x.to_netcdf(filename)
    return filename


@pytest.fixture
def not_netcdf_output_file(gx_tmp_path):
    filename = gx_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


# Here we test the functions that read and verify a real netCDF file.
# Tests beyond here make use of 'monkeypatching' to provide idealised
# GX outputs, as this avoids filling the project with dozens of
# netCDF files to represent each possible GX setup.
def test_get_raw_data(reader, gx_output_file):
    raw_data, gk_input, input_str = reader._get_raw_data(gx_output_file)
    assert isinstance(gk_input, GKInputGX)
    assert isinstance(input_str, str)


def test_get_raw_data_not_gx(reader, not_gx_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_gx_output_file)


def test_get_raw_data_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_netcdf_output_file)


def test_verify_gx_output(reader, gx_output_file):
    # Expect exception to be raised if this fails
    reader.verify_file_type(gx_output_file)


def test_verify_not_gx(reader, not_gx_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_gx_output_file)


def test_verify_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_netcdf_output_file)


def test_infer_path_from_input_file_gx():
    input_path = Path("dir/to/input_file.in")
    output_path = GKOutputReaderGX.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/input_file.out.nc")


@pytest.mark.parametrize(
    "load_fields",
    [
        True,
        False,
    ],
)
def test_amplitude(load_fields):

    path = template_dir / "outputs" / "GX_linear"

    pyro = Pyro(gk_file=path / "gx.in")

    pyro.load_gk_output(load_fields=load_fields)

    eigenfunctions = pyro.gk_output.data["eigenfunctions"].isel(
        time=-1, ky=1, missing_dims="ignore"
    )
    field_squared = np.abs(eigenfunctions) ** 2

    amplitude = np.sqrt(
        field_squared.pint.dequantify().sum(dim="field").integrate(coord="theta")
        / (2 * np.pi)
    )

    assert hasattr(eigenfunctions.data, "units")
    assert np.isclose(amplitude, 1.0)


# Golden answer tests
# This data was gathered from templates/outputs/GX_linear

reference_data_commit_hash = "def4c998"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gx_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GX_linear"

    pyro = Pyro(gk_file=path / "gx.in", name="test_gk_output_gx")
    norm = pyro.norms

    request.cls.data = GKOutputReaderGX().read_from_file(path / "gx.out.nc", norm=norm)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGXGoldenAnswers:
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
            "heat",
            "eigenvalues",
            "eigenfunctions",
            "growth_rate",
            "mode_frequency",
            "growth_rate_tolerance",
        ],
    )
    def test_data_vars(self, array_similar, var):
        assert array_similar(self.reference_data[var], self.data[var], nan_to_zero=True)

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
