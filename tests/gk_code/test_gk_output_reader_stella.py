from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from pyrokinetics import Pyro, template_dir
from pyrokinetics.gk_code import GKInputSTELLA, GKOutputReaderSTELLA
from pyrokinetics.gk_code.gk_output import GKOutput


@pytest.fixture(scope="module")
def stella_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_stella")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderSTELLA()


@pytest.fixture
def stella_output_file():
    return template_dir / "outputs" / "STELLA_linear" / "stella.out.nc"


@pytest.fixture
def not_stella_output_file(stella_tmp_path):
    filename = stella_tmp_path / "not_stella.out.nc"
    x = xr.Dataset(coords={"x": [1, 2, 3]})
    x.to_netcdf(filename)
    return filename


@pytest.fixture
def not_netcdf_output_file(stella_tmp_path):
    filename = stella_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


# Here we test the functions that read and verify a real netCDF file.
# Tests beyond here make use of 'monkeypatching' to provide idealised
# STELLA outputs, as this avoids filling the project with dozens of
# netCDF files to represent each possible STELLA setup.
def test_get_raw_data(reader, stella_output_file):
    raw_data, gk_input, input_str = reader._get_raw_data(stella_output_file)
    assert isinstance(gk_input, GKInputSTELLA)
    assert isinstance(input_str, str)


def test_get_raw_data_not_stella(reader, not_stella_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_stella_output_file)


def test_get_raw_data_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_netcdf_output_file)


def test_verify_stella_output(reader, stella_output_file):
    # Expect exception to be raised if this fails
    reader.verify_file_type(stella_output_file)


def test_verify_not_stella(reader, not_stella_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_stella_output_file)


def test_verify_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_netcdf_output_file)


def test_infer_path_from_input_file_stella():
    input_path = Path("dir/to/input_file.in")
    output_path = GKOutputReaderSTELLA.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/input_file.out.nc")


def test_stella_read_omega_file(tmp_path):
    """Can we match growth rate/frequency from netCDF file"""

    path = template_dir / "outputs" / "STELLA_linear"
    pyro = Pyro(gk_file=path / "stella.in", name="test_gk_output_stella")
    pyro.load_gk_output()

    with nc.Dataset(path / "stella.out.nc") as netcdf_data:
        cdf_mode_freq = netcdf_data["omega"][-1, 0, 0, 0]
        cdf_gamma = netcdf_data["omega"][-1, 0, 0, 1]

    assert np.isclose(
        pyro.gk_output.data["growth_rate"]
        .isel(time=-1, ky=0, kx=0)
        .data.to(pyro.norms.stella)
        .m,
        cdf_gamma,
        rtol=0.1,
    )
    assert np.isclose(
        pyro.gk_output.data["mode_frequency"]
        .isel(time=-1, ky=0, kx=0)
        .data.to(pyro.norms.stella)
        .m,
        cdf_mode_freq,
        rtol=0.1,
    )


# Golden answer tests
# This data was gathered from templates/outputs/STELLA_linear

reference_data_commit_hash = "cecf2584"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"stella_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "STELLA_linear"

    pyro = Pyro(gk_file=path / "stella.in", name="test_gk_output_stella")
    norm = pyro.norms

    request.cls.data = GKOutputReaderSTELLA().read_from_file(
        path / "stella.out.nc", norm=norm
    )


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestSTELLAGoldenAnswers:
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
            "momentum",
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
