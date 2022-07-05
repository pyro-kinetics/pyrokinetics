from pyrokinetics.gk_code import GKOutputReaderGS2, GKInputGS2
from pyrokinetics import template_dir
from itertools import product, combinations
from pathlib import Path
import xarray as xr
import numpy as np
import pytest

from pyrokinetics.tests.gk_code.utils import array_similar, get_golden_answer_data


@pytest.fixture(scope="module")
def gs2_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gs2")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGS2()


@pytest.fixture
def gs2_output_file():
    return template_dir / "outputs" / "GS2_linear" / "gs2.out.nc"


@pytest.fixture
def not_gs2_output_file(gs2_tmp_path):
    filename = gs2_tmp_path / "not_gs2.out.nc"
    x = xr.Dataset(coords={"x": [1, 2, 3]})
    x.to_netcdf(filename)
    return filename


@pytest.fixture
def not_netcdf_output_file(gs2_tmp_path):
    filename = gs2_tmp_path / "hello_world.txt"
    with open(filename, "w") as file:
        file.write("hello world!")
    return filename


# Here we test the functions that read and verify a real netCDF file.
# Tests beyond here make use of 'monkeypatching' to provide idealised
# GS2 outputs, as this avoids filling the project with dozens of
# netCDF files to represent each possible GS2 setup.
def test_get_raw_data(reader, gs2_output_file):
    raw_data, gk_input, input_str = reader._get_raw_data(gs2_output_file)
    assert isinstance(gk_input, GKInputGS2)
    assert isinstance(input_str, str)


def test_get_raw_data_not_gs2(reader, not_gs2_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_gs2_output_file)


def test_get_raw_data_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader._get_raw_data(not_netcdf_output_file)


def test_verify_gs2_output(reader, gs2_output_file):
    # Expect exception to be raised if this fails
    reader.verify(gs2_output_file)


def test_verify_not_gs2(reader, not_gs2_output_file):
    with pytest.raises(Exception):
        reader.verify(not_gs2_output_file)


def test_verify_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader.verify(not_netcdf_output_file)


def test_infer_path_from_input_file_gs2():
    input_path = Path("dir/to/input_file.in")
    output_path = GKOutputReaderGS2.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/input_file.out.nc")


# Golden answer tests
# Compares against results obtained using GKCode methods from commit 7d551eaa
# This data was gathered from templates/outputs/GS2_linear

reference_data_commit_hash = "7d551eaa"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gs2_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    ds = get_golden_answer_data(cdf_path)
    request.cls.reference_data = ds


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GS2_linear" / "gs2.out.nc"
    request.cls.data = GKOutputReaderGS2().read(path)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGS2GoldenAnswers:
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


# Define mock reader that generates idealised GS2 raw data
# Returns a 3-tuple. The first element is the reader, while the second is a dict
# of the expected dimensions in the output, and the third is a copy of the inputs
@pytest.fixture
def mock_reader(monkeypatch, request):

    linear = request.param[0] == "linear"
    fields = request.param[1]
    flux_type = request.param[2]

    dims = {
        "t": 21,
        "time": 21,
        "ky": 2,
        "kx": 3,
        "theta": 13,
        "energy": 12,
        "pitch": 13,
        "lambda": 13,  # synonym for pitch used by GS2
    }

    gs2_coords = {
        "t": np.linspace(0, 10.0, dims["t"]),
        "ky": [0.0, 0.5],
        "kx": [0.0, 0.4, -0.4],
        "theta": np.linspace(-np.pi, np.pi, dims["theta"]),
        "energy": np.linspace(0.001, 0.95, dims["energy"]),
        "lambda": np.linspace(0.05, 1.2, dims["lambda"]),
    }

    # Define expected shapes
    real_or_imag = 2
    nspecies = 2
    nfields = len(fields)
    nmoments = 3
    gs2_field_shape = (dims["t"], dims["ky"], dims["kx"], dims["theta"], real_or_imag)
    gs2_flux_shape = (dims["t"], nspecies)
    gs2_flux_by_mode_shape = (dims["t"], nspecies, dims["ky"], dims["kx"])

    class MockGKInputGS2:
        """class that contains only relevant parts of GKInputGS2"""

        def __init__(self):
            # species data
            self.data = {
                "species_knobs": {"nspec": 2},
                "species_parameters_1": {"z": -1},
                "species_parameters_2": {"z": 1},
            }
            # field data
            self.data["knobs"] = {}
            for field in fields:
                self.data["knobs"][f"f{field}"] = 1.0
            # fphi is 1 by default, so the user would have to manually set it to 0
            # for Pyrokinetics to register it as 0.
            if "phi" not in fields:
                self.data["knobs"]["fphi"] = 0.0

        def is_linear(self):
            return linear

    def mock(filename):
        """ignores filename, creates idealised results for _get_raw_data"""
        # Expected coords in a GS2 output file
        coords = gs2_coords
        # Expected fields and fluxes
        data_vars = {}

        for field in fields:
            data_vars[f"{field}_t"] = (
                ("t", "ky", "kx", "theta", "ri"),
                np.ones(gs2_field_shape),
            )

        moments = ["part", "heat", "mom"]
        for field, moment in product(fields, moments):
            flux_field = "es" if field == "phi" else field
            if flux_type == "flux_by_mode":
                data_vars[f"{flux_field}_{moment}_{flux_type}"] = (
                    ("t", "species", "ky", "kx"),
                    np.ones(gs2_flux_by_mode_shape),
                )
            elif flux_type == "flux":
                data_vars[f"{flux_field}_{moment}_{flux_type}"] = (
                    ("t", "species"),
                    np.ones(gs2_flux_shape),
                )
            else:
                break

        raw_data = xr.Dataset(coords=coords, data_vars=data_vars)
        gk_input = MockGKInputGS2()
        input_str = "hello world"
        return raw_data, gk_input, input_str

    monkeypatch.setattr(GKOutputReaderGS2, "_get_raw_data", staticmethod(mock))

    # Expected shapes in the output, plus a copy of the input data
    expected = {
        "field_shape": (nfields, dims["theta"], dims["kx"], dims["ky"], dims["time"]),
        "flux_shape": (nspecies, nmoments, nfields, dims["ky"], dims["time"]),
        "eigenvalues_shape": (dims["kx"], dims["ky"], dims["time"]),
        "growth_rate_tolerance_shape": (dims["kx"], dims["ky"]),
        "coords": {
            "time": dims["time"],
            "kx": dims["kx"],
            "ky": dims["ky"],
            "theta": dims["theta"],
            "energy": dims["energy"],
            "pitch": dims["pitch"],
            "field": nfields,
            "species": nspecies,
            "moment": nmoments,
        },
    }
    expected["growth_rate_shape"] = expected["eigenvalues_shape"]
    expected["mode_frequency_shape"] = expected["eigenvalues_shape"]
    expected["eigenfunctions_shape"] = expected["field_shape"]

    # Copy of the inputs
    inputs = {
        "linear": linear,
        "fields": fields,
        "flux_type": flux_type,
    }

    return GKOutputReaderGS2(), expected, inputs


# List options for use in parametrize lists
linear_opts = ["linear", "nonlinear"]
flux_opts = ["flux", "flux_by_mode", None]
# all possible combinations of fields
all_fields = ["phi", "apar", "bpar"]
field_opts = [[*fields] for r in range(4) for fields in combinations(all_fields, r)]


@pytest.mark.parametrize(
    "mock_reader",
    [(linear, all_fields, "flux_by_mode") for linear in linear_opts],
    indirect=True,
)
def test_read(mock_reader):
    reader, expected, inputs = mock_reader
    dataset = reader.read("dummy_filename")
    # Expect the resulting dataset to have all field and flux data, plus a copy
    # of the input file
    assert np.array_equal(dataset["fields"].shape, expected["field_shape"])
    assert dataset["fields"].dtype == complex
    assert np.array_equal(dataset["fluxes"].shape, expected["flux_shape"])
    assert dataset["fluxes"].dtype == float
    assert dataset.input_file == "hello world"
    eigen_vals = [
        "eigenvalues",
        "eigenfunctions",
        "mode_frequency",
        "growth_rate",
        "growth_rate_tolerance",
    ]
    if inputs["linear"]:
        for eigen in eigen_vals:
            assert np.array_equal(dataset[eigen].shape, expected[f"{eigen}_shape"])
    else:
        for eigen in eigen_vals:
            with pytest.raises(KeyError):
                dataset[eigen]


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts],
    indirect=True,
)
def test_init_dataset(mock_reader):
    reader, expected, inputs = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader._init_dataset(raw_data, gk_input)
    assert isinstance(data, xr.Dataset)
    for coord, size in expected["coords"].items():
        assert coord in data.coords
        assert size == data.dims[coord]
        assert data.attrs[f"n{coord}"] == size
    assert np.array_equal(["particle", "energy", "momentum"], data.coords["moment"])
    assert np.array_equal(["electron", "ion1"], data.coords["species"])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts],
    indirect=True,
)
def test_set_fields(mock_reader):
    reader, expected, inputs = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader._init_dataset(raw_data, gk_input)
    data = reader._set_fields(data, raw_data)
    # If we didn't include any fields, data["fields"] should not exist
    if not inputs["fields"]:
        with pytest.raises(KeyError):
            data["fields"]
        return
    assert np.array_equal(data["fields"].shape, expected["field_shape"])
    # Expect all present fields to be finite
    assert np.all(data["fields"])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", all_fields, f) for f in flux_opts],
    indirect=True,
)
def test_set_fluxes(mock_reader):
    reader, expected, inputs = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader._init_dataset(raw_data, gk_input)
    data = reader._set_fluxes(data, raw_data)
    # If no fluxes are found, data["fluxes"] should not exist
    if inputs["flux_type"] is None:
        with pytest.raises(KeyError):
            data["fluxes"]
        return
    assert np.array_equal(data["fluxes"].shape, expected["flux_shape"])
    # Expect all present fluxes to be finite
    assert np.all(data["fluxes"])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts[1:]],  # skip fields=[]
    indirect=True,
)
def test_set_eigenvalues(mock_reader):
    reader, expected, inputs = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader._init_dataset(raw_data, gk_input)
    data = reader._set_fields(data, raw_data)
    data = reader._set_eigenvalues(data)
    for x in ["eigenvalues", "mode_frequency", "growth_rate"]:
        assert np.array_equal(data[x].shape, expected[f"{x}_shape"])
    data = reader._set_growth_rate_tolerance(data)
    assert np.array_equal(
        data["growth_rate_tolerance"].shape,
        expected["growth_rate_tolerance_shape"],
    )


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts[1:]],  # skip fields=[]
    indirect=True,
)
def test_set_eigenfunctions(mock_reader):
    reader, expected, inputs = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader._init_dataset(raw_data, gk_input)
    data = reader._set_fields(data, raw_data)
    data = reader._set_eigenfunctions(data)
    assert np.array_equal(
        data["eigenfunctions"].shape,
        expected["eigenfunctions_shape"],
    )
