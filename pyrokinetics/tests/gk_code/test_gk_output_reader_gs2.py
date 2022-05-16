from pyrokinetics.gk_code import GKOutputReaderGS2, GKInputGS2
from itertools import product, combinations
from pathlib import Path
import xarray as xr
import numpy as np
import pytest


@pytest.fixture(scope="module")
def gs2_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_gk_output_reader_gs2")
    return tmp_dir


@pytest.fixture
def reader():
    return GKOutputReaderGS2()


@pytest.fixture
def gs2_output_file():
    return Path(__file__).parent.parent / "test_files" / "gs2.out.nc"


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
    assert raw_data.attrs["software_name"] == "GS2"
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
    # TODO if inputs["linear"], check eigenvalues present


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
