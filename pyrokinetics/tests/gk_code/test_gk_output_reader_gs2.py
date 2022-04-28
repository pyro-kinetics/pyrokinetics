from pyrokinetics.gk_code import GKOutputReaderGS2, GKInputGS2
from itertools import product
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
# TODO include option to have only 1 kx and 1 ky to test eigenvalues and eigenfunctions
# Expected coords in a GS2 output file

dims = {
    "t": 21,
    "time": 21,
    "ky": 2,
    "kx": 3,
    "theta": 13,
    "energy": 12,
    "lambda": 13,
    "pitch": 13,
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
nfields = 3
nmoments = 3
gs2_field_shape = (dims["t"], dims["ky"], dims["kx"], dims["theta"], real_or_imag)
gs2_flux_shape = (dims["t"], nspecies)
gs2_flux_by_mode_shape = (dims["t"], nspecies, dims["ky"], dims["kx"])
pyro_field_shape = (nfields, dims["theta"], dims["kx"], dims["ky"], dims["time"])
pyro_flux_shape = (nspecies, nmoments, nfields, dims["ky"], dims["time"])

# Define variants on the GS2 data
linear_opts = ["linear", "nonlinear"]
field_opts = ["all_fields", "some_fields", "no_fields"]
flux_opts = ["all_fluxes", "some_fluxes", "does_not_have_fluxes"]
flux_types = ["flux", "flux_by_mode"]


@pytest.fixture
def mock_reader(monkeypatch, request):

    linear = request.param[0] == "linear"
    all_fields = request.param[1] == "has_fields"
    some_fields = request.param[1] == "some_fields" or all_fields
    all_fluxes = request.param[2] == "all_fluxes"
    some_fluxes = request.param[2] == "some_fluxes" or all_fluxes
    flux_type = request.param[3]

    class MockGKInputGS2:
        """class that contains only relevant parts of GKInputGS2"""

        def __init__(self):
            self.data = {
                "species_knobs": {"nspec": 2},
                "species_parameters_1": {"z": -1},
                "species_parameters_2": {"z": 1},
            }

        def is_linear(self):
            return linear

    def mock(filename):
        """ignores filename, creates idealised results for _get_raw_data"""
        # Expected coords in a GS2 output file
        coords = gs2_coords
        # Expected fields and fluxes
        data_vars = dict()

        fields = ["phi", "apar", "bpar"]
        if some_fields:
            for field in fields:
                data_vars[f"{field}_t"] = (
                    ("t", "ky", "kx", "theta", "ri"),
                    np.ones(gs2_field_shape),
                )
                if some_fields and not all_fields:
                    break

        flux_fields = ["es", "apar", "bpar"]
        moments = ["part", "heat", "mom"]
        if some_fluxes:
            for field, moment in product(flux_fields, moments):
                if flux_type == "flux_by_mode":
                    data_vars[f"{field}_{moment}_{flux_type}"] = (
                        ("t", "species", "ky", "kx"),
                        np.ones(gs2_flux_by_mode_shape),
                    )
                else:
                    data_vars[f"{field}_{moment}_{flux_type}"] = (
                        ("t", "species"),
                        np.ones(gs2_flux_shape),
                    )
                if some_fluxes and not all_fluxes:
                    break
        # Store linear, has_fields, and has_fluxes as attrs
        # These are not present in a real GS2 file, but they're useful for testing
        attrs = {
            "linear": linear,
            "all_fields": all_fields,
            "some_fields": some_fields,
            "all_fluxes": all_fluxes,
            "some_fluxes": some_fluxes,
        }

        raw_data = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        gk_input = MockGKInputGS2()
        input_str = "hello world"
        return raw_data, gk_input, input_str

    monkeypatch.setattr(GKOutputReaderGS2, "_get_raw_data", staticmethod(mock))

    return GKOutputReaderGS2()


@pytest.mark.parametrize(
    "mock_reader",
    [(linear, "all_fields", "all_fluxes", "flux_by_mode") for linear in linear_opts],
    indirect=True,
)
def test_read(mock_reader):
    raw_data, _, _ = mock_reader._get_raw_data("dummy_filename")
    dataset = mock_reader.read("dummy_filename")
    # Expect the resulting dataset to have all field and flux data, plus a copy
    # of the input file
    if raw_data.some_fields:
        assert np.array_equal(dataset["fields"].shape, pyro_field_shape)
        assert dataset["fields"].dtype == complex
    else:
        with pytest.raises(KeyError):
            dataset["fields"]
    if raw_data.some_fluxes:
        assert np.array_equal(dataset["fluxes"].shape, pyro_flux_shape)
        assert dataset["fluxes"].dtype == float
    else:
        with pytest.raises(KeyError):
            dataset["fluxes"]
    assert dataset.input_file == "hello world"


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, "all_fluxes", "flux") for f in field_opts],
    indirect=True,
)
def test_init_dataset(mock_reader):
    raw_data, gk_input, _ = mock_reader._get_raw_data("dummy_filename")
    data = mock_reader._init_dataset(raw_data, gk_input)
    assert isinstance(data, xr.Dataset)
    expected_coords = {
        "time": dims["time"],
        "kx": dims["kx"],
        "ky": dims["ky"],
        "theta": dims["theta"],
        "energy": dims["energy"],
        "pitch": dims["pitch"],
    }
    for coord, size in expected_coords.items():
        assert coord in data.coords and size == data.dims[coord]
        assert data.attrs[f"n{coord}"] == size
    assert "field" in data.coords
    assert np.all(np.isin(["phi", "apar", "bpar"], data.coords["field"]))
    assert data.attrs["nfield"] == nfields
    assert "moment" in data.coords
    assert np.all(np.isin(["particle", "energy", "momentum"], data.coords["moment"]))
    assert "species" in data.coords
    assert np.all(np.isin(["electron", "ion1"], data.coords["species"]))
    assert data.attrs["nspecies"] == nspecies


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, "all_fluxes", "flux") for f in field_opts],
    indirect=True,
)
def test_set_fields(mock_reader):
    raw_data, gk_input, _ = mock_reader._get_raw_data("dummy_filename")
    data = mock_reader._init_dataset(raw_data, gk_input)
    data = mock_reader._set_fields(data, raw_data)
    if raw_data.some_fields:
        assert np.array_equal(data["fields"].shape, pyro_field_shape)
        # Expect all present fields to be finite
        if raw_data.all_fields:
            assert np.all(data["fields"])
        else:
            assert np.any(data["fields"]) and not np.all(data["fields"])
    else:
        with pytest.raises(KeyError):
            data["fields"]


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", "all_fields", f1, f2) for f1, f2 in product(flux_opts, flux_types)],
    indirect=True,
)
def test_set_fluxes(mock_reader):
    raw_data, gk_input, _ = mock_reader._get_raw_data("dummy_filename")
    data = mock_reader._init_dataset(raw_data, gk_input)
    data = mock_reader._set_fluxes(data, raw_data)
    if raw_data.some_fluxes:
        assert np.array_equal(data["fluxes"].shape, pyro_flux_shape)
        # Expect all present fluxes to be finite
        if raw_data.all_fluxes:
            assert np.all(data["fluxes"])
        else:
            assert np.any(data["fluxes"]) and not np.all(data["fluxes"])
    else:
        with pytest.raises(KeyError):
            data["fluxes"]
