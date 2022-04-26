from pyrokinetics.gk_code import GKOutputReaderGS2, GKInputGS2
from itertools import product
import xarray as xr
import numpy as np
import pytest
import pathlib


@pytest.fixture
def reader():
    return GKOutputReaderGS2()


@pytest.fixture
def output_file():
    return pathlib.Path(__file__).parent.parent / "test_files" / "gs2.out.nc"


# Here we test the function that reads a real netCDF file.
# Tests beyond here make use of 'monkeypatching' to provide idealised
# GS2 outputs, as this avoids filling the project with dozens of
# netCDF files to represent each possible GS2 setup.
def test_get_raw_data(reader, output_file):
    raw_data, gk_input, input_str = reader._get_raw_data(output_file)
    assert raw_data.attrs["software_name"] == "GS2"
    assert isinstance(gk_input, GKInputGS2)
    assert isinstance(input_str, str)


@pytest.fixture
def mock_reader(monkeypatch):
    class MockGKInputGS2:
        """class that contains only relevant parts of GKInputGS2"""

        def __init__(self):
            self.data = {
                "knobs": {
                    "fphi": 1.0,
                    "fapar": 1.0,
                    "fbpar": 1.0,
                },
                "species_knobs": {"nspec": 2},
                "species_parameters_1": {"z": -1},
                "species_parameters_2": {"z": 1},
            }

        def is_linear(self):
            return True

    def mock(filename):
        """ignores filename, creates idealised results for _get_raw_data"""
        # Expected coords in a GS2 output file
        coords = {
            "t": np.linspace(0, 10.0, 21),
            "ky": [0.0, 0.5],
            "kx": [0.0, 0.4, -0.4],
            "theta": np.linspace(-np.pi, np.pi, 13),
            "energy": np.linspace(0.001, 0.95, 12),
            "lambda": np.linspace(0.05, 1.2, 13),
        }
        # Expected fields and fluxes
        data_vars = dict()
        fields = ["phi", "apar", "bpar"]
        for field in fields:
            data_vars[f"{field}_t"] = (
                ("t", "ky", "kx", "theta", "ri"),
                np.ones((21, 2, 3, 13, 2)),
            )
        moments = ["part", "heat", "mom"]
        for field, moment in product(fields, moments):
            data_vars[f"{field}_{moment}_flux_by_mode"] = (
                ("t", "species", "ky", "kx"),
                np.ones((21, 2, 2, 3)),
            )
        raw_data = xr.Dataset(coords=coords, data_vars=data_vars)
        gk_input = MockGKInputGS2()
        input_str = "hello world"
        return raw_data, gk_input, input_str

    monkeypatch.setattr(GKOutputReaderGS2, "_get_raw_data", staticmethod(mock))

    return GKOutputReaderGS2()


def test_read(mock_reader):
    dataset = mock_reader.read("dummy_filename")
    # Expect the resulting dataset to have all field and flux data, plus a copy
    # of the input file
    assert np.array_equal(dataset["fields"].shape, (3, 13, 3, 2, 21))
    assert dataset["fields"].dtype == complex
    assert np.array_equal(dataset["fluxes"].shape, (2, 3, 3, 2, 21))
    assert dataset["fluxes"].dtype == float
    assert dataset.input_file == "hello world"
