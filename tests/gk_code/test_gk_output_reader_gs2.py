from pyrokinetics.gk_code import GKOutputReaderGS2, GKInputGS2
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics import template_dir, Pyro
from pyrokinetics.normalisation import SimulationNormalisation as Normalisation
from itertools import product, combinations
from pathlib import Path
import xarray as xr
import numpy as np
import pytest
from types import SimpleNamespace as basic_object


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
    reader.verify_file_type(gs2_output_file)


def test_verify_not_gs2(reader, not_gs2_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_gs2_output_file)


def test_verify_not_netcdf(reader, not_netcdf_output_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_netcdf_output_file)


def test_infer_path_from_input_file_gs2():
    input_path = Path("dir/to/input_file.in")
    output_path = GKOutputReaderGS2.infer_path_from_input_file(input_path)
    assert output_path == Path("dir/to/input_file.out.nc")


# Golden answer tests
# This data was gathered from templates/outputs/GS2_linear

reference_data_commit_hash = "4a932797"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gs2_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GS2_linear"

    pyro = Pyro(gk_file=path / "gs2.in", name="test_gk_output_gs2")
    norm = pyro.norms

    request.cls.data = GKOutputReaderGS2().read_from_file(
        path / "gs2.out.nc", norm=norm
    )


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGS2GoldenAnswers:
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
        "ky": np.linspace(0, 0.5, dims["ky"]),
        "kx": np.linspace(-0.5, 0.5, dims["kx"]),
        "theta": np.linspace(-np.pi, np.pi, dims["theta"]),
        "energy": np.linspace(0.001, 0.95, dims["energy"]),
        "lambda": np.linspace(0.05, 1.2, dims["lambda"]),
    }

    if linear:
        dims["kx"] = 1
        dims["ky"] = 1
        gs2_coords["ky"] = [0.5]
        gs2_coords["kx"] = [0.0]

    # Define expected shapes
    real_or_imag = 2
    nspecies = 2
    nfields = len(fields)
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

        def get_local_geometry(self):
            geometry = basic_object()
            geometry.Rmaj = 3.0
            geometry.bunit_over_b0 = 1.0205177029353276
            return geometry

    def mock(filename):
        """ignores filename, creates idealised results for _get_raw_data"""
        # Expected coords in a GS2 output file
        coords = gs2_coords
        # Expected fields and fluxes
        data_vars = {}

        data_vars["bmag"] = np.ones(dims["theta"])

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
        "field_shape": (dims["theta"], dims["kx"], dims["ky"], dims["time"]),
        "flux_shape": (nfields, nspecies, dims["ky"], dims["time"]),
        "eigenvalues_shape": (dims["kx"], dims["ky"], dims["time"]),
        "eigenfunction_shape": (
            nfields,
            dims["theta"],
            dims["kx"],
            dims["ky"],
            dims["time"],
        ),
        "coords": {
            "time": dims["time"],
            "kx": dims["kx"],
            "ky": dims["ky"],
            "theta": dims["theta"],
            "energy": dims["energy"],
            "pitch": dims["pitch"],
        },
    }
    expected["growth_rate_shape"] = expected["eigenvalues_shape"]
    expected["mode_frequency_shape"] = expected["eigenvalues_shape"]
    expected["eigenfunctions_shape"] = expected["eigenfunction_shape"]

    # Copy of the inputs
    inputs = {
        "linear": linear,
        "fields": fields,
        "flux_type": flux_type,
    }

    local_norm = Normalisation("test")

    return GKOutputReaderGS2(), expected, inputs, local_norm


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
def test_read_from_file(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    dataset = reader.read_from_file("dummy_filename", local_norm)
    # Expect the resulting dataset to have all field and flux data, plus a copy
    # of the input file
    for field in dataset["field"].data:
        assert np.array_equal(dataset[field].shape, expected["field_shape"])
        assert dataset[field].dtype == complex
    for flux in dataset["flux"].data:
        assert np.array_equal(dataset[flux].shape, expected["flux_shape"])
        assert dataset[flux].dtype == float
    assert dataset.input_file == "hello world"
    eigen_vals = [
        "eigenvalues",
        "eigenfunctions",
        "mode_frequency",
        "growth_rate",
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
def test_get_coords(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    raw_data, gk_input, input_str = reader._get_raw_data("dummy_filename")
    downsize = 1
    coords = reader._get_coords(raw_data, gk_input, downsize)
    assert isinstance(coords, dict)
    for coord, size in expected["coords"].items():
        assert coord in coords.keys()
        assert size == len(coords[coord])
    assert np.array_equal(["particle", "heat", "momentum"], coords["flux"])
    assert np.array_equal(["electron", "ion1"], coords["species"])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts],
    indirect=True,
)
def test_get_fields(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    raw_data, gk_input, input_str = reader._get_raw_data("dummy_filename")
    fields = reader._get_fields(raw_data)
    # If we didn't include any fields, data["fields"] should not exist
    if not inputs["fields"]:
        with pytest.raises(KeyError):
            fields["phi"]
        return
    for field in fields:
        assert np.array_equal(fields[field].shape, expected["field_shape"])
        # Expect all present fields to be finite
        assert np.all(fields[field])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", all_fields, f) for f in flux_opts],
    indirect=True,
)
def test_get_fluxes(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    raw_data, gk_input, input_str = reader._get_raw_data("dummy_filename")
    downsize = 1

    coords = reader._get_coords(raw_data, gk_input, downsize)
    fluxes = reader._get_fluxes(raw_data, gk_input, coords)

    # If no fluxes are found, data["fluxes"] should not exist
    if inputs["flux_type"] is None:
        with pytest.raises(KeyError):
            fluxes["heat"]
        return

    for moment in fluxes:
        assert np.array_equal(fluxes[moment].shape, expected["flux_shape"])
        # Expect all present fluxes to be finite
        assert np.all(fluxes[moment])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts[1:]],  # skip fields=[]
    indirect=True,
)
def test_get_eigenvalues(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    data = reader.read_from_file("dummy_filename", local_norm)

    for x in ["eigenvalues", "mode_frequency", "growth_rate"]:
        assert np.array_equal(data[x].shape, expected[f"{x}_shape"])


@pytest.mark.parametrize(
    "mock_reader",
    [("linear", f, None) for f in field_opts[1:]],  # skip fields=[]
    indirect=True,
)
def test_get_eigenfunctions(mock_reader):
    reader, expected, inputs, local_norm = mock_reader
    raw_data, gk_input, _ = reader._get_raw_data("dummy_filename")
    data = reader.read_from_file("dummy_filename", local_norm)

    assert np.array_equal(
        data["eigenfunctions"].shape,
        expected["eigenfunctions_shape"],
    )
