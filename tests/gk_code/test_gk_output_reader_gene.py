import shutil
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.gk_code import GKOutputReaderGENE
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics.normalisation import SimulationNormalisation as Normalisation
from pyrokinetics.units import ureg

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
    shutil.copy(template_dir / "input.gene", mock_dir / "parameters_0000")
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
    reader.verify_file_type(gene_output_dir)


def test_verify_gene_missing_parameters(reader, gene_output_dir_missing_parameters):
    with pytest.raises(Exception):
        reader.verify_file_type(gene_output_dir_missing_parameters)


def test_verify_not_gene_dir(reader, empty_gene_dir):
    with pytest.raises(Exception):
        reader.verify_file_type(empty_gene_dir)


def test_verify_not_gene_file(reader, not_gene_file):
    with pytest.raises(Exception):
        reader.verify_file_type(not_gene_file)


@pytest.mark.parametrize(
    "input_path",
    [
        Path("dir/to/parameters_0003"),
        Path("dir/to/nrg_0017"),
        Path("dir/to/input_file"),
        Path("dir_0001/to_5102/parameters_0005"),
    ],
)
def test_infer_path_from_input_file_gene(input_path):
    output_path = GKOutputReaderGENE.infer_path_from_input_file(input_path)
    # If the last four chars are digits, expect to find "parameters_####".
    # Otherwise, get the dir
    last_4_chars = str(input_path)[-4:]
    if last_4_chars.isdigit():
        assert output_path == input_path.parent / f"parameters_{last_4_chars}"
    else:
        assert output_path == input_path.parent


# Golden answer tests
# This data was gathered from templates/outputs/GENE_linear

reference_data_commit_hash = "788f630f"


@pytest.fixture(scope="class")
def golden_answer_reference_data(request):
    this_dir = Path(__file__).parent
    cdf_path = (
        this_dir
        / "golden_answers"
        / f"gene_linear_output_{reference_data_commit_hash}.netcdf4"
    )
    request.cls.reference_data = GKOutput.from_netcdf(cdf_path)


@pytest.fixture(scope="class")
def golden_answer_data(request):
    path = template_dir / "outputs" / "GENE_linear" / "parameters_0001"
    norm = Normalisation("test_gk_output_gene")

    request.cls.data = GKOutputReaderGENE().read_from_file(path, norm=norm)


@pytest.mark.usefixtures("golden_answer_reference_data", "golden_answer_data")
class TestGENEGoldenAnswers:
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


@pytest.mark.parametrize(
    "load_fields",
    [True],
)
def test_amplitude(load_fields):
    path = template_dir / "outputs" / "GENE_linear"

    pyro = Pyro(gk_file=path / "parameters_0001")

    pyro.load_gk_output(load_fields=load_fields)
    eigenfunctions = pyro.gk_output.data["eigenfunctions"].isel(time=-1)
    field_squared = np.abs(eigenfunctions) ** 2

    amplitude = np.sqrt(
        field_squared.sum(dim="field").integrate(coord="theta") / (2 * np.pi)
    )
    assert hasattr(eigenfunctions.data, "units")
    assert np.isclose(ureg.Quantity(amplitude.data).magnitude, 1.0)


def test_gene_read_omega_file(tmp_path):
    """Can we read growth rate/frequency from `omega` text file"""

    shutil.copytree(template_dir / "outputs/GENE_linear", tmp_path, dirs_exist_ok=True)
    fields_file = tmp_path / "field_0001"
    fields_file.unlink()
    norm = Normalisation("test_gk_output_gene")

    data = GKOutputReaderGENE().read_from_file(tmp_path / "parameters_0001", norm=norm)
    assert np.allclose(
        data["growth_rate"].isel(time=-1, ky=0, kx=0).data.magnitude, 1.848
    )
    assert np.allclose(
        data["mode_frequency"].isel(time=-1, ky=0, kx=0).data.magnitude, 12.207
    )


def test_load_flux_spectra_linear_raises():
    """Flux spectra are a nonlinear concept in this PR; reading with
    load_flux_spectra=True on a linear run should raise NotImplementedError
    rather than silently skip or produce empty arrays."""
    path = template_dir / "outputs" / "GENE_linear" / "parameters_0001"
    norm = Normalisation("test_gk_output_gene")
    with pytest.raises(NotImplementedError, match="nonlinear"):
        GKOutputReaderGENE().read_from_file(
            path, norm=norm, load_flux_spectra=True
        )


def test_flux_spectra_container_plumbing():
    """FluxSpectra should round-trip through GKOutput with a dedicated
    flux_time coord, without disturbing existing data vars."""
    import numpy as np

    from pyrokinetics.gk_code.gk_output import Coords, FluxSpectra, GKOutput

    norm = Normalisation("test_flux_spectra_plumbing")
    nsp, nkx, nky, nt_flux = 2, 3, 4, 5
    shape = (nsp, nkx, nky, nt_flux)
    fs = FluxSpectra(
        particle_es=np.zeros(shape),
        heat_es=np.ones(shape),
        particle_em=2 * np.ones(shape),
        heat_em=3 * np.ones(shape),
    )
    coords = Coords(
        time=np.array([0.0, 1.0]),
        kx=np.linspace(0, 1, nkx),
        ky=np.linspace(0, 1, nky),
        theta=np.linspace(-np.pi, np.pi, 7),
        pitch=None,
        energy=None,
        species=np.array(["electron", "ion"]),
        field=np.array(["phi"]),
    )
    out = GKOutput(
        coords=coords,
        norm=norm,
        flux_spectra=fs,
        flux_time=np.linspace(10.0, 12.0, nt_flux),
        linear=False,
        gk_code="GENE",
    )
    assert "particle_es" in out.data.data_vars
    assert out.data["heat_em"].dims == ("species", "kx", "ky", "flux_time")
    assert out.data.sizes["flux_time"] == nt_flux
