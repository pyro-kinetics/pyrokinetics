import shutil
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.gk_code import GKOutputReaderGENE
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics.normalisation import SimulationNormalisation as Normalisation
from pyrokinetics.pyro import _find_gene_restart_paths
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


# ---------------------------------------------------------------------------
# GENE restart concatenation (load_restarts)
# ---------------------------------------------------------------------------


GENE_RESTART_PREFIXES = [
    "parameters",
    "field",
    "nrg",
    "omega",
    "mom_electron",
    "mom_ion",
    "miller",
]


def _clone_gene_run(src_dir: Path, dest_dir: Path, suffix: str) -> None:
    """Copy a GENE output directory, renaming the trailing ``_####`` suffix."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for prefix in GENE_RESTART_PREFIXES:
        for src in src_dir.glob(f"{prefix}_*"):
            shutil.copy(src, dest_dir / f"{prefix}_{suffix}")


def _clone_gene_run_dat(src_dir: Path, dest_dir: Path) -> None:
    """Copy a GENE output directory into the unnumbered ``.dat`` convention."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for prefix in GENE_RESTART_PREFIXES:
        for src in src_dir.glob(f"{prefix}_*"):
            shutil.copy(src, dest_dir / f"{prefix}.dat")


def test_find_gene_restart_paths_sorted(tmp_path):
    """Sibling restart files should be discovered and numerically sorted, with
    the unnumbered '.dat' segment appended last."""
    # Deliberately create the files out of numeric order.
    for suffix in ["0003", "0001", "0010", "0002"]:
        (tmp_path / f"parameters_{suffix}").touch()
    # Unrelated numbered files in the same directory must be ignored.
    (tmp_path / "nrg_0001").touch()
    (tmp_path / "parameters.dat").touch()

    found = _find_gene_restart_paths(tmp_path / "parameters_0001")
    assert [p.name for p in found] == [
        "parameters_0001",
        "parameters_0002",
        "parameters_0003",
        "parameters_0010",
        "parameters.dat",
    ]


def test_find_gene_restart_paths_from_bare_prefix(tmp_path):
    """A bare prefix should discover all numbered restarts and the '.dat' file."""
    for suffix in ["0002", "0001"]:
        (tmp_path / f"parameters_{suffix}").touch()
    (tmp_path / "parameters.dat").touch()
    # Unrelated prefix must not be picked up.
    (tmp_path / "nrg_0001").touch()

    found = _find_gene_restart_paths(tmp_path / "parameters")
    assert [p.name for p in found] == [
        "parameters_0001",
        "parameters_0002",
        "parameters.dat",
    ]


def test_find_gene_restart_paths_bare_prefix_only_dat(tmp_path):
    """A bare prefix with only a '.dat' segment should resolve to that file."""
    (tmp_path / "parameters.dat").touch()
    found = _find_gene_restart_paths(tmp_path / "parameters")
    assert [p.name for p in found] == ["parameters.dat"]


def test_find_gene_restart_paths_bare_prefix_no_match(tmp_path):
    """A bare prefix with no matching output returns None."""
    assert _find_gene_restart_paths(tmp_path / "parameters") is None


def test_find_gene_restart_paths_single_file(tmp_path):
    """A lone numbered file should yield itself with no siblings."""
    only = tmp_path / "parameters_0001"
    only.touch()
    assert _find_gene_restart_paths(only) == [only]


def test_find_gene_restart_paths_directory_returns_none(tmp_path):
    """Directories are handled by the regular reader, so return None."""
    (tmp_path / "parameters.dat").touch()
    assert _find_gene_restart_paths(tmp_path) is None


def test_find_gene_restart_paths_dat_file(tmp_path):
    """A '.dat' file should resolve to itself (plus any numbered siblings)."""
    (tmp_path / "parameters.dat").touch()
    found = _find_gene_restart_paths(tmp_path / "parameters.dat")
    assert [p.name for p in found] == ["parameters.dat"]


def test_find_gene_restart_paths_ignores_different_widths(tmp_path):
    """Files with differing suffix widths aren't part of the same sequence."""
    (tmp_path / "parameters_0001").touch()
    (tmp_path / "parameters_0002").touch()
    (tmp_path / "parameters_10").touch()  # different width — unrelated

    found = _find_gene_restart_paths(tmp_path / "parameters_0001")
    assert [p.name for p in found] == ["parameters_0001", "parameters_0002"]


def test_load_gk_output_load_restarts_concatenates_time(tmp_path):
    """Pyro.load_gk_output(load_restarts=True) should concatenate siblings on time.

    Copies the GENE_linear template twice (as suffixes 0001 and 0002) into a
    tmp path. Because the two copies carry identical time coordinates, the
    dedup step should collapse the concat back to the baseline length — which
    both proves discovery worked (we read both) and that the resulting time
    axis stays monotonic.
    """
    src = template_dir / "outputs/GENE_linear"

    _clone_gene_run(src, tmp_path, "0001")
    _clone_gene_run(src, tmp_path, "0002")

    # Baseline: single-file load without restart concatenation.
    pyro_single = Pyro(gk_file=tmp_path / "parameters_0001")
    pyro_single.load_gk_output(load_restarts=False)
    baseline_time = pyro_single.gk_output.data["time"].values
    baseline_ntime = len(baseline_time)

    # Confirm both restart files are discovered.
    restart_paths = _find_gene_restart_paths(tmp_path / "parameters_0001")
    assert [p.name for p in restart_paths] == ["parameters_0001", "parameters_0002"]

    # Load with restart concatenation enabled.
    pyro = Pyro(gk_file=tmp_path / "parameters_0001")
    pyro.load_gk_output(load_restarts=True)
    time = pyro.gk_output.data["time"].values

    # Duplicate timestamps across identical copies should collapse to the
    # original length, leaving the time axis strictly monotonically increasing.
    assert len(time) == baseline_ntime
    assert np.all(np.diff(time) > 0)
    np.testing.assert_allclose(time, baseline_time)


def test_load_gk_output_load_restarts_bare_prefix_with_dat(tmp_path):
    """Passing the bare prefix should load both numbered restarts and the
    '.dat' segment and concatenate them along time."""
    src = template_dir / "outputs/GENE_linear"

    # A numbered restart and an unnumbered '.dat' segment side by side.
    _clone_gene_run(src, tmp_path, "0001")
    _clone_gene_run_dat(src, tmp_path)

    # The bare prefix should discover the numbered restart and the '.dat' file.
    restart_paths = _find_gene_restart_paths(tmp_path / "parameters")
    assert [p.name for p in restart_paths] == ["parameters_0001", "parameters.dat"]

    # Loading the '.dat' segment on its own must also work via the reader.
    pyro_dat = Pyro(gk_file=tmp_path / "parameters_0001")
    pyro_dat.load_gk_output(path=tmp_path / "parameters.dat", load_restarts=False)
    assert pyro_dat.gk_output.data["time"].size > 0

    # Load everything via the bare prefix.
    pyro = Pyro(gk_file=tmp_path / "parameters_0001")
    pyro.load_gk_output(path=tmp_path / "parameters", load_restarts=True)
    time = pyro.gk_output.data["time"].values

    # The two segments are identical copies, so dedup collapses to one run's
    # length and the time axis stays strictly monotonic.
    assert time.size > 0
    assert np.all(np.diff(time) > 0)


def test_pyroscan_load_restarts_default_true():
    """PyroScan.load_gk_output should default to load_restarts=True so that
    GENE scans with unequal run lengths are concatenated automatically."""
    import inspect

    from pyrokinetics.pyroscan import PyroScan

    sig = inspect.signature(PyroScan.load_gk_output)
    assert sig.parameters["load_restarts"].default is True


def test_pyro_load_restarts_default_false():
    """A bare Pyro.load_gk_output should keep the conservative default of
    load_restarts=False — callers opt in explicitly."""
    import inspect

    from pyrokinetics import Pyro as PyroCls

    sig = inspect.signature(PyroCls.load_gk_output)
    assert sig.parameters["load_restarts"].default is False
