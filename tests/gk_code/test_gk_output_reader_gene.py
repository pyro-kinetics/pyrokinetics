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


def _write_cbc_hdf5(work):
    """Rewrite the binary parts of a copied ``GENE_nonlinear_cbc`` run as their
    HDF5 equivalents, so the HDF5 readers can be exercised on the same data.

    The fixture only ships the binary files (git compresses them well), so the
    ``mom_*``, ``field`` and geometry files are converted here and the binaries
    removed. ``parameters`` and ``nrg`` are left as text, as GENE writes them
    alongside the HDF5 output.
    """
    import struct

    import h5py

    complex_size = 16
    int_size = 4
    time_data_size = struct.calcsize("=idi")
    # parameters mirror the CBC fixture; hard-code to keep the conversion
    # independent of any pyro parsing for this step.
    nx, nky, nz = 16, 4, 16

    def read_block(binary, offset, dtype=np.complex128):
        mm = np.memmap(
            binary, dtype=dtype, mode="r", offset=offset, shape=(nx, nky, nz), order="F"
        )
        # HDF5 layout is (nz, nky, nkx) with a compound {real, imaginary} dtype
        payload = np.ascontiguousarray(np.swapaxes(np.asarray(mm), 0, 2))
        compound = np.empty(
            payload.shape, dtype=[("real", "<f8"), ("imaginary", "<f8")]
        )
        compound["real"] = payload.real
        compound["imaginary"] = payload.imag
        return compound

    moment_names = ("dens", "T_par", "T_perp", "q_par", "q_perp", "u_par")
    block_size = nx * nky * nz * complex_size
    mom_block = time_data_size + len(moment_names) * (2 * int_size + block_size)

    for species in ("ions", "electrons"):
        binary = work / f"mom_{species}.dat"
        times = []
        with h5py.File(work / f"mom_{species}.dat.h5", "w") as fh:
            group = fh.create_group(f"mom_{species}")
            with open(binary, "rb") as f:
                for it in range(binary.stat().st_size // mom_block):
                    f.seek(it * mom_block)
                    times.append(struct.unpack("=idi", f.read(time_data_size))[1])
                    for name in moment_names:
                        f.seek(int_size, 1)
                        group.require_group(name).create_dataset(
                            f"{it:010d}", data=read_block(binary, f.tell())
                        )
                        f.seek(block_size + int_size, 1)
            group.create_dataset("time", data=np.asarray(times))
        binary.unlink()

    field_names = ("phi", "A_par")
    field_block = time_data_size + len(field_names) * (2 * int_size + block_size)
    binary = work / "field.dat"
    times = []
    with h5py.File(work / "field.dat.h5", "w") as fh:
        with open(binary, "rb") as f:
            for it in range(binary.stat().st_size // field_block):
                f.seek(it * field_block)
                times.append(struct.unpack("=idi", f.read(time_data_size))[1])
                for name in field_names:
                    f.seek(int_size, 1)
                    fh.require_group(f"field/{name}").create_dataset(
                        f"{it:010d}", data=read_block(binary, f.tell())
                    )
                    f.seek(block_size + int_size, 1)
        fh.create_dataset("field/time", data=np.asarray(times))
    binary.unlink()

    # Geometry: only the Jacobian column is needed by the flux spectra
    import f90nml

    geometry = work / "miller.dat"
    skiprows = 18 + ("edge_opt" in f90nml.read(geometry)["parameters"])
    jacobian = np.loadtxt(geometry, skiprows=skiprows)[:, -6]
    with h5py.File(work / "miller.dat.h5", "w") as fh:
        fh.create_dataset("Bfield_terms/Jacobian", data=jacobian)
    geometry.unlink()


@pytest.fixture(params=["binary", "h5"])
def cbc_run_dir(request, tmp_path):
    """A writable copy of the nonlinear CBC fixture, in either the binary or
    the HDF5 flavour."""
    work = tmp_path / "cbc"
    work.mkdir()
    for f in (template_dir / "outputs" / "GENE_nonlinear_cbc").iterdir():
        shutil.copy(f, work / f.name)
    if request.param == "h5":
        pytest.importorskip("h5py")
        _write_cbc_hdf5(work)
    return work


def _read_cbc(path, name, **kwargs):
    # The CBC fixture carries no reference values, so it can only be expressed
    # in GENE's own convention.
    return GKOutputReaderGENE().read_from_file(
        path,
        norm=Normalisation(name),
        output_convention="gene",
        **kwargs,
    )


def test_kxky_flux_spectra_matches_nrg(cbc_run_dir):
    """The (kx, ky)-resolved fluxes computed from the moments and fields should
    sum to the volume-integrated fluxes GENE writes to ``nrg``."""
    spectra = _read_cbc(cbc_run_dir, "test_kxky_spectra", kxky_flux_spectra=True)
    nrg = _read_cbc(cbc_run_dir, "test_kxky_nrg")

    assert spectra.data["heat"].dims == ("field", "species", "kx", "ky", "time")
    assert spectra.data["particle"].dims == ("field", "species", "kx", "ky", "time")
    # GENE writes no momentum moments, so no momentum spectrum is built
    assert "momentum" not in spectra.data.data_vars
    assert list(spectra.data.coords["flux"].values) == ["particle", "heat"]
    np.testing.assert_allclose(
        ureg.Quantity(spectra.data["time"].data).magnitude,
        ureg.Quantity(nrg.data["time"].data).magnitude,
    )

    for var in ("particle", "heat"):
        for field in ("phi", "apar"):
            summed = spectra.data[var].sel(field=field).sum(dim=["kx", "ky"])
            np.testing.assert_allclose(
                ureg.Quantity(summed.data).magnitude,
                ureg.Quantity(nrg.data[var].sel(field=field).data).magnitude,
                rtol=1e-3,
                err_msg=f"{var} ({field}) spectrum does not sum to the nrg flux",
            )


def test_load_moments_cbc(cbc_run_dir):
    """Moments load from both the binary and HDF5 layouts."""
    out = _read_cbc(cbc_run_dir, "test_cbc_moments", load_moments=True)
    assert out.data["density"].dims == ("theta", "kx", "species", "ky", "time")
    assert out.data["density"].shape == (16, 16, 2, 4, 13)


def test_kxky_flux_spectra_linear_raises():
    """Flux spectra are a nonlinear concept: a linear run holds a single mode,
    so asking for a (kx, ky) spectrum should raise rather than silently return
    something meaningless."""
    path = template_dir / "outputs" / "GENE_linear" / "parameters_0001"
    norm = Normalisation("test_kxky_linear")
    with pytest.raises(NotImplementedError, match="nonlinear"):
        GKOutputReaderGENE().read_from_file(path, norm=norm, kxky_flux_spectra=True)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"load_fields": False}, ValueError, "load_fields"),
        ({"load_fluxes": False}, ValueError, "load_fluxes"),
    ],
)
def test_kxky_flux_spectra_requires_fields_and_fluxes(kwargs, error, match):
    path = template_dir / "outputs" / "GENE_nonlinear_cbc"
    norm = Normalisation(f"test_kxky_{'_'.join(kwargs)}")
    with pytest.raises(error, match=match):
        GKOutputReaderGENE().read_from_file(
            path,
            norm=norm,
            output_convention="gene",
            kxky_flux_spectra=True,
            **kwargs,
        )


def test_kxky_flux_spectra_without_moments_raises(tmp_path):
    """Moment files are the input to the spectra, so their absence should be
    reported plainly rather than as a shape or key error."""
    work = tmp_path / "no_moments"
    work.mkdir()
    for f in (template_dir / "outputs" / "GENE_nonlinear_cbc").iterdir():
        if not f.name.startswith("mom_"):
            shutil.copy(f, work / f.name)

    with pytest.raises(FileNotFoundError, match="mom_"):
        _read_cbc(work, "test_kxky_no_moments", kxky_flux_spectra=True)
