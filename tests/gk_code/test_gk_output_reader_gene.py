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
        GKOutputReaderGENE().read_from_file(path, norm=norm, load_flux_spectra=True)


@pytest.mark.parametrize(
    "suffix",
    ["binary", "h5"],
)
def test_flux_spectra_nonlinear_cbc_matches_nrg(tmp_path, suffix):
    """End-to-end numerical check on a small CBC nonlinear fixture.

    The (kx, ky)-summed flux spectra are compared against the
    volume-integrated fluxes in ``nrg.dat`` on a shared late-time window.
    Runs twice: once with the binary ``mom_*.dat`` / ``field.dat`` /
    ``miller.dat`` files that ship with the fixture, and once with only
    their ``.dat.h5`` siblings, to exercise both readers.

    Uses the low-level helpers rather than the full ``GKOutput`` wrapper
    because the output-units path is not the thing being validated here.
    """
    import csv
    import shutil

    rundir = template_dir / "outputs" / "GENE_nonlinear_cbc"
    work = tmp_path / "cbc"
    work.mkdir()
    for f in rundir.iterdir():
        shutil.copy(f, work / f.name)

    if suffix == "h5":
        # Drop binary mom / field / geom (keep parameters and nrg) so the
        # h5 fallback path is exercised. (The fixture only ships binaries
        # since git compresses them well; for this mode we synthesise the
        # h5 files from the binaries.)
        pytest.importorskip("h5py")
        import struct

        import h5py

        complex_size = 16
        int_size = 4
        time_data_size = struct.calcsize("=idi")
        # parameters mirror the CBC fixture; hard-code to keep test
        # independent of any pyro parsing for this step.
        nx, nky, nz = 16, 4, 16
        nmoment = 6
        moment_size = nx * nky * nz * complex_size
        mom_block = time_data_size + nmoment * (2 * int_size + moment_size)
        h5_moment_names = ("dens", "T_par", "T_perp", "q_par", "q_perp", "u_par")

        for species in ("ions", "electrons"):
            binary = work / f"mom_{species}.dat"
            h5_file = work / f"mom_{species}.dat.h5"
            size = binary.stat().st_size
            ntime = size // mom_block
            times = []
            with h5py.File(h5_file, "w") as fh:
                g = fh.create_group(f"mom_{species}")
                with open(binary, "rb") as fb:
                    for it in range(ntime):
                        fb.seek(it * mom_block)
                        t_val = struct.unpack("=idi", fb.read(time_data_size))[1]
                        times.append(t_val)
                        for i_m, mname in enumerate(h5_moment_names):
                            fb.seek(int_size, 1)
                            mm = np.memmap(
                                binary,
                                dtype=np.complex128,
                                mode="r",
                                offset=fb.tell(),
                                shape=(nx, nky, nz),
                                order="F",
                            )
                            # h5 layout is (nz, nky, nkx) — swap axes
                            # 0 and 2 of the memmapped (nx, nky, nz).
                            payload = np.ascontiguousarray(
                                np.swapaxes(np.asarray(mm), 0, 2)
                            )
                            compound = np.empty(
                                payload.shape,
                                dtype=[("real", "<f8"), ("imaginary", "<f8")],
                            )
                            compound["real"] = payload.real
                            compound["imaginary"] = payload.imag
                            mom_group = g.require_group(mname)
                            mom_group.create_dataset(f"{it:010d}", data=compound)
                            fb.seek(moment_size + int_size, 1)
                g.create_dataset("time", data=np.asarray(times))
            binary.unlink()

        # Same for field.dat -> field.dat.h5 (simpler: two fields, phi+apar)
        nfield = 2
        field_size = nx * nky * nz * complex_size
        field_block = time_data_size + nfield * (2 * int_size + field_size)
        field_bin = work / "field.dat"
        field_h5 = work / "field.dat.h5"
        size = field_bin.stat().st_size
        ntime_f = size // field_block
        times_f = []
        with h5py.File(field_h5, "w") as fh:
            with open(field_bin, "rb") as fb:
                for it in range(ntime_f):
                    fb.seek(it * field_block)
                    t_val = struct.unpack("=idi", fb.read(time_data_size))[1]
                    times_f.append(t_val)
                    for i_f, fname in enumerate(("phi", "A_par")):
                        fb.seek(int_size, 1)
                        mm = np.memmap(
                            field_bin,
                            dtype=np.complex128,
                            mode="r",
                            offset=fb.tell(),
                            shape=(nx, nky, nz),
                            order="F",
                        )
                        payload = np.ascontiguousarray(
                            np.swapaxes(np.asarray(mm), 0, 2)
                        )
                        compound = np.empty(
                            payload.shape,
                            dtype=[("real", "<f8"), ("imaginary", "<f8")],
                        )
                        compound["real"] = payload.real
                        compound["imaginary"] = payload.imag
                        sub = fh.require_group(f"field/{fname}")
                        sub.create_dataset(f"{it:010d}", data=compound)
                        fb.seek(field_size + int_size, 1)
            fh.create_dataset("field/time", data=np.asarray(times_f))
        field_bin.unlink()

        # Geometry: write a minimal miller.dat.h5 with the Jacobian column.
        import f90nml

        geom_txt = work / "miller.dat"
        geom_nml = f90nml.read(geom_txt)
        skiprows = 18 + ("edge_opt" in geom_nml["parameters"])
        geom_arr = np.loadtxt(geom_txt, skiprows=skiprows)
        with h5py.File(work / "miller.dat.h5", "w") as fh:
            fh.create_dataset("Bfield_terms/Jacobian", data=geom_arr[:, -6])
        geom_txt.unlink()

    reader = GKOutputReaderGENE()
    raw_data, gk_input, _ = reader._get_raw_data(work)
    coords = reader._get_coords(raw_data, gk_input, downsample={})
    fields = reader._get_fields(raw_data, gk_input, coords, downsample={})
    raw_moms, flux_time = reader._read_raw_moments(
        raw_data, gk_input, coords, downsample={}
    )
    jac = reader._read_geom_jacobian(raw_data, gk_input)
    spectra = reader._get_flux_spectra(raw_moms, fields, jac, gk_input, coords)

    species = list(coords["species"])
    flux_time = np.asarray(flux_time)

    # Parse nrg.dat into (ntime, nspecies, 10).
    nspecies = len(species)
    rows = []
    with open(work / "nrg.dat") as f:
        nrg_reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        while True:
            try:
                t_line = next(nrg_reader)
            except StopIteration:
                break
            t = float(t_line[0])
            block = [np.asarray(next(nrg_reader), dtype=float) for _ in range(nspecies)]
            rows.append((t, np.stack(block)))
    nrg_time = np.array([r[0] for r in rows])
    nrg_vals = np.stack([r[1] for r in rows])
    nrg_cols = {
        "particle_es": nrg_vals[..., 4],
        "particle_em": nrg_vals[..., 5],
        "heat_es": nrg_vals[..., 6],
        "heat_em": nrg_vals[..., 7],
    }

    # Use the second half of the run for both cadences.
    t_lo = 0.5 * flux_time[-1]
    t_hi = flux_time[-1]
    mom_mask = (flux_time >= t_lo) & (flux_time <= t_hi)
    nrg_mask = (nrg_time >= t_lo) & (nrg_time <= t_hi)
    assert mom_mask.sum() >= 2, "need at least 2 mom snapshots in window"
    assert nrg_mask.sum() >= 2, "need at least 2 nrg samples in window"

    for var in ("particle_es", "heat_es", "particle_em", "heat_em"):
        arr = spectra[var]  # (nspec, nkx, nky, nt_flux)
        total_t = arr.sum(axis=(1, 2))  # (nspec, nt_flux)
        spec_avg = total_t[:, mom_mask].mean(axis=-1)
        nrg_avg = nrg_cols[var][nrg_mask, :].mean(axis=0)
        for i, s in enumerate(species):
            assert abs(nrg_avg[i]) > 1e-20, f"nrg {var} {s} is zero"
            ratio = spec_avg[i] / nrg_avg[i]
            # 15% tolerance accommodates sampling noise of the shipped
            # fixture (7 mom snapshots vs ~50 nrg samples, short run
            # hasn't saturated) while still catching the kinds of bugs
            # we care about — sign flips, axis misalignment, missing
            # factor of 2 (hermitian symmetry). A longer run on a real
            # box would hit ~1% (see docs/howtos/gene_flux_spectra.rst).
            assert abs(ratio - 1.0) < 0.15, (
                f"{suffix} {var} {s}: spec-sum/nrg = {ratio:.4f} "
                f"(spec={spec_avg[i]:.3e}, nrg={nrg_avg[i]:.3e})"
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
