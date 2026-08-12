"""
Tests for :class:`~pyrokinetics.pyrohypercube.PyroHypercube`.

The reference points are ``PyroScan`` (a hypercube must agree with it wherever
both apply — namely on a grid) and a run tree pyrokinetics did not write (which
is the case ``PyroScan`` cannot handle at all).
"""

import shutil

import numpy as np
import pytest

from pyrokinetics import Pyro, PyroHypercube, PyroScan, template_dir
from pyrokinetics.units import ureg as units

GS2_TEMPLATE_DIR = template_dir / "outputs" / "GS2_linear"
CGYRO_SCAN_DIR = template_dir / "outputs" / "CGYRO_linear_scan"

# The values written into the foreign GS2 tree, in pyrokinetics units
FOREIGN_KY = [0.1, 0.2, 0.3]


@pytest.fixture(scope="module")
def gs2_run_tree(tmp_path_factory):
    """
    A directory of GS2 runs that pyrokinetics did not lay out.

    The run directories are named by iteration rather than by parameter value,
    there is no ``pyroscan.json``, and the tree holds directories that are not
    runs at all — so nothing about it can be assumed, only read.
    """
    root = tmp_path_factory.mktemp("gs2_run_tree")

    for i, ky in enumerate(FOREIGN_KY):
        run_directory = root / f"iteration_{i}"
        pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
        pyro.numerics.ky = ky / units.rhoref_pyro
        pyro.write_gk_file(file_name=run_directory / "gs2.in")
        # A real output for each run, so gk_output loading has something to read
        shutil.copy(GS2_TEMPLATE_DIR / "gs2.out.nc", run_directory / "gs2.out.nc")

    # Decoys: a directory holding no input file, and a stray file
    (root / "plots").mkdir()
    (root / "iteration_notes.txt").write_text("not a run\n")

    return root


@pytest.fixture(scope="module")
def gs2_hypercube(gs2_run_tree):
    return PyroHypercube.from_directory(
        gs2_run_tree, pattern="iteration_*", params=["ky"], gk_code="GS2"
    )


# ---------------------------------------------------------------------------
# Samples, not a grid
# ---------------------------------------------------------------------------
def test_samples_are_paired_not_multiplied():
    """N values of two parameters make N runs, not N**2."""
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    parameter_dict = {
        "ky": np.array([0.1, 0.2, 0.3]) / units.rhoref_pyro,
        "electron_temp_gradient": np.array([1.0, 2.0, 3.0]) / units.lref_minor_radius,
    }
    cube = PyroHypercube(pyro, parameter_dict=parameter_dict)

    assert cube.n_samples == 3
    assert len(cube.pyro_dict) == 3

    samples = list(cube.sample_points())
    assert len(samples) == 3
    for i, sample in enumerate(samples):
        assert np.isclose(sample["ky"].magnitude, parameter_dict["ky"][i].magnitude)
        assert np.isclose(
            sample["electron_temp_gradient"].magnitude,
            parameter_dict["electron_temp_gradient"][i].magnitude,
        )

    # The same parameter_dict as a PyroScan is a 3x3 grid
    assert len(PyroScan(pyro, parameter_dict=parameter_dict).pyro_dict) == 9


def test_ragged_parameters_rejected():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    with pytest.raises(ValueError, match="one value per sample"):
        PyroHypercube(
            pyro,
            parameter_dict={
                "ky": np.array([0.1, 0.2, 0.3]) / units.rhoref_pyro,
                "kappa": np.array([1.0, 2.0]),
            },
        )


def test_duplicate_sample_names_rejected():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    with pytest.raises(ValueError, match="unique"):
        PyroHypercube(
            pyro,
            parameter_dict={"ky": np.array([0.1, 0.2]) / units.rhoref_pyro},
            sample_names=["a", "a"],
        )


def test_default_sample_names():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    cube = PyroHypercube(
        pyro, parameter_dict={"ky": np.array([0.1, 0.2]) / units.rhoref_pyro}
    )
    assert cube.sample_names == ["sample_0000", "sample_0001"]
    assert list(cube.pyro_dict) == cube.sample_names


# ---------------------------------------------------------------------------
# Reading a run tree pyrokinetics did not write
# ---------------------------------------------------------------------------
def test_from_directory_recovers_varied_values(gs2_hypercube):
    cube = gs2_hypercube

    assert cube.sample_names == ["iteration_0", "iteration_1", "iteration_2"]
    assert cube.n_samples == 3

    ky = cube.parameter_dict["ky"]
    assert np.allclose(ky.magnitude, FOREIGN_KY)
    assert ky.units == units.rhoref_pyro**-1

    # Every sample keeps the Pyro read from its own input file
    for name, pyro in cube.pyro_dict.items():
        assert pyro.gk_file.parent.name == name
        assert pyro.gk_code == "GS2"


def test_from_directory_matches_input_files(gs2_run_tree):
    """A pattern may match the input files themselves rather than directories."""
    cube = PyroHypercube.from_directory(
        gs2_run_tree, pattern="*/gs2.in", params=["ky"], gk_code="GS2"
    )
    assert cube.sample_names == ["iteration_0", "iteration_1", "iteration_2"]
    assert np.allclose(cube.parameter_dict["ky"].magnitude, FOREIGN_KY)


def test_from_directory_nested_tree(gs2_run_tree, tmp_path):
    """
    Runs nested under several model directories make one set of samples.

    This is the shape of the databases this exists for: ``MTM_MODELS/iteration_N``
    beside ``KB_MODELS/iteration_N``, read together and kept apart by name.
    """
    nested = tmp_path / "nested"
    for model in ("MTM_MODELS", "KB_MODELS"):
        for i in range(len(FOREIGN_KY)):
            shutil.copytree(
                gs2_run_tree / f"iteration_{i}", nested / model / f"iteration_{i}"
            )

    cube = PyroHypercube.from_directory(
        nested, pattern="*/iteration_*", params=["ky"], gk_code="GS2"
    )

    assert cube.sample_names == [
        f"{model}/iteration_{i}"
        for model in ("KB_MODELS", "MTM_MODELS")
        for i in range(len(FOREIGN_KY))
    ]
    assert np.allclose(cube.parameter_dict["ky"].magnitude, FOREIGN_KY * 2)

    cube.load_gk_output()
    assert cube.gk_output.data["growth_rate"].shape == (6,)

    # Decks mirror the tree they came from
    written = cube.write_gk_decks("TGLF", target=nested / "decks")
    assert [
        str(path.parent.relative_to(nested / "decks")) for path in written
    ] == cube.sample_names


def test_from_directory_explicit_parameter_location(gs2_run_tree):
    """Parameters outside the default map are given as attribute + location."""
    cube = PyroHypercube.from_directory(
        gs2_run_tree,
        pattern="iteration_*",
        params={"beta": ["numerics", ["beta"]]},
        gk_code="GS2",
    )
    assert list(cube.parameter_dict) == ["beta"]
    assert len(cube.parameter_dict["beta"]) == 3


def test_from_directory_unknown_parameter(gs2_run_tree):
    with pytest.raises(ValueError, match="don't know where to find"):
        PyroHypercube.from_directory(
            gs2_run_tree, pattern="iteration_*", params=["not_a_parameter"]
        )


def test_from_directory_requires_params(gs2_run_tree):
    with pytest.raises(ValueError, match="params"):
        PyroHypercube.from_directory(gs2_run_tree, pattern="iteration_*")


def test_from_directory_no_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        PyroHypercube.from_directory(tmp_path, pattern="*", params=["ky"])


def test_load_gk_output_on_sample_dimension(gs2_hypercube):
    cube = gs2_hypercube
    cube.load_gk_output()
    data = cube.gk_output.data

    assert data.sizes["sample"] == 3
    assert data["growth_rate"].dims == ("sample",)

    # The varied parameter rides along the sample dimension rather than being
    # a dimension of its own — there is no grid for it to span.
    assert "ky" not in data.dims
    assert data["ky"].dims == ("sample",)
    assert np.allclose(data["ky"].data.magnitude, FOREIGN_KY)
    assert data["ky"].data.units == units.rhoref_pyro**-1

    # Each row says which run it came from
    assert list(data["sample_name"].values) == cube.sample_names


# ---------------------------------------------------------------------------
# Agreement with PyroScan on a grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("output_convention", ["pyrokinetics", "cgyro"])
def test_matches_pyroscan_on_a_gridded_scan(output_convention):
    """
    Pointed at a PyroScan directory, a hypercube must reproduce it exactly.

    A non-gridded reader that cannot reproduce the gridded one on a grid is
    broken: the only difference allowed is the shape of the output.
    """
    scan = PyroScan(pyroscan_json=CGYRO_SCAN_DIR / "pyroscan.json", load_base_pyro=True)
    scan.load_gk_output(output_convention=output_convention)

    cube = PyroHypercube.from_directory(
        CGYRO_SCAN_DIR, pattern="ky_*", params=["ky"], gk_code="CGYRO"
    )
    cube.load_gk_output(output_convention=output_convention)

    grid = scan.gk_output.data
    flat = cube.gk_output.data

    assert grid["growth_rate"].dims == ("ky",)
    assert flat["growth_rate"].dims == ("sample",)
    assert list(flat["sample_name"].values) == ["ky_0.10", "ky_0.20", "ky_0.30"]

    grid_ky = grid["ky"].data * grid["ky"].attrs["units"]
    assert np.array_equal(grid_ky.magnitude, flat["ky"].data.magnitude)
    assert grid_ky.units == flat["ky"].data.units

    for name in ("growth_rate", "mode_frequency", "heat", "particle"):
        expected = grid[name].data
        actual = flat[name].data
        assert expected.units == actual.units, name
        assert np.array_equal(
            np.asarray(expected.magnitude).ravel(),
            np.asarray(actual.magnitude).ravel(),
        ), name


# ---------------------------------------------------------------------------
# netCDF
# ---------------------------------------------------------------------------
def test_netcdf_roundtrip(gs2_run_tree, tmp_path):
    cube = PyroHypercube.from_directory(
        gs2_run_tree, pattern="iteration_*", params=["ky"], gk_code="GS2"
    )
    cube.load_gk_output()
    before = cube.gk_output.data

    netcdf_file = tmp_path / "hypercube.nc"
    cube.to_netcdf(netcdf_file)

    reloaded = PyroHypercube.from_directory(
        gs2_run_tree, pattern="iteration_*", params=["ky"], gk_code="GS2"
    )
    reloaded.from_netcdf(netcdf_file)
    after = reloaded.gk_output.data

    assert after.sizes["sample"] == before.sizes["sample"]
    assert list(after["sample_name"].values) == list(before["sample_name"].values)
    assert np.array_equal(after["ky"].data.magnitude, before["ky"].data.magnitude)
    assert after["ky"].data.units == before["ky"].data.units

    for name in ("growth_rate", "mode_frequency", "heat"):
        assert before[name].data.units == after[name].data.units, name
        assert np.array_equal(
            np.asarray(before[name].data.magnitude),
            np.asarray(after[name].data.magnitude),
        ), name


def test_to_netcdf_without_output(gs2_hypercube, tmp_path):
    cube = PyroHypercube(
        gs2_hypercube.base_pyro,
        parameter_dict={"ky": np.array([0.1, 0.2]) / units.rhoref_pyro},
    )
    with pytest.raises(RuntimeError, match="no output loaded"):
        cube.to_netcdf(tmp_path / "empty.nc")


# ---------------------------------------------------------------------------
# Writing input files
# ---------------------------------------------------------------------------
def test_write_tglf_decks_from_gs2_hypercube(gs2_hypercube, tmp_path):
    target = tmp_path / "tglf"
    written = gs2_hypercube.write_gk_decks(
        "TGLF",
        overrides={"NBASIS_MAX": 6, "WIDTH": 3.0, "USE_BPER": True},
        target=target,
    )

    assert [path.parent.name for path in written] == gs2_hypercube.sample_names
    assert all(path.is_file() for path in written)

    # The source tree is untouched: writing decks writes decks, and runs nothing
    assert sorted(p.name for p in written[0].parent.iterdir()) == ["input.TGLF"]

    for path, ky in zip(written, FOREIGN_KY):
        settings = dict(
            line.split(" = ", 1)
            for line in path.read_text().splitlines()
            if " = " in line
        )
        # Overrides are matched to the case TGLF itself uses, so each appears once
        assert list(settings).count("NBASIS_MAX") == 1
        assert settings["NBASIS_MAX"] == "6"
        assert settings["WIDTH"] == "3.0"
        assert settings["USE_BPER"] == "T"
        # ...and each sample keeps its own varied value
        assert float(settings["KY"]) == pytest.approx(ky, rel=1e-6)


def test_write_gk_decks_requires_target(gs2_hypercube):
    with pytest.raises(ValueError, match="target"):
        gs2_hypercube.write_gk_decks("TGLF")


def test_write_gk_decks_same_code(gs2_hypercube, tmp_path):
    """No gk_code given: write the samples out in their own code."""
    written = gs2_hypercube.write_gk_decks(target=tmp_path / "gs2", file_name="gs2.in")
    assert len(written) == 3
    for path, ky in zip(written, FOREIGN_KY):
        pyro = Pyro(gk_file=path)
        convention = pyro.norms.pyrokinetics
        assert pyro.numerics.ky.to(convention, convention.context).magnitude == (
            pytest.approx(ky)
        )


# ---------------------------------------------------------------------------
# Round trip through pyroscan.json
# ---------------------------------------------------------------------------
def test_write_and_reload(tmp_path):
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    rng = np.random.default_rng(0)
    parameter_dict = {
        "ky": rng.uniform(0.1, 0.5, 5) / units.rhoref_pyro,
        "electron_temp_gradient": rng.uniform(1.0, 4.0, 5) / units.lref_minor_radius,
    }
    cube = PyroHypercube(pyro, parameter_dict=parameter_dict, base_directory=tmp_path)
    cube.write(file_name="gs2.in")

    reloaded = PyroHypercube(cube.base_pyro, pyroscan_json=tmp_path / "pyroscan.json")

    assert reloaded.sample_names == cube.sample_names
    assert reloaded.n_samples == 5
    for name, values in cube.parameter_dict.items():
        assert np.allclose(
            np.asarray(reloaded.parameter_dict[name].magnitude),
            np.asarray(values.magnitude),
        )

    # Each sample's input file carries that sample's values, not the base's
    for i, name in enumerate(cube.sample_names):
        written = Pyro(gk_file=tmp_path / name / "gs2.in")
        convention = written.norms.pyrokinetics
        assert written.numerics.ky.to(convention, convention.context).magnitude == (
            pytest.approx(parameter_dict["ky"][i].magnitude)
        )
