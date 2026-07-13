import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.pyroscan import PyroScan
from pyrokinetics.units import ureg as units

docs_dir = Path(__file__).parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_SCENE  # noqa


@pytest.fixture(scope="module")
def nonlinear_tmp_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_pyroscan_gk_output_reader")
    return tmp_dir


@pytest.mark.parametrize(
    "json_dir  ",
    [
        ("CGYRO_linear_scan"),
    ],
)
def test_pyroscan_read_linear(json_dir):
    json_path = template_dir / "outputs" / json_dir
    pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)

    pyro_scan.load_gk_output()
    assert "growth_rate" in pyro_scan.gk_output.data.data_vars
    assert "mode_frequency" in pyro_scan.gk_output.data.data_vars
    # Default reduces over time, so no 'time' dimension is kept.
    assert "time" not in pyro_scan.gk_output.data["growth_rate"].dims


def test_pyroscan_keep_time_default_false():
    """PyroScan.load_gk_output should default to keep_time=False."""
    import inspect

    sig = inspect.signature(PyroScan.load_gk_output)
    assert sig.parameters["keep_time"].default is False


def test_pyroscan_keep_time_retains_whole_trace():
    """keep_time=True retains the full time trace, with shorter runs NaN-padded
    so each scan point's trace matches that run's own output."""
    json_path = template_dir / "outputs" / "CGYRO_linear_scan"

    # Gather each run's true growth_rate trace (kx-min selected).
    ref_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)
    per_run = []
    for pyro in ref_scan.pyro_dict.values():
        pyro.load_gk_output()
        d = pyro.gk_output.data
        kx_min = float(np.min(np.abs(d.kx)))
        per_run.append(
            np.asarray(d["growth_rate"].sel(kx=kx_min).data.magnitude).ravel()
        )
        pyro.gk_output = None
    max_ntime = max(len(t) for t in per_run)

    pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)
    pyro_scan.load_gk_output(keep_time=True)

    gr = pyro_scan.gk_output.data["growth_rate"]
    assert "time" in gr.dims
    assert gr.sizes["time"] == max_ntime

    mag = np.asarray(gr.data.magnitude if hasattr(gr.data, "magnitude") else gr.data)
    # dims are (ky, time); ky is the scan axis in pyro_dict order.
    for i, trace in enumerate(per_run):
        row = mag[i]
        np.testing.assert_allclose(row[: len(trace)], trace)
        # Shorter runs are NaN-padded at the tail.
        assert np.all(np.isnan(row[len(trace) :]))


@pytest.mark.parametrize(
    " json_dir, zip_path",
    [
        (
            "TGLF_transport_scan",
            template_dir / "outputs" / "TGLF_transport_scan" / "pyroscan_nonlinear.zip",
        ),
    ],
)
def test_pyroscan_read_tglf_nonlinear(json_dir, zip_path, nonlinear_tmp_path):
    json_path = nonlinear_tmp_path / json_dir
    shutil.unpack_archive(zip_path, json_path)
    pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)

    pyro_scan.load_gk_output(load_fields=False)
    data = pyro_scan.gk_output.data
    assert "phi" not in data.data_vars
    assert "bpar" not in data.data_vars
    assert "apar" not in data.data_vars
    assert "ky" in data.coords
    assert data.coords["ky"].size > 0
    assert "gamma_exb" in data.coords
    assert data.coords["gamma_exb"].size > 0
    assert len(data.coords["gamma_exb"].attrs) > 0
    assert "field" in data.coords
    assert data.coords["field"].size > 0

    pyro_scan.load_gk_output(load_fluxes=False)
    data = pyro_scan.gk_output.data
    assert "particle" not in data.data_vars
    assert "heat" not in data.data_vars
    assert "momentum" not in data.data_vars

    pyro_scan.load_gk_output(load_fields=True)
    assert "phi" in pyro_scan.gk_output.data.data_vars

    pyro_scan.load_gk_output(load_fluxes=True)
    data = pyro_scan.gk_output.data
    assert "particle" in data.data_vars
    assert "heat" in data.data_vars
    assert "momentum" in data.data_vars


@pytest.mark.parametrize(
    " json_dir, zip_path",
    [
        (
            "CGYRO_nonlinear_scan",
            template_dir
            / "outputs"
            / "CGYRO_nonlinear_scan"
            / "pyroscan_nonlinear.zip",
        ),
    ],
)
def test_pyroscan_read_nonlinear(json_dir, zip_path, nonlinear_tmp_path):
    json_path = nonlinear_tmp_path / json_dir
    shutil.unpack_archive(zip_path, json_path)
    pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)

    pyro_scan.load_gk_output(load_fields=False)
    assert "phi" not in pyro_scan.gk_output.data.data_vars
    assert "bpar" not in pyro_scan.gk_output.data.data_vars
    assert "apar" not in pyro_scan.gk_output.data.data_vars

    pyro_scan.load_gk_output(load_fluxes=False)
    assert "particle" not in pyro_scan.gk_output.data.data_vars
    assert "heat" not in pyro_scan.gk_output.data.data_vars
    assert "momentum" not in pyro_scan.gk_output.data.data_vars

    pyro_scan.load_gk_output(load_fields=True)
    assert "phi" in pyro_scan.gk_output.data.data_vars

    pyro_scan.load_gk_output(load_fluxes=True)
    assert "particle" in pyro_scan.gk_output.data.data_vars
    assert "heat" in pyro_scan.gk_output.data.data_vars
    assert "momentum" in pyro_scan.gk_output.data.data_vars


def test_pyroscan_netcdf_with_physical_units(nonlinear_tmp_path):
    """
    Saving a PyroScan ``gk_output`` with physical units produces a netCDF
    whose unit strings include the base pyro's normalisation run name
    (e.g. ``nref_electron_pyroscan_base0000``). Reloading via a pyroscan
    built from the original CGYRO input — which does not know about those
    run-specific units — must still work.

    Fix contract:
      * Saved netCDF carries generic simulation units (no run-name suffix).
      * Loading a netCDF auto-detects ``pyroscan_norms.json`` next to the
        scan so the base pyro's references can be restored.
    """
    import netCDF4 as nc

    zip_path = (
        template_dir
        / "outputs"
        / "CGYRO_nonlinear_scan_units"
        / "pyroscan_nonlinear_units.zip"
    )
    scan_dir = nonlinear_tmp_path / "scan_with_units"
    shutil.unpack_archive(zip_path, scan_dir)

    ps = PyroScan(pyroscan_json=scan_dir / "pyroscan.json", load_base_pyro=True)
    ps.load_gk_output()

    nc_path = scan_dir / "pyroscan_output.nc"
    ps.gk_output.to_netcdf(nc_path)

    # The saved netCDF must not carry unit strings tied to the writer's
    # normalisation run name — those cannot be resolved by a fresh pyro.
    writer_name = ps.base_pyro.norms.name
    with nc.Dataset(nc_path) as dset:
        for vname, var in dset.variables.items():
            unit_str = getattr(var, "units", "")
            assert writer_name not in unit_str, (
                f"Variable '{vname}' saved with run-specific unit "
                f"suffix: '{unit_str}'"
            )

    # Reload via a pyro built from the original CGYRO nonlinear input
    # that sits alongside the scan fixture (NOT the pyroscan_base.input
    # inside the zip). This represents the user's on-disk CGYRO input
    # and has no reference values of its own — the fix must recover
    # them from pyroscan_norms.json in the scan directory.
    orig_cgyro_input = (
        template_dir / "outputs" / "CGYRO_nonlinear_scan_units" / "input.cgyro"
    )
    fresh_pyro = Pyro(gk_file=orig_cgyro_input)
    alte_vals = np.array([2.0, 3.0, 4.0]) / fresh_pyro.norms.units.lref_minor_radius
    fresh_ps = PyroScan(
        fresh_pyro,
        parameter_dict={"alte": alte_vals},
        base_directory=scan_dir,
    )
    fresh_ps.add_parameter_key("alte", "local_species", ["electron", "inverse_lt"])

    fresh_ps.load_gk_output(netcdf_file=nc_path)

    loaded = fresh_ps.gk_output.data
    assert "particle" in loaded.data_vars
    assert "heat" in loaded.data_vars
    assert "phi" in loaded.data_vars


@pytest.mark.parametrize(
    "json_dir, zip_path",
    [
        (
            "TGLF_transport_scan",
            template_dir / "outputs" / "TGLF_transport_scan" / "pyroscan_nonlinear.zip",
        ),
    ],
)
def test_pyroscan_read_faulty(json_dir, zip_path, nonlinear_tmp_path):
    json_path = nonlinear_tmp_path / json_dir
    shutil.unpack_archive(zip_path, json_path)
    pyro_scan = PyroScan(
        pyroscan_json=json_path / "pyroscan_faulty.json", load_base_pyro=True
    )

    pyro_scan.load_gk_output()
    print(pyro_scan.gk_output)
    assert "growth_rate" in pyro_scan.gk_output.data.data_vars
    assert not all(
        x is None for x in pyro_scan.gk_output.data["growth_rate"].sel(mode=0)
    )


def assert_close_or_equal(attr, left_pyroscan, right_pyroscan):
    left = getattr(left_pyroscan, attr)
    right = getattr(right_pyroscan, attr)

    if attr == "parameter_dict":
        assert left.keys() == right.keys()
        norms = left_pyroscan.base_pyro.norms
        for left_value, right_value in zip(left.values(), right.values()):
            # write() always converts to pyrokinetics simulation units, so apply
            # the same conversion to the left side before comparing.
            if hasattr(left_value, "convert_physical_units"):
                left_mag = left_value.convert_physical_units(norms).magnitude
            else:
                left_mag = getattr(left_value, "magnitude", left_value)
            right_mag = getattr(right_value, "magnitude", right_value)
            assert np.allclose(left_mag, right_mag)
    elif attr == "pyroscan_json":
        norms = left_pyroscan.base_pyro.norms
        for json_key in left.keys():
            if json_key == "parameter_dict":
                assert left[json_key].keys() == right[json_key].keys()
                for left_value, right_value in zip(
                    left[json_key].values(), right[json_key].values()
                ):
                    if hasattr(left_value, "convert_physical_units"):
                        left_mag = left_value.convert_physical_units(norms).magnitude
                    else:
                        left_mag = getattr(left_value, "magnitude", left_value)
                    right_mag = getattr(right_value, "magnitude", right_value)
                    assert np.allclose(left_mag, right_mag)
            else:
                assert json_key in right.keys()
                if isinstance(left[json_key], (str, list, type(None), dict, Path)):
                    assert np.all(left[json_key] == right[json_key])
                else:
                    assert np.allclose(
                        left[json_key], right[json_key]
                    ), f"{left} != {right}"
    else:
        if isinstance(left, (str, list, type(None), dict, Path)):
            assert np.all(left == right)
        else:
            assert np.allclose(left, right), f"{left} != {right}"


PYROSCAN_CONFIGS = [
    # Simple numerics scan
    {"parameter_dict": {"ky": np.array([0.1, 0.2])}, "runfile_dict": None},
    # Local geometry scan
    {"parameter_dict": {"kappa": np.array([0.1, 0.2, 0.3])}, "runfile_dict": None},
    # Species gradient scan
    {
        "parameter_dict": {"electron_temp_gradient": np.array([1.0, 2.0])},
        "runfile_dict": None,
    },
    # Runfile dict with string keys — tests new feature
    {
        "parameter_dict": {"beta": np.array([0.005, 0.01])},
        "runfile_dict": {
            "beta_0.005": "this_file_has_a_beta_005",
            "beta_0.01": "this_file_has_a_beta_010",
        },
        "parameter_keys": [
            {"key": "beta", "attr": "numerics", "location": ["beta"]},
        ],
    },
    {
        "parameter_dict": {
            # Typical ky unit: 1/rho_ref in GENE/GS2
            "ky": np.array([0.1, 0.2])
            / units.rhoref_pyro
        },
        "runfile_dict": None,
    },
]


@pytest.fixture(params=PYROSCAN_CONFIGS)
def param_pyroscan(request, tmp_path):
    cfg = request.param

    pyro = example_SCENE.main(tmp_path)

    ps = PyroScan(
        pyro,
        parameter_dict=cfg["parameter_dict"],
        runfile_dict=cfg.get("runfile_dict"),
        base_directory=tmp_path,
    )

    # Apply additional parameter key mappings if provided
    for pk in cfg.get("parameter_keys", []):
        ps.add_parameter_key(
            parameter_key=pk["key"],
            parameter_attr=pk["attr"],
            parameter_location=pk["location"],
        )

    return ps


def test_read_write_parametrized(param_pyroscan, tmp_path):
    """Runs read/write comparison on all parametrized PyroScan configurations."""

    ps = param_pyroscan

    # Write the pyroscan
    ps.write(file_name="param_test.in", base_directory=tmp_path)

    # Read it back
    loaded = PyroScan(ps.base_pyro, pyroscan_json=tmp_path / "pyroscan.json")

    for attr in [
        "base_directory",
        "file_name",
        "p_prime_type",
        "parameter_dict",
        "parameter_map",
        "parameter_separator",
        "pyroscan_json",
        "run_directories",
        "value_fmt",
        "value_size",
    ]:
        assert_close_or_equal(attr, ps, loaded)


def test_runfile_dict_tuple_migration(tmp_path):
    """Ensure tuple keys in runfile_dict are migrated to strings."""

    pyro = example_SCENE.main(tmp_path)

    # PyroScan canonicalises parameter_dict values to pyrokinetics simulation
    # units at construction, so runfile_dict keys must use the post-conversion
    # magnitudes (format_single_run_name uses the raw magnitude string).
    ky_vals = np.array([0.1, 0.2]) / units.rhoref_unit
    ky_pyro_mag = ky_vals.convert_physical_units(pyro.norms).magnitude

    old_style = {(f"ky_{v}",): f"dir{i + 1}" for i, v in enumerate(ky_pyro_mag)}

    ps = PyroScan(
        pyro,
        parameter_dict={"ky": ky_vals},
        runfile_dict=old_style,
        base_directory=tmp_path,
    )

    ps.write(file_name="migrate.in", base_directory=tmp_path)

    loaded = PyroScan(pyro, pyroscan_json=tmp_path / "pyroscan.json")

    assert all(isinstance(k, str) for k in loaded.runfile_dict.keys())


def test_format_run_name():
    scan = PyroScan(Pyro(gk_code="GS2"), value_separator="|", parameter_separator="@")

    assert scan.format_single_run_name({"ky": 0.1, "nx": 55}) == "ky|0.10@nx|55.00"


def test_format_run_name_units():
    scan = PyroScan(Pyro(gk_code="GS2"), value_separator="|", parameter_separator="@")

    assert (
        scan.format_single_run_name(
            {"ky": 0.1 * units.rhoref_pyro**-1, "nx": 55 * units.dimensionless}
        )
        == "ky|0.10@nx|55.00"
    )


def test_create_single_run():
    scan = PyroScan(
        Pyro(gk_code="GS2"),
        base_directory="some_dir",
        value_separator="|",
        parameter_separator="@",
    )

    run_parameters = {"ky": 0.1, "nx": 55}
    name, new_run = scan.create_single_run(run_parameters)

    assert name == scan.format_single_run_name(run_parameters)
    assert new_run.file_name == "input.in"
    assert new_run.run_directory == Path(f"./some_dir/{name}").absolute()
    assert new_run.run_parameters == run_parameters


def test_norms_persisted_across_write_load(tmp_path):
    """
    Normalisations from the original pyro must survive a write/reload cycle.

    This covers the case where a user loads a GENE file (which carries
    normalisations), converts to TGLF, creates a PyroScan, writes it, and
    later reloads it with ``load_base_pyro=True``.  The TGLF input file
    alone cannot store normalisations, so they must be saved separately
    in ``pyroscan_norms.json``.
    """
    # Load a GENE input file that carries normalisations
    gene_file = template_dir / "input_wunits.gene"
    pyro = Pyro(gk_file=gene_file, gk_code="GENE")

    # Convert to TGLF — norms are retained in-memory but TGLF can't store them
    pyro.convert_gk_code("TGLF")

    # Grab the original reference values *before* writing
    orig_refs = pyro.get_reference_values()

    # Create a PyroScan over gamma_exb (the real-world use case)
    gamma_exb_values = np.array([0.0, 0.1, 0.2, 0.3]) * pyro.numerics.gamma_exb.units
    ps = PyroScan(
        pyro,
        parameter_dict={"gamma_exb": gamma_exb_values},
        base_directory=tmp_path / "scan_norms",
    )
    ps.add_parameter_key("gamma_exb", "numerics", ["gamma_exb"])
    ps.write(file_name="input.tglf", base_directory=tmp_path / "scan_norms")

    # Reload the PyroScan from disk — no original pyro supplied
    loaded = PyroScan(
        pyroscan_json=tmp_path / "scan_norms" / "pyroscan.json",
        load_base_pyro=True,
    )

    # The reloaded base_pyro must carry the same reference values
    loaded_refs = loaded.base_pyro.get_reference_values()
    for key in orig_refs:
        if orig_refs[key] is None:
            continue
        assert np.isclose(
            orig_refs[key].magnitude, loaded_refs[key].magnitude, rtol=1e-5
        ), f"Reference value {key} differs: {orig_refs[key]} vs {loaded_refs[key]}"

    # After reload, the scan values must be applied correctly
    loaded.update_self_parameters()
    for i, (name, p) in enumerate(loaded.pyro_dict.items()):
        expected = gamma_exb_values[i].magnitude
        actual = p.numerics.gamma_exb.magnitude
        assert np.isclose(
            actual, expected, rtol=1e-5
        ), f"Run {name}: gamma_exb = {actual}, expected {expected}"


def test_norms_not_persisted_without_units(tmp_path):
    """
    When the original GENE file has no physical units, the pyroscan
    should still write and reload successfully — just without saving
    pyroscan_norms.json.
    """
    # Load a GENE input file without a &units section
    gene_file = template_dir / "input.gene"
    pyro = Pyro(gk_file=gene_file)

    # Convert to TGLF
    pyro.convert_gk_code("TGLF")

    # Create a PyroScan and write — should warn but not crash
    ps = PyroScan(
        pyro,
        parameter_dict={"ky": np.array([0.1, 0.2])},
        base_directory=tmp_path / "scan_no_norms",
    )
    with pytest.warns(UserWarning, match="Could not save normalisation"):
        ps.write(file_name="input.tglf", base_directory=tmp_path / "scan_no_norms")

    # No norms file should have been created
    norms_file = tmp_path / "scan_no_norms" / "pyroscan_norms.json"
    assert not norms_file.exists()

    # Reload should still work
    loaded = PyroScan(
        pyroscan_json=tmp_path / "scan_no_norms" / "pyroscan.json",
        load_base_pyro=True,
    )
    assert len(loaded.pyro_dict) == 2


def test_pyroscan_reload_unitless_parameter(tmp_path):
    """
    Reload a PyroScan whose parameter has no units (e.g. dimensionless kappa
    or integer ntheta). The reloaded scan must round-trip through pyroscan.json
    and load_gk_output must iterate over unitless values without crashing.
    """
    pyro = example_SCENE.main(tmp_path)

    parameter_dict = {
        "kappa": np.array([1.5, 1.6, 1.7]),
        "ntheta": np.array([16, 32]),
    }

    ps = PyroScan(pyro, parameter_dict=parameter_dict, base_directory=tmp_path)
    ps.add_parameter_key("ntheta", "numerics", ["ntheta"])
    ps.write(file_name="unitless.in", base_directory=tmp_path)

    loaded = PyroScan(ps.base_pyro, pyroscan_json=tmp_path / "pyroscan.json")

    assert set(loaded.parameter_dict) == set(parameter_dict)
    for key, expected in parameter_dict.items():
        actual = loaded.parameter_dict[key]
        assert not hasattr(actual, "units"), f"{key} unexpectedly gained units"
        np.testing.assert_allclose(np.asarray(actual), expected)

    # load_gk_output once iterated parameter_dict values assuming each had
    # .magnitude, which crashed for unitless parameters. Without real output
    # files a FileNotFoundError is acceptable here; only AttributeError on
    # .magnitude indicates the unit-handling bug.
    with pytest.raises(FileNotFoundError):
        loaded.load_gk_output()


def test_apply_func(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    parameter_dict = {"aln": [1.0, 2.0, 3.0]}

    pyro_scan = PyroScan(pyro, parameter_dict=parameter_dict)

    pyro_scan.add_parameter_key("aln", "local_species", ["electron", "inverse_ln"])

    def maintain_quasineutrality(pyro):
        for species in pyro.local_species.names:
            if species != "electron":
                pyro.local_species[species].inverse_ln = (
                    pyro.local_species.electron.inverse_ln
                )

    parameter_kwargs = {}
    pyro_scan.add_parameter_func("aln", maintain_quasineutrality, parameter_kwargs)

    # Add function to pyro
    pyro_scan.write(file_name="test_pyroscan_func.input", base_directory=tmp_path)

    for pyro in pyro_scan.pyro_dict.values():
        for species in pyro.local_species.names:
            assert (
                pyro.local_species.electron.inverse_ln
                == pyro.local_species[species].inverse_ln
            )


def test_parameter_func_not_applied_twice(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    parameter_dict = {"alt": [1.0]}

    pyro_scan = PyroScan(pyro, parameter_dict=parameter_dict)
    pyro_scan.add_parameter_key("alt", "local_species", ["electron", "inverse_lt"])

    def increment_electron(pyro):
        pyro.local_species.electron.inverse_lt *= 2.0

    pyro_scan.add_parameter_func("alt", increment_electron, {})

    pyro_scan.write(file_name="test.input", base_directory=tmp_path)

    # Now check value
    for pyro_obj in pyro_scan.pyro_dict.values():
        val = pyro_obj.local_species.electron.inverse_lt

        assert np.isclose(val.magnitude, 2.0)
