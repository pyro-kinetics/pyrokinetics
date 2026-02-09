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
    "json_dir,  nonlinear_tmp_path",
    [
        (
            "CGYRO_linear_scan",
            template_dir / "outputs" / "CGYRO_linear_scan" / "pyroscan_nonlinear.zip",
        ),
    ],
)
def test_pyroscan_read_linear(json_dir, nonlinear_tmp_path):
    json_path = nonlinear_tmp_path / json_dir
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


@pytest.mark.parametrize(
    " json_dir, zip_path",
    [
        (
            "TGLF_transport_scan",
            template_dir / "outputs" / "TGLF_transport_scan" / "pyroscan_nonlinear.zip",
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
        for left_value, right_value in zip(left.values(), right.values()):
            assert np.allclose(left_value, right_value)
    elif attr == "pyroscan_json":
        for json_key in left.keys():
            if json_key == "parameter_dict":
                assert left[json_key].keys() == right[json_key].keys()
                for left_value, right_value in zip(
                    left[json_key].values(), right[json_key].values()
                ):
                    assert np.allclose(left_value, right_value)
            else:
                assert json_key in right.keys()
                if isinstance(left[json_key], (str, list, type(None), dict, Path)):
                    assert np.all(left[json_key] == right[json_key])
                else:
                    assert np.allclose(left[json_key], right[json_key]), (
                        f"{left} != {right}"
                    )
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
            "ky": np.array([0.1, 0.2]) / units.rhoref_pyro
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

    old_style = {
        ("ky_0.1",): "dir1",
        ("ky_0.2",): "dir2",
    }

    ps = PyroScan(
        pyro,
        parameter_dict={"ky": np.array([0.1, 0.2])},
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


def test_apply_func(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    parameter_dict = {"aln": [1.0, 2.0, 3.0]}

    pyro_scan = PyroScan(pyro, parameter_dict=parameter_dict)

    pyro_scan.add_parameter_key("aln", "local_species", ["electron", "inverse_ln"])

    def maintain_quasineutrality(pyro):
        for species in pyro.local_species.names:
            if species != "electron":
                pyro.local_species[
                    species
                ].inverse_ln = pyro.local_species.electron.inverse_ln

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
