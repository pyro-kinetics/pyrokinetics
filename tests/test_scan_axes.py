"""
Scan axes: a joint axis (parameters varied together, as a Latin hypercube)
crossed with grid axes, e.g. every hypercube point run at the same ky list.
"""

import shutil

import numpy as np
import pytest

from pyrokinetics import Pyro, PyroHypercube, PyroScan, template_dir
from pyrokinetics.units import ureg as units

GS2_TEMPLATE_DIR = template_dir / "outputs" / "GS2_linear"
TEMP_GRADIENTS = [1.0, 2.0, 3.0]
KY = [0.1, 0.5]


def _cube(tmp_path, **kwargs):
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    cube = PyroScan(
        pyro,
        cube={
            "electron_temp_gradient": np.array(TEMP_GRADIENTS) / units.lref_minor_radius
        },
        grid={"ky": np.array(KY) / units.rhoref_pyro},
        value_fmt="g",
        base_directory=tmp_path,
        file_name="gs2.in",
        **kwargs,
    )
    cube.write()
    for run_directory in cube.run_directories:
        shutil.copy(GS2_TEMPLATE_DIR / "gs2.out.nc", run_directory / "gs2.out.nc")
    return cube


def test_cube_crossed_with_grid_runs_every_point_at_every_value(tmp_path):
    cube = _cube(tmp_path)
    assert cube.n_samples == 3
    assert len(cube.pyro_dict) == 6
    assert list(cube.pyro_dict)[:3] == [
        "sample_0000/ky_0.1",
        "sample_0000/ky_0.5",
        "sample_0001/ky_0.1",
    ]
    runs = list(cube.outer_product())
    assert [r["ky"].m for r in runs] == KY * 3
    assert [r["electron_temp_gradient"].m for r in runs] == list(
        np.repeat(TEMP_GRADIENTS, 2)
    )


def test_crossed_output_has_sample_and_grid_dims(tmp_path):
    cube = _cube(tmp_path)
    cube.load_gk_output(load_fields=False)
    data = cube.gk_output.data
    assert data.growth_rate.dims == ("sample", "ky")
    assert data.electron_temp_gradient.dims == ("sample",)
    assert np.allclose(data.ky, KY)
    assert data.run_name.sel(sample=2).values[1] == "sample_0002/ky_0.5"
    assert list(data.sample_name.values) == [
        "sample_0000",
        "sample_0001",
        "sample_0002",
    ]


def test_sample_pyro_takes_one_index_per_axis(tmp_path):
    cube = _cube(tmp_path)
    cube.load_gk_output(load_fields=False)
    pyro = cube.sample_pyro(2, ky=1)
    written = Pyro(gk_file=tmp_path / "sample_0002" / "ky_0.5" / "gs2.in")
    assert np.isclose(
        pyro.numerics.ky.m, written.numerics.ky.to(pyro.norms.pyrokinetics).m
    )
    assert np.isclose(pyro.numerics.ky.m, 0.5)
    assert np.isclose(pyro.local_species.electron.inverse_lt.m, 3.0)
    assert pyro.gk_output.growth_rate.dims == ()
    with pytest.raises(ValueError, match="missing \\['ky'\\]"):
        cube.sample_pyro(2)


def test_crossed_cube_reloads_from_json(tmp_path):
    cube = _cube(tmp_path)
    cube.load_gk_output(load_fields=False)
    reloaded = PyroScan(pyroscan_json=tmp_path / "pyroscan.json", load_base_pyro=True)
    assert list(reloaded.pyro_dict) == list(cube.pyro_dict)
    assert reloaded.n_samples == 3
    reloaded.load_gk_output(load_fields=False)
    assert np.array_equal(
        reloaded.gk_output.data.growth_rate.values,
        cube.gk_output.data.growth_rate.values,
    )
    # The same leaf through the old class name
    hypercube = PyroHypercube(
        pyroscan_json=tmp_path / "pyroscan.json", load_base_pyro=True
    )
    assert list(hypercube.pyro_dict) == list(cube.pyro_dict)


def test_pyrohypercube_grid_argument_is_the_same_scan(tmp_path):
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    hypercube = PyroHypercube(
        pyro,
        parameter_dict={
            "electron_temp_gradient": np.array(TEMP_GRADIENTS) / units.lref_minor_radius
        },
        grid={"ky": np.array(KY) / units.rhoref_pyro},
        value_fmt="g",
    )
    assert list(hypercube.pyro_dict) == list(_cube(tmp_path).pyro_dict)


def test_run_names_use_grid_values_as_typed():
    """GS2 aky 0.1 is stored as pyro ky 0.1/sqrt(2), but the directory says 0.1."""
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    ky = np.array(KY) * pyro.norms.gs2.rhoref**-1
    cube = PyroScan(
        pyro,
        cube={"electron_temp_gradient": np.array([1.0]) / units.lref_minor_radius},
        grid={"ky": ky},
        value_fmt="g",
    )
    assert list(cube.pyro_dict) == ["sample_0000/ky_0.1", "sample_0000/ky_0.5"]
    assert np.allclose(cube.parameter_dict["ky"].m, np.array(KY) / np.sqrt(2))


def test_one_parameter_joint_axes_match_a_grid():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    parameter_dict = {
        "electron_temp_gradient": np.array(TEMP_GRADIENTS) / units.lref_minor_radius,
        "ky": np.array(KY) / units.rhoref_pyro,
    }
    grid = PyroScan(pyro, parameter_dict=dict(parameter_dict))
    joint = PyroScan(
        pyro,
        parameter_dict=dict(parameter_dict),
        joint_axes={"temp": ["electron_temp_gradient"]},
    )
    for a, b in zip(grid.outer_product(), joint.outer_product()):
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k].m, b[k].m) for k in a)


def test_joint_axis_lengths_must_agree():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    with pytest.raises(ValueError, match="one value per point"):
        PyroScan(
            pyro,
            cube={
                "ky": np.array([0.1, 0.2]) / units.rhoref_pyro,
                "electron_temp_gradient": np.array([1.0]) / units.lref_minor_radius,
            },
        )


def test_repeated_run_directories_raise():
    pyro = Pyro(gk_file=GS2_TEMPLATE_DIR / "gs2.in")
    with pytest.raises(ValueError, match="finer value_fmt"):
        PyroScan(
            pyro,
            cube={"electron_temp_gradient": np.array([1.0]) / units.lref_minor_radius},
            grid={"ky": np.array([0.101, 0.102]) / units.rhoref_pyro},
            value_fmt=".2f",
        )
