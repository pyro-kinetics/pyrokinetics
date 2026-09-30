import shutil

import numpy as np
import pytest
import xarray as xr

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.field_line import FieldLine
from pyrokinetics.pyroscan import PyroScan


def _em_gftm_run(dest, mix=0.0):
    """
    Copy of the GFTM_linear template with a synthetic apar added to its
    wavefunction: mode 0 has tearing parity (even), mode 1 ballooning (odd).
    ``mix`` adds the other parity, so the tearing parameter depends on geometry.
    """
    shutil.copytree(template_dir / "outputs" / "GFTM_linear", dest, dirs_exist_ok=True)
    wavefunction = dest / "out.gftm.wavefunction"
    lines = wavefunction.read_text().splitlines()
    nmode, _, ntheta = (int(x) for x in lines[0].split())
    theta = np.loadtxt(lines[2:])[:, 0]

    envelope = np.exp(-((theta / 3) ** 2))
    zero = np.zeros_like(theta)
    # per mode: phi (re, im), apar (re, im)
    columns = [theta]
    even, odd = envelope, theta / 3 * envelope
    for apar in (even + mix * odd, odd + mix * even)[:nmode]:
        columns += [envelope, zero, apar, zero]
    np.savetxt(
        wavefunction,
        np.column_stack(columns),
        header=f"{nmode} 2 {ntheta}\ntheta RE(phi) IM(phi) RE(apar) IM(apar)",
        comments="",
    )
    return dest


@pytest.mark.parametrize(
    "template, file_name",
    [("TGLF_linear", "input.tglf"), ("GFTM_linear", "input.gftm")],
)
def test_linear_fields_are_theta_resolved(template, file_name):
    pyro = Pyro(gk_file=template_dir / "outputs" / template / file_name)
    pyro.load_gk_output()
    data = pyro.gk_output.data

    assert data["phi"].dims == ("theta", "kx", "ky", "mode")
    # eigenfunctions stays a (field, theta, mode) alias of the fields
    assert data["eigenfunctions"].dims == ("field", "theta", "mode")
    np.testing.assert_allclose(
        data["phi"].isel(kx=0, ky=0).pint.dequantify().values,
        data["eigenfunctions"].sel(field="phi").pint.dequantify().values,
        atol=1e-12,
    )


def test_gftm_tearing_parameter_per_mode(tmp_path):
    run = _em_gftm_run(tmp_path / "run")
    pyro = Pyro(gk_file=run / "input.gftm")
    pyro.load_gk_output()
    assert pyro.gk_output.data["apar"].dims == ("theta", "kx", "ky", "mode")

    tearing = FieldLine(pyro).compute_linear_tearing_parameter()

    assert tearing.dims == ("kx", "ky", "mode")
    tearing = tearing.isel(kx=0, ky=0)
    assert tearing.sel(mode=0) > 0.99
    assert tearing.sel(mode=1) < 0.01

    even = FieldLine(pyro).compute_linear_parity()
    assert even.dims == ("kx", "ky", "mode")
    np.testing.assert_allclose(even.isel(kx=0, ky=0).values, [1.0, 0.0], atol=1e-12)


def test_stella_tearing_parameter_has_no_mode():
    pyro = Pyro(gk_file=template_dir / "outputs" / "STELLA_linear" / "stella.in")
    pyro.load_gk_output()

    tearing = FieldLine(pyro).compute_linear_tearing_parameter()

    assert set(tearing.dims) == {"kx", "ky"}
    assert np.all((tearing >= 0) & (tearing <= 1))

    even = FieldLine(pyro).compute_linear_parity()
    assert set(even.dims) == {"kx", "ky"}
    assert np.all((even >= 0) & (even <= 1))


def test_pyroscan_tearing_parameter_matches_per_run(tmp_path):
    source = _em_gftm_run(tmp_path / "source", mix=0.5)
    pyro = Pyro(gk_file=source / "input.gftm")
    kappa = float(pyro.local_geometry.kappa)
    scan = PyroScan(
        pyro,
        parameter_dict={"kappa": [kappa, 2 * kappa]},
        base_directory=tmp_path / "scan",
        file_name="input.gftm",
    )
    scan.write()
    for name in scan.pyro_dict:
        for output in source.glob("out.gftm.*"):
            shutil.copy(output, tmp_path / "scan" / name)

    # As in use: reloaded, so its pyros hold the base geometry, not the scan's
    scan = PyroScan(
        pyroscan_json=tmp_path / "scan" / "pyroscan.json", load_base_pyro=True
    )
    scan.load_gk_output(load_tearing_parameter=True)
    tearing = scan.gk_output.data["tearing_parameter"]
    even = scan.gk_output.data["apar_even_fraction"]
    assert tearing.dims == even.dims == ("kappa", "ky", "mode")

    for i, name in enumerate(scan.pyro_dict):
        run = Pyro(gk_file=tmp_path / "scan" / name / "input.gftm")
        run.load_gk_output()
        field_line = FieldLine(run)
        # ky is ky/bunit_over_b0, which kappa changes, so each run has its own ky
        for stored, expected in (
            (tearing, field_line.compute_linear_tearing_parameter()),
            (even, field_line.compute_linear_parity()),
        ):
            actual = stored.isel(kappa=i).dropna("ky", how="all")
            np.testing.assert_allclose(actual.data.m, expected.isel(kx=0).values)

    # Each run uses its own geometry
    per_run = [tearing.isel(kappa=i).dropna("ky", how="all").data.m for i in (0, 1)]
    assert not np.allclose(*per_run)


def test_parity_of_synthetic_fields():
    pyro = Pyro(gk_file=template_dir / "outputs" / "STELLA_linear" / "stella.in")
    pyro.load_gk_output()
    apar = pyro.gk_output.data["apar"]
    theta = apar.theta
    envelope = np.exp(-((theta / 3) ** 2)) * (1 + 0.5j * np.cos(theta))

    for shape, expected in ((envelope, 1.0), (np.sin(theta) * envelope, 0.0)):
        pyro.gk_output.data["apar"] = apar.copy(
            data=(shape * xr.ones_like(apar.pint.dequantify())).values * apar.data.u
        )
        even = FieldLine(pyro).compute_linear_parity()
        np.testing.assert_allclose(even.values, expected, atol=1e-12)
