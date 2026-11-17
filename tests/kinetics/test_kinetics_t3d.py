import netCDF4 as nc
import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.constants import deuterium_mass, electron_mass, tritium_mass
from pyrokinetics.equilibrium import read_equilibrium
from pyrokinetics.kinetics import read_kinetics, supported_kinetics_types


@pytest.fixture
def equilibrium():
    return read_equilibrium(template_dir / "test.geqdsk")


@pytest.fixture
def mock_t3d_log_file(tmp_path):
    rho = np.linspace(0.02, 0.96, 25)
    midpoints = 0.5 * (rho[:-1] + rho[1:])
    ne_t0 = 1.6 * (1.0 - 0.35 * rho**2)
    ne_t1 = 2.0 * (1.0 - 0.30 * rho**2)
    Te_t0 = 10.0 * (1.0 - 0.50 * rho**2)
    Te_t1 = 12.5 * (1.0 - 0.45 * rho**2)
    TD_t0 = 9.0 * (1.0 - 0.50 * rho**2)
    TD_t1 = 11.0 * (1.0 - 0.45 * rho**2)

    log_dict = {
        "t": [0.0, 1.0],
        "norms": {"t_ref": 0.5},
        "grid": {"N_radial": 25, "rho": rho, "midpoints": midpoints},
        "species_tags": ["e", "D", "T"],
        "n_e": [ne_t0, ne_t1],
        "n_D": [0.5 * ne_t0, 0.5 * ne_t1],
        "n_T": [0.5 * ne_t0, 0.5 * ne_t1],
        "T_e": [Te_t0, Te_t1],
        "T_D": [TD_t0, TD_t1],
        "T_T": [TD_t0, TD_t1],
    }
    path = tmp_path / "input_step.log.npy"
    np.save(path, log_dict, allow_pickle=True)
    return path


@pytest.fixture
def mock_t3d_nc_file(tmp_path):
    rho = np.linspace(0.02, 0.96, 20)
    path = tmp_path / "t3d_out.nc"
    with nc.Dataset(path, "w") as ds:
        ds.trinity_infile = "input.in"
        g_grid = ds.createGroup("grid")
        g_grid.createDimension("rho", len(rho))
        v_rho = g_grid.createVariable("rho", "f8", ("rho",))
        v_rho[:] = rho

        g_time = ds.createGroup("time")
        g_time.createDimension("t", 2)
        v_t = g_time.createVariable("t", "f8", ("t",))
        v_t[:] = [0.0, 2.0]

        g_sp = ds.createGroup("species")
        g_sp.createDimension("t", 2)
        g_sp.createDimension("rho", len(rho))
        for tag, z_val, n_scale, t_scale in (
            ("e", -1.0, 1.8, 12.0),
            ("D", 1.0, 1.8, 10.5),
        ):
            vn = g_sp.createVariable(f"n_{tag}", "f8", ("t", "rho"))
            vn.Z = z_val
            vn[0, :] = 0.8 * n_scale * (1.0 - 0.3 * rho**2)
            vn[1, :] = n_scale * (1.0 - 0.3 * rho**2)

            vt = g_sp.createVariable(f"T_{tag}", "f8", ("t", "rho"))
            vt[0, :] = 0.8 * t_scale * (1.0 - 0.4 * rho**2)
            vt[1, :] = t_scale * (1.0 - 0.4 * rho**2)
    return path


def test_t3d_registered():
    assert "T3D" in supported_kinetics_types()


def test_t3d_requires_equilibrium(mock_t3d_log_file):
    with pytest.raises(ValueError, match="Please load an Equilibrium."):
        read_kinetics(mock_t3d_log_file, "T3D")


@pytest.mark.parametrize("kinetics_type", ["T3D", None])
def test_read_t3d_log_npy(mock_t3d_log_file, equilibrium, kinetics_type):
    kin = read_kinetics(mock_t3d_log_file, kinetics_type, eq=equilibrium)
    assert kin.kinetics_type == "T3D"
    assert sorted(kin.species_names) == ["deuterium", "electron", "tritium"]

    elec = kin.species_data["electron"]
    deut = kin.species_data["deuterium"]
    trit = kin.species_data["tritium"]

    assert elec.mass == electron_mass
    assert deut.mass == deuterium_mass
    assert trit.mass == tritium_mass
    assert np.isclose(elec.get_charge(0.5).m, -1.0)
    assert np.isclose(deut.get_charge(0.5).m, 1.0)

    # Check analytic parabolic profile atpsi_n=0.5: n_e(rho) = 2.0e20 * (1 - 0.3 * rho^2),
    # so -1/n dn/drho = (0.6 * rho) / (1 - 0.3 * rho^2)
    rho_mid = float(elec.get_rho(0.5).m)
    expected_ne = 2.0e20 * (1.0 - 0.30 * rho_mid**2)
    expected_omn = (0.60 * rho_mid) / (1.0 - 0.30 * rho_mid**2)
    expected_Te = 12.5e3 * (1.0 - 0.45 * rho_mid**2)
    expected_omt = (0.90 * rho_mid) / (1.0 - 0.45 * rho_mid**2)

    assert np.isclose(elec.get_dens(0.5).to("meter**-3").m, expected_ne, rtol=2e-3)
    assert np.isclose(elec.get_temp(0.5).to("eV").m, expected_Te, rtol=2e-3)
    assert np.isclose(elec.get_norm_dens_gradient(0.5).m, expected_omn, rtol=1e-2)
    assert np.isclose(elec.get_norm_temp_gradient(0.5).m, expected_omt, rtol=1e-2)


def test_read_t3d_time_selection(mock_t3d_log_file, equilibrium):
    kin_t0 = read_kinetics(mock_t3d_log_file, "T3D", eq=equilibrium, time_index=0)
    kin_time = read_kinetics(mock_t3d_log_file, "T3D", eq=equilibrium, time=0.05)
    assert np.isclose(
        kin_t0.species_data["electron"].get_temp(0.4).m,
        kin_time.species_data["electron"].get_temp(0.4).m,
    )
    with pytest.raises(ValueError, match="Cannot set both"):
        read_kinetics(mock_t3d_log_file, "T3D", eq=equilibrium, time_index=0, time=0.1)


@pytest.mark.parametrize("kinetics_type", ["T3D", None])
def test_read_t3d_netcdf(mock_t3d_nc_file, equilibrium, kinetics_type):
    kin = read_kinetics(mock_t3d_nc_file, kinetics_type, eq=equilibrium)
    assert kin.kinetics_type == "T3D"
    assert sorted(kin.species_names) == ["deuterium", "electron"]
    rho_mid = float(kin.species_data["electron"].get_rho(0.5).m)
    assert np.isclose(
        kin.species_data["electron"].get_dens(0.5).to("meter**-3").m,
        1.8e20 * (1.0 - 0.3 * rho_mid**2),
        rtol=2e-3,
    )


def test_pyro_t3d_to_nonlinear_gene(mock_t3d_log_file, tmp_path):
    pyro = Pyro(
        eq_file=template_dir / "test.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=mock_t3d_log_file,
        kinetics_type="T3D",
    )
    pyro.load_local(psi_n=0.5, local_geometry="Miller")
    pyro.gk_code = "GENE"
    norm = pyro.norms.pyrokinetics
    pyro.numerics.apar = True
    pyro.numerics.bpar = True
    pyro.numerics.nonlinear = True
    pyro.numerics.nkx = 64
    pyro.numerics.nky = 16
    pyro.numerics.ky = 0.05 / norm.rhoref
    pyro.numerics.gamma_exb = 0.02 * norm.vref / norm.lref

    out_gene = tmp_path / "parameters_0001"
    pyro.write_gk_file(file_name=out_gene)
    assert out_gene.exists()

    reloaded = Pyro(gk_file=out_gene, gk_code="GENE")
    assert reloaded.numerics.nonlinear is True
    assert reloaded.numerics.nkx == 64
    assert reloaded.numerics.nky == 16
    assert reloaded.numerics.bpar is True
