import copy

import f90nml
import netCDF4 as nc
import numpy as np
import pytest

import pyrokinetics as pk
from pyrokinetics.pyroscan import PyroScan


def test_enforce_beta_prime():
    """Test that enforcing a consistent beta_prime work correctly."""
    eq_file = pk.template_dir / "test.geqdsk"
    kinetics_file = pk.template_dir / "jetto.jsp"

    pyro = pk.Pyro(eq_file=eq_file, kinetics_file=kinetics_file)
    pyro.load_local(psi_n=0.5)
    pyro.gk_code = "CGYRO"

    assert pyro.local_geometry.B0.m == 1.0
    assert np.isclose(
        pyro.local_geometry.B0.to(pyro.norms.cgyro).m,
        (1.0 / pyro.local_geometry.bunit_over_b0.m),
    )
    assert pyro.gk_input.data["BETA_STAR_SCALE"] != 1.0

    pyro.enforce_consistent_beta_prime()
    pyro.update_gk_code()
    assert np.isclose(pyro.gk_input.data["BETA_STAR_SCALE"], 1.0)


@pytest.mark.parametrize("gk_code", ["GS2", "CGYRO", "GENE", "TGLF"])
def test_enforce_pvg(gk_code):
    """Test that domega_drho (PVG) is updated consistently with gamma_exb."""
    pyro = pk.Pyro(gk_file=pk.gk_templates[gk_code])

    gamma_exb_val = 0.1 * pyro.numerics.gamma_exb.units
    pyro.numerics.gamma_exb = gamma_exb_val

    pyro.enforce_consistent_pvg()

    q = pyro.local_geometry.q
    rho = pyro.local_geometry.rho
    expected = -(q / rho) * gamma_exb_val

    for name in pyro.local_species.names:
        actual = pyro.local_species[name].domega_drho.to(
            expected.units, pyro.norms.context
        )
        assert np.isclose(actual.m, expected.m), (
            f"{gk_code}/{name}: expected domega_drho={expected:.4f}, got {actual:.4f}"
        )


def test_enforce_pvg_raises_without_geometry():
    """Test that enforce_consistent_pvg raises when geometry is missing."""
    pyro = pk.Pyro(gk_code="CGYRO")
    pyro.local_geometry = None
    with pytest.raises(ValueError, match="enforce_consistent_pvg"):
        pyro.enforce_consistent_pvg()


def test_enforce_pvg_in_pyroscan(tmp_path):
    """Test enforce_consistent_pvg used as a parameter_func in a gamma_exb PyroScan."""
    pyro = pk.Pyro(gk_file=pk.gk_templates["CGYRO"])

    gamma_exb_values = np.array([0.0, 0.05, 0.10]) * pyro.numerics.gamma_exb.units
    param_dict = {"gamma_exb": gamma_exb_values}

    pyro_scan = PyroScan(pyro, param_dict, base_directory=tmp_path)
    pyro_scan.add_parameter_func(
        "gamma_exb", pk.Pyro.enforce_consistent_pvg, {}
    )
    pyro_scan.write(file_name="input.cgyro", base_directory=tmp_path)

    for _, run_pyro in zip(
        pyro_scan.outer_product(), pyro_scan.pyro_dict.values()
    ):
        g_exb = run_pyro.numerics.gamma_exb
        q = run_pyro.local_geometry.q
        rho = run_pyro.local_geometry.rho
        expected = -(q / rho) * g_exb

        for name in run_pyro.local_species.names:
            actual = run_pyro.local_species[name].domega_drho.to(
                expected.units, run_pyro.norms.context
            )
            assert np.isclose(actual.m, expected.m), (
                f"gamma_exb={g_exb:.3f}, species={name}: "
                f"expected domega_drho={expected:.4f}, got {actual:.4f}"
            )


@pytest.mark.parametrize("exbrate", [0.0, 0.05, 0.1, 0.2, -0.1])
def test_enforce_pvg_consistent_with_gene_exb_scan(tmp_path, exbrate):
    """Verify enforce_consistent_pvg produces the same PVG as a GENE ExB scan.

    In GENE, PVG is not a separate input parameter.  When ``pfsrate`` equals
    ``exbrate``, GENE internally computes::

        domega_drho = -(q / rho) * exbrate

    and passes that through to TGLF as ``vpar_shear_1``.
    ``enforce_consistent_pvg`` uses the identical formula, so both paths must
    produce the same ``vpar_shear_1`` in a converted TGLF input file.
    """
    # ------------------------------------------------------------------
    # Path 1: write a GENE file with exbrate = pfsrate, load it back,
    #         then convert to TGLF.  This is what GENE itself would use.
    # ------------------------------------------------------------------
    pyro_base = pk.Pyro(gk_file=pk.gk_templates["GENE"])
    gene_data = copy.deepcopy(pyro_base.gk_input.data)
    gene_data["external_contr"] = f90nml.Namelist(
        {"exbrate": exbrate, "pfsrate": exbrate, "omega0_tor": 0.0}
    )
    gene_path = tmp_path / "test_gene_exb.gene"
    f90nml.write(gene_data, str(gene_path))

    pyro_gene = pk.Pyro(gk_file=gene_path, gk_code="GENE")
    pyro_gene.gk_code = "TGLF"
    vpar_shear_gene = pyro_gene.gk_input.data.get("vpar_shear_1", 0.0)

    # ------------------------------------------------------------------
    # Path 2: load the same base GENE file, set gamma_exb, call
    #         enforce_consistent_pvg, then convert to TGLF.
    # ------------------------------------------------------------------
    pyro_enforce = pk.Pyro(gk_file=pk.gk_templates["GENE"])
    pyro_enforce.numerics.gamma_exb = exbrate * pyro_enforce.numerics.gamma_exb.units
    pyro_enforce.enforce_consistent_pvg()
    pyro_enforce.gk_code = "TGLF"
    vpar_shear_enforce = pyro_enforce.gk_input.data.get("vpar_shear_1", 0.0)

    assert np.isclose(vpar_shear_enforce, vpar_shear_gene), (
        f"exbrate={exbrate}: enforce_consistent_pvg gives vpar_shear_1="
        f"{vpar_shear_enforce:.6f} but GENE ExB scan gives {vpar_shear_gene:.6f}"
    )


def test_normalise():
    """Test that a local geometry can be renormalised with simulation units."""
    pyro = pk.Pyro(gk_file=pk.gk_templates["GS2"])
    geometry = pyro.local_geometry
    norms = pyro.norms

    Rmaj = geometry.Rmaj
    a_minor = geometry.a_minor
    assert Rmaj.units == norms.units.lref_minor_radius
    assert a_minor.units == norms.units.lref_minor_radius

    # Convert to a different units standard
    # LocalGeometry.normalise() is an in-place operation
    geometry.normalise(norms.gene)
    assert np.isfinite(geometry.Rmaj.magnitude)
    assert np.isfinite(geometry.a_minor.magnitude)
    assert geometry.Rmaj.units == norms.units.lref_major_radius
    assert geometry.a_minor.units == norms.units.lref_major_radius
    assert (geometry.Rmaj / geometry.a_minor) == (Rmaj / a_minor)


@pytest.fixture(scope="module")
def setup_area_volume():

    transp_file = pk.template_dir / "transp.cdf"
    data = nc.Dataset(transp_file)

    transp_dvol = data["DVOL"][-1, :] * 1e-6
    transp_volume = np.cumsum(transp_dvol)

    transp_darea = data["DAREA"][-1, :] * 1e-4
    transp_area = np.cumsum(transp_darea)

    transp_surface = data["SURF"][-1, :] * 1e-4

    transp_psi_n = data["PLFLX"][-1, :] / data["PLFLX"][-1, -1]

    transp_r = data["RMNMP"][-1, :] * 1e-2

    transp_dVdr = np.gradient(transp_volume, transp_r)
    transp_dAdr = np.gradient(transp_area, transp_r)
    transp_dSdr = np.gradient(transp_surface, transp_r)

    # Load up pyro object
    pyro = pk.Pyro(
        eq_file=transp_file,
        eq_type="TRANSP",
        eq_kwargs={"neighbors": 64},
        kinetics_file=transp_file,
        kinetics_type="TRANSP",
    )

    # Rename the ion species in the original pyro object

    return {
        "pyro": pyro,
        "transp_psi_n": transp_psi_n,
        "transp_area": transp_area,
        "transp_surface": transp_surface,
        "transp_volume": transp_volume,
        "transp_darea": transp_dAdr,
        "transp_dsurface": transp_dSdr,
        "transp_dvolume": transp_dVdr,
    }


@pytest.mark.parametrize("psi_n", [0.25, 0.5, 0.75, 0.95])
def test_flux_surface_area_volume(setup_area_volume, psi_n):

    pyro = setup_area_volume["pyro"]
    transp_psi_n = setup_area_volume["transp_psi_n"]
    transp_area = setup_area_volume["transp_area"]
    transp_surface = setup_area_volume["transp_surface"]
    transp_volume = setup_area_volume["transp_volume"]
    transp_darea = setup_area_volume["transp_darea"]
    transp_dsurface = setup_area_volume["transp_dsurface"]
    transp_dvolume = setup_area_volume["transp_dvolume"]

    psi_n_index = np.argmin(np.abs(transp_psi_n - psi_n))

    pyro.load_local_geometry(
        psi_n=transp_psi_n[psi_n_index], local_geometry="MXH", n_moments=7
    )

    area, surface, volume = pyro.local_geometry.get_flux_surface_area_volume()

    surface = surface.to("meter**2").m
    area = area.to("meter**2").m
    volume = volume.to("meter**3").m

    assert np.isclose(area, transp_area[psi_n_index], rtol=1e-2)
    assert np.isclose(surface, transp_surface[psi_n_index], rtol=1e-2)
    assert np.isclose(volume, transp_volume[psi_n_index], rtol=1e-2)

    darea, dsurface, dvolume = (
        pyro.local_geometry.get_flux_surface_area_volume_derivatives()
    )

    dsurface = dsurface.to("meter").m
    darea = darea.to("meter").m
    dvolume = dvolume.to("meter**2").m

    assert np.isclose(darea, transp_darea[psi_n_index], rtol=1e-2)
    assert np.isclose(dsurface, transp_dsurface[psi_n_index], rtol=1e-2)
    assert np.isclose(dvolume, transp_dvolume[psi_n_index], rtol=1e-2)
