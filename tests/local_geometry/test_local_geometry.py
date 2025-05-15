import numpy as np
import netCDF4 as nc
import pyrokinetics as pk

import pytest


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
