import numpy as np
from numpy.testing import assert_allclose
import netCDF4 as nc

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics


def test_gs2_geometry():
    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")

    diag = Diagnostics(pyro)

    geometry_terms = diag.gs2_geometry_terms(ntheta_multiplier=1)

    gs2_data = nc.Dataset(template_dir / "outputs/GS2_linear/gs2.out.nc")

    pyro_theta = geometry_terms["theta"]

    gs2_theta = gs2_data["theta"][:].data

    ignore_keys = ["dpdrho"]

    for key in geometry_terms.keys():
        if key not in ignore_keys:
            # GS2 bug in aplot
            if key == "aplot":
                gs2_aplot = gs2_data[key][:].data
                gs2_theta = gs2_data["theta"][:].data
                gs2_dpsidr = 1.0 / gs2_data["drhodpsi"][:].data
                gs2_aplot = np.where(
                    abs(gs2_theta) > np.pi, gs2_aplot / gs2_dpsidr, gs2_aplot
                )

                gs2_geo_term = np.interp(pyro_theta, gs2_theta, gs2_aplot)
            else:
                gs2_geo_term = np.interp(pyro_theta, gs2_theta, gs2_data[key][:].data)

            assert_allclose(gs2_geo_term, geometry_terms[key], rtol=3e-2, atol=1e-8)
