import netCDF4 as nc
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import Redl2021, Sauter1999
from pyrokinetics.units import ureg as units


def test_bootstrap_current():

    # Equilibrium and Kinetics data file
    scene_cdf = template_dir / "scene.cdf"
    scene_eqdsk = template_dir / "step.geqdsk"

    downsize = 5
    scene_data = nc.Dataset(scene_cdf)
    scene_psin = scene_data["rho_psi"][::-downsize] ** 2

    # SCENE gives <Jbs. B> / <B^2>
    scene_jdotb_b2 = (
        scene_data["Jbs.B"][::-downsize].data
        * units.ampere
        / units.tesla
        / units.meter**2
    )
    scene_zeff = scene_data["zeff"][::-downsize].data * units.elementary_charge

    redl_jdotb_b2 = scene_jdotb_b2 * 0.0
    sauter_jdotb_b2 = scene_jdotb_b2 * 0.0

    # Load up pyro object
    pyro = Pyro(
        eq_file=scene_eqdsk,
        eq_type="GEQDSK",
        kinetics_file=scene_cdf,
        kinetics_type="SCENE",
    )

    for i, psi_n in enumerate(scene_psin[1:]):
        try:
            pyro.load_local(psi_n=psi_n, local_geometry="MXH")
        except Exception:
            continue

        pyro.local_species.zeff = scene_zeff[i + 1]

        redl = Redl2021(pyro)
        redl_jdotb_b2[i + 1] = (redl.JbsdotB / redl.B2_fsa).to("ampere / tesla / m**2")

        sauter = Sauter1999(pyro)
        sauter_jdotb_b2[i + 1] = (sauter.JbsdotB / sauter.B2_fsa).to(
            "ampere / tesla / m**2"
        )

    assert_allclose(redl_jdotb_b2.m, scene_jdotb_b2.m, rtol=6e-2)
    assert_allclose(sauter_jdotb_b2.m, scene_jdotb_b2.m, rtol=15e-2)
