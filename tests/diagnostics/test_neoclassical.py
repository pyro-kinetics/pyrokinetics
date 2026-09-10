import netCDF4 as nc
import numpy as np
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import (
    Redl2021,
    Sauter1999,
    integrate_toroidal_current,
)
from pyrokinetics.units import ureg as units


def test_bootstrap_current():

    # Equilibrium and Kinetics data file
    scene_cdf = template_dir / "scene.cdf"
    scene_eqdsk = template_dir / "step.geqdsk"

    scene_data = nc.Dataset(scene_cdf)
    indices = slice(10, -10, 2)
    scene_psin = scene_data["rho_psi"][indices] ** 2

    # SCENE gives <Jbs. B> / <B^2>
    scene_jbsdotb_b2 = (
        scene_data["Jbs.B"][indices].data * units.ampere / units.tesla / units.meter**2
    )
    scene_zeff = scene_data["zeff"][indices].data * units.elementary_charge

    scene_jtotdotb_b2 = (
        scene_data["Jtot.B"][indices].data * units.ampere / units.tesla / units.meter**2
    )

    redl_jbsdotb_b2 = scene_jbsdotb_b2 * 0.0
    sauter_jbsdotb_b2 = scene_jbsdotb_b2 * 0.0
    redl_jtotdotb_b2 = scene_jtotdotb_b2 * 0.0
    sauter_jtotdotb_b2 = scene_jtotdotb_b2 * 0.0

    # Load up pyro object
    pyro = Pyro(
        eq_file=scene_eqdsk,
        eq_type="GEQDSK",
        kinetics_file=scene_cdf,
        kinetics_type="SCENE",
    )

    for i, psi_n in enumerate(scene_psin):
        try:
            pyro.load_local(psi_n=psi_n, local_geometry="MXH")
        except Exception:
            continue

        pyro.local_species.zeff = scene_zeff[i]

        redl = Redl2021(pyro)
        redl_jbsdotb_b2[i] = (redl.JbsdotB / redl.B2_fsa).to("ampere / tesla / m**2")
        redl_jtotdotb_b2[i] = (redl.JdotB / redl.B2_fsa).to("ampere / tesla / m**2")

        sauter = Sauter1999(pyro)
        sauter_jbsdotb_b2[i] = (sauter.JbsdotB / sauter.B2_fsa).to(
            "ampere / tesla / m**2"
        )
        sauter_jtotdotb_b2[i] = (sauter.JdotB / sauter.B2_fsa).to(
            "ampere / tesla / m**2"
        )

    assert_allclose(redl_jbsdotb_b2.m, scene_jbsdotb_b2.m, rtol=6e-2)
    assert_allclose(sauter_jbsdotb_b2.m, scene_jbsdotb_b2.m, rtol=15e-2)
    assert_allclose(redl_jtotdotb_b2.m, scene_jtotdotb_b2.m, rtol=1e-2)
    assert_allclose(sauter_jtotdotb_b2.m, scene_jtotdotb_b2.m, rtol=1e-2)


def test_toroidal_current():
    """
    Compare the toroidal current density and its decomposition against SCENE.

    SCENE's ``J_tor`` is the flux-surface averaged toroidal current density
    ``<J_phi>``, and is built from the parallel (bootstrap + external) currents
    only, i.e. it excludes the Pfirsch-Schlueter and diamagnetic contributions.
    """

    scene_cdf = template_dir / "scene.cdf"
    scene_eqdsk = template_dir / "step.geqdsk"

    scene_data = nc.Dataset(scene_cdf)
    indices = slice(10, -10, 2)
    scene_psin = scene_data["rho_psi"][indices] ** 2

    j_units = units.ampere / units.meter**2
    scene_jtor = scene_data["J_tor"][indices].data * j_units
    scene_jbs_tor = scene_data["Jbs_tor"][indices].data * j_units
    scene_zeff = scene_data["zeff"][indices].data * units.elementary_charge

    pyro = Pyro(
        eq_file=scene_eqdsk,
        eq_type="GEQDSK",
        kinetics_file=scene_cdf,
        kinetics_type="SCENE",
    )

    parallel_jtor = scene_jtor * 0.0
    bs_jtor = scene_jtor * 0.0

    for i, psi_n in enumerate(scene_psin):
        try:
            pyro.load_local(psi_n=psi_n, local_geometry="MXH")
        except Exception:
            continue

        pyro.local_species.zeff = scene_zeff[i]

        redl = Redl2021(pyro)

        # SCENE's J_tor omits the Pfirsch-Schlueter + diamagnetic part
        parallel_jtor[i] = (redl.Jphi_fsa - redl.Jphi_psdia_fsa).to(j_units)
        bs_jtor[i] = redl.Jphi_bs_fsa.to(j_units)

        # The decomposition must sum back to the total, for both definitions
        assert_allclose(
            (redl.Jphi_bs_fsa + redl.Jphi_ext_fsa + redl.Jphi_psdia_fsa).m,
            redl.Jphi_fsa.m,
            rtol=1e-10,
        )
        assert_allclose(
            (redl.Jphi_bs_eff + redl.Jphi_ext_eff + redl.Jphi_psdia_eff).m,
            redl.Jphi_eff.m,
            rtol=1e-10,
        )

        # The flux-surface average of the local J_phi(theta) profile must equal
        # the directly computed <J_phi>
        assert_allclose(
            pyro.metric_terms.flux_surface_average(redl.Jphi).m,
            redl.Jphi_fsa.m,
            rtol=1e-8,
        )

        # Enclosed plasma current shares the sign convention of J_phi
        assert np.sign(redl.Ip.m) == np.sign(redl.Jphi_fsa.m)

    assert_allclose(parallel_jtor.m, scene_jtor.m, rtol=2e-2)
    assert_allclose(bs_jtor.m, scene_jbs_tor.m, rtol=6e-2)


def test_integrate_toroidal_current():
    """
    The radially integrated current components must sum to the total, and that
    total must agree with the enclosed current obtained independently from
    Ampere's law on each surface.
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )

    result = integrate_toroidal_current(
        pyro, np.linspace(0.02, 0.9, 25), local_geometry="MXH"
    )

    components = result["Ip_bs"] + result["Ip_ext"] + result["Ip_psdia"]
    assert_allclose(components.m, result["Ip"].m, rtol=1e-10)

    # Skip the innermost surfaces, where the near-axis contribution assumed by the
    # integration dominates
    assert_allclose(
        result["Ip"][5:].to("MA").m,
        result["Ip_ampere"][5:].to("MA").m,
        rtol=1e-2,
    )

    # Integrating outwards, the enclosed current grows monotonically in magnitude
    assert np.all(np.diff(np.abs(result["Ip"].m)) > 0)


def test_Fprime_from_total_current_round_trip():
    """
    get_Fprime_from_total_current inverts the Grad-Shafranov relation that
    get_total_current uses, both for the equilibrium <J.B> and for a modified
    one (the workflow in docs/examples/example_modify_shear.py).
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )

    for psi_n in [0.3, 0.5, 0.7]:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")

        redl = Redl2021(pyro)
        metric = pyro.metric_terms

        # The F' that get_total_current used to build <J.B> must come back out
        dpsidr = metric.dpsidr * -redl.ip_ccw
        expected = metric.dB_zeta_dr / dpsidr * redl.bt_ccw

        recovered = redl.get_Fprime_from_total_current(redl.JdotB)
        assert_allclose(recovered.to(expected.units).m, expected.m, rtol=1e-10)

        # A modified <J.B> must give an F' that reproduces it when pushed back
        # through the Grad-Shafranov relation
        modified = 1.5 * redl.JdotB
        modified_Fprime = redl.get_Fprime_from_total_current(modified)

        _, F, _, mu0_dpdpsi, mu0 = redl._get_grad_shafranov_terms()
        rebuilt = (modified_Fprime * redl.B2_fsa + F * mu0_dpdpsi) / mu0

        assert_allclose(rebuilt.to(modified.units).m, modified.m, rtol=1e-10)

        # Changing the current really does change F', so the round trip above
        # is not passing simply because the two inputs coincide
        assert not np.isclose(
            modified_Fprime.to(expected.units).m, recovered.to(expected.units).m
        )
