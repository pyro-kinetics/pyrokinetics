import netCDF4 as nc
import numpy as np
import pytest
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


def test_radial_coordinate_rho_tor():
    """
    ``radial_coordinate="rho_tor"`` rebases the effective toroidal current density
    onto rho = sqrt(Psi_tor / (pi B_geo)), the toroidal flux label used by JETTO
    and other transport codes.

    Only the ``(1 / 2 pi x) dIp/dx`` family may depend on that choice: the
    flux-surface averaged densities, the parallel current and the Ampere law Ip
    are all label independent and must be untouched.
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )
    eq = pyro.eq

    # Independent value for the conversion factor d(r^2)/d(rho^2), by finite
    # difference over the equilibrium splines. _get_radial_label instead uses the
    # analytic 2 pi r B_geo (dr/dpsi) / q, in which Psi_tor cancels, so this is a
    # genuine cross-check of that algebra rather than a restatement of it.
    psi_n_fine = np.linspace(0.05, 0.95, 400)
    B_geo = np.abs(eq.B_0)
    a_tor = np.sqrt(eq.psi_tor(1.0) / (np.pi * B_geo))
    r_fine = eq.r_minor(psi_n_fine).to("meter").m
    rho_fine = (a_tor * eq.rho_tor(psi_n_fine)).to("meter").m
    factor_fd = np.gradient(r_fine**2, rho_fine**2)

    for psi_n in [0.3, 0.5, 0.7]:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")

        default = Redl2021(pyro)
        rho_tor = Redl2021(pyro, radial_coordinate="rho_tor")

        # The default is the minor radius, and is an exact no-op
        assert default.label_factor.m == 1.0
        assert_allclose(default.rho_label.m, pyro.metric_terms.rho.m, rtol=1e-12)

        # The analytic factor agrees with the finite difference one
        assert_allclose(
            rho_tor.label_factor.m,
            np.interp(psi_n, psi_n_fine, factor_fd),
            rtol=1e-2,
        )

        # rho_label is a length, and is the toroidal flux label
        assert_allclose(
            rho_tor.rho_label.to("meter").m,
            np.interp(psi_n, psi_n_fine, rho_fine),
            rtol=1e-3,
        )

        # Every effective current density picks up exactly that one factor
        for name in ["Jphi_eff", "Jphi_bs_eff", "Jphi_ext_eff", "Jphi_psdia_eff"]:
            assert_allclose(
                getattr(rho_tor, name).m,
                (getattr(default, name) * rho_tor.label_factor).m,
                rtol=1e-12,
            )

        # dIp/dx carries dr/dx instead, since dIp/dx = 2 pi x Jphi_eff(x)
        drdrho = (rho_tor.label_factor * rho_tor.rho_label / default.rho_label).m
        for name in ["dIp_drho", "dIp_bs_drho", "dIp_ext_drho", "dIp_psdia_drho"]:
            assert_allclose(
                getattr(rho_tor, name).m,
                (getattr(default, name) * drdrho).m,
                rtol=1e-12,
            )

        # The decomposition still closes on the new label
        assert_allclose(
            (rho_tor.Jphi_bs_eff + rho_tor.Jphi_ext_eff + rho_tor.Jphi_psdia_eff).m,
            rho_tor.Jphi_eff.m,
            rtol=1e-10,
        )

        # Nothing that is not a (1 / 2 pi x) dIp/dx quantity may move
        for name in ["Jphi_fsa", "Jphi_bs_fsa", "Jphi_psdia_fsa", "JdotB", "Ip"]:
            assert_allclose(
                getattr(rho_tor, name).m, getattr(default, name).m, rtol=1e-12
            )


def test_radial_coordinate_errors():
    """
    ``"rho_tor"`` needs a global equilibrium, because the conversion is pinned by
    B_geo and a local GK input does not carry it. Unknown labels are rejected.
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )
    pyro.load_local(psi_n=0.5, local_geometry="MXH")

    with pytest.raises(ValueError, match="'r_minor' or 'rho_tor'"):
        Redl2021(pyro, radial_coordinate="rho_toroidal")

    # A GK input file alone gives no equilibrium, so B_geo is unavailable
    local_only = Pyro(gk_file=template_dir / "input.gs2")
    with pytest.raises(ValueError, match="requires a global Equilibrium"):
        Redl2021(local_only, ntheta=128, radial_coordinate="rho_tor")
