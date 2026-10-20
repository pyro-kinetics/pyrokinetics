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

    # 7%: with alpha as published (Redl eq. 21, alpha0 inside the 1 + 0.18 sqrt(nu*_i)
    # denominator) this case sits 3-6% above SCENE; the shipped transcription error
    # had pulled it about 0.5% lower
    assert_allclose(redl_jbsdotb_b2.m, scene_jbsdotb_b2.m, rtol=7e-2)
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
    # 7%: see test_bootstrap_current (alpha as published)
    assert_allclose(bs_jtor.m, scene_jbs_tor.m, rtol=7e-2)


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

        # The F' that get_total_current used to build <J.B> must come back out.
        # dpsi/dr follows the plasma current; metric.dpsidr carries sign(q).
        dpsidr = np.abs(metric.dpsidr) * redl.ip_ccw
        expected = metric.dB_zeta_dr / dpsidr * redl.bt_ccw

        recovered = redl.get_Fprime_from_total_current(redl.JdotB)
        assert_allclose(recovered.to(expected.units).m, expected.m, rtol=1e-10)

        # A modified <J.B> must give an F' that reproduces it when pushed back
        # through the Grad-Shafranov relation
        modified = 1.5 * redl.JdotB
        modified_Fprime = redl.get_Fprime_from_total_current(modified)

        _, F, _, mu0_dpdpsi, mu0 = redl._get_grad_shafranov_terms()
        rebuilt = -(modified_Fprime * redl.B2_fsa + F * mu0_dpdpsi) / mu0

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


@pytest.mark.parametrize("model", [Redl2021, Sauter1999])
def test_main_ion_is_majority_thermal_ion(model):
    """
    ``get_kinetic_species_data`` and ``_get_collisionality_ion`` must agree on
    which species is "the" ion.

    Both stand in for the single ion species the Sauter/Redl fits are written
    for: the collisionality takes its charge, temperature and Coulomb logarithm,
    and the pressure terms substitute its temperature and temperature gradient
    for those of the fast species. Picking it by ``LocalSpecies`` order rather
    than by density made the two disagree whenever the majority ion was not
    listed first.
    """

    pyro = Pyro(
        eq_file=template_dir / "test.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "jetto.jsp",
        kinetics_type="JETTO",
    )
    pyro.load_local(psi_n=0.5, local_geometry="MXH")
    ls = pyro.local_species

    # The template carries a fast species, which must be left out of the pick
    assert "deuterium_fast" in ls.names
    bs = model(pyro)
    assert "deuterium_fast" not in bs._get_thermal_ion_names()
    assert bs._get_main_ion().name == "deuterium"
    assert bs._get_collisionality_ion()[0].name == "deuterium"

    # Demote the first-listed ion to a trace species: the pick must follow
    # density, not order, and the fast species must inherit the new majority's
    # temperature gradient.
    ion_names = [name for name in ls.names if ls[name].z.m > 0]
    assert ion_names[0] == "deuterium"
    ls["deuterium"].dens = 1e-4 * ls["deuterium"].dens.units
    ls["impurity1"].inverse_lt = 5.0 * ls["deuterium"].inverse_lt

    bs = model(pyro)
    assert bs._get_main_ion().name == "impurity1"
    assert bs._get_collisionality_ion()[0].name == "impurity1"

    i_fast = ion_names.index("deuterium_fast")
    assert_allclose(
        bs.dlnTi_dpsi[i_fast].m,
        (ls["impurity1"].inverse_lt / pyro.local_geometry.dpsidr).m,
        rtol=1e-12,
    )


def test_trapped_fraction_quadrature_converged():
    """
    The pitch angle quadrature must be converged at the default ``nlambda``, and
    stay converged in towards the magnetic axis.

    ``get_trapped_fraction`` integrates in :math:`u = \\sqrt{1 - \\lambda
    B_{max}}` rather than in :math:`\\lambda` because the integrand collapses
    over a boundary layer of width :math:`\\sim 2 \\epsilon` at the
    trapped-passing boundary. On a uniform grid in :math:`\\lambda` that layer
    falls between grid points as the flux surface shrinks, so the quadrature
    underestimates :math:`f_t` near the axis and eventually returns a negative
    value. Compare against a heavily refined grid, which the substitution makes
    unnecessary but which any implementation has to agree with.
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )

    previous = 0.0
    for psi_n in (0.002, 0.01, 0.1, 0.5):
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")

        ftrap = Redl2021(pyro).trapped_fraction
        converged = Redl2021(pyro, nlambda=4001).trapped_fraction

        assert_allclose(ftrap, converged, rtol=1e-4)

        # A trapped fraction outside (0, 1) is unphysical, and would produce
        # NaNs in the coefficients that take its square root
        assert 0.0 < ftrap < 1.0

        # f_t ~ 1.46 sqrt(eps) on a circular surface, so it must fall towards
        # the axis rather than degrade into quadrature noise
        assert ftrap > previous
        previous = ftrap


def test_trapped_fraction_nlambda_validated():
    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )
    pyro.load_local(psi_n=0.5, local_geometry="MXH")

    with pytest.raises(ValueError, match="nlambda must be at least 3"):
        Redl2021(pyro, nlambda=2)


def _flipped_geqdsk(path, flip_ip, flip_bt):
    """
    Write a COCOS-1-consistent copy of step.geqdsk with Ip and/or B0 reversed

    In COCOS 1, ``sign(psi_lcfs - psi_axis) = sign(Ip)`` and
    ``sign(q) = sign(Ip * B0)``, so reversing a field direction means reversing
    every quantity that carries an odd power of it: reversing ``B0`` flips ``F``
    and ``q`` but leaves ``FF'`` alone, and reversing ``Ip`` flips ``psi``, ``q``,
    ``p'`` and ``FF'``.
    """
    from dataclasses import asdict

    from freeqdsk import geqdsk

    with open(template_dir / "step.geqdsk") as f:
        data = asdict(geqdsk.read(f))

    if flip_bt:
        data["fpol"] = -data["fpol"]
        data["bcentr"] = -data["bcentr"]
        data["qpsi"] = -data["qpsi"]

    if flip_ip:
        data["cpasma"] = -data["cpasma"]
        data["psi"] = -data["psi"]
        data["simagx"] = -data["simagx"]
        data["sibdry"] = -data["sibdry"]
        data["qpsi"] = -data["qpsi"]
        data["pprime"] = -data["pprime"]
        data["ffprime"] = -data["ffprime"]

    for key in ("comment", "shot", "nx", "ny", "r_grid", "z_grid"):
        data.pop(key, None)

    with open(path, "w") as f:
        geqdsk.write(data, f, label="cocos_flip")

    return path


@pytest.mark.parametrize(
    "flip_ip,flip_bt", [(False, False), (False, True), (True, False), (True, True)]
)
def test_current_signs_follow_cocos(tmp_path, flip_ip, flip_bt):
    """
    Currents must follow the equilibrium's field directions, not their magnitudes

    pyrokinetics' psi increases outward for a counter-clockwise plasma current, so
    the parallel current moments carry ``ip_ccw * bt_ccw`` and the toroidal
    currents carry ``ip_ccw``. Reversing either direction must reverse them and
    leave the magnitudes untouched.
    """

    eq_file = _flipped_geqdsk(
        tmp_path / f"step_{int(flip_ip)}{int(flip_bt)}.geqdsk", flip_ip, flip_bt
    )

    pyro = Pyro(
        eq_file=eq_file,
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )
    pyro.load_local(psi_n=0.3, local_geometry="MXH")

    bootstrap = Redl2021(pyro)

    ip_ccw = int(bootstrap.ip_ccw.m)
    bt_ccw = int(bootstrap.bt_ccw.m)
    assert ip_ccw == (-1 if flip_ip else 1)
    assert bt_ccw == (-1 if flip_bt else 1)

    # dpsi/dr follows the plasma current alone, even though the q it is rebuilt
    # from follows ip_ccw * bt_ccw
    assert np.sign(bootstrap._get_grad_shafranov_terms()[0].m) == ip_ccw
    assert np.sign(pyro.local_geometry.dpsidr.m) == ip_ccw

    # Parallel current moments are co-directed with the field when the plasma
    # current and toroidal field are aligned
    assert np.sign(bootstrap.JdotB.m) == ip_ccw * bt_ccw
    assert np.sign(bootstrap.JbsdotB.m) == ip_ccw * bt_ccw

    # Toroidal currents follow the plasma current
    assert np.sign(bootstrap.Ip.m) == ip_ccw
    assert np.sign(bootstrap.Jphi_fsa.m) == ip_ccw
    assert np.sign(bootstrap.Jphi_bs_fsa.m) == ip_ccw

    # The bootstrap current is a fraction of the total, not larger than it
    assert abs(bootstrap.JbsdotB.m) < abs(bootstrap.JdotB.m)

    # <J_phi> is exactly the sum of its three parts in every sign convention
    assert_allclose(
        bootstrap.Jphi_fsa.m,
        (bootstrap.Jphi_bs_fsa + bootstrap.Jphi_ext_fsa + bootstrap.Jphi_psdia_fsa).m,
        rtol=1e-10,
    )


@pytest.mark.parametrize("model", [Redl2021, Sauter1999])
def test_bootstrap_sign_reverses_with_gradients(model):
    """
    A hollow profile drives the bootstrap current backwards

    <J_bs.B> is linear in the logarithmic gradients, so reversing every gradient
    must reverse it exactly. Taking its magnitude instead would leave both the
    bootstrap current and the external remainder unchanged here.
    """

    pyro = Pyro(
        eq_file=template_dir / "step.geqdsk",
        eq_type="GEQDSK",
        kinetics_file=template_dir / "scene.cdf",
        kinetics_type="SCENE",
    )
    pyro.load_local(psi_n=0.3, local_geometry="MXH")

    peaked = model(pyro)
    assert peaked.JbsdotB.m > 0.0

    for name in pyro.local_species.names:
        species = pyro.local_species[name]
        species.inverse_ln = -species.inverse_ln
        species.inverse_lt = -species.inverse_lt

    hollow = model(pyro)

    assert_allclose(hollow.JbsdotB.m, -peaked.JbsdotB.m, rtol=1e-10)
    assert_allclose(hollow.Jbs.m, -peaked.Jbs.m, rtol=1e-10)

    # <J.B> comes from the equilibrium and does not depend on the kinetic
    # profiles, so the external remainder must absorb the whole reversal
    assert_allclose(hollow.JdotB.m, peaked.JdotB.m, rtol=1e-10)
    assert_allclose(
        (hollow.JextdotB - peaked.JextdotB).m, 2 * peaked.JbsdotB.m, rtol=1e-10
    )


def _published_alpha_sauter(ft, nui):
    """Sauter et al., Phys. Plasmas 6, 2834 (1999), Eqs. (17a)-(17b) as printed, with the
    + sign of the 0.315 term (the 2002 erratum; CHEASE and NEO): the
    1 + 0.5 sqrt(nu*_i) denominator divides alpha0 as well."""
    a0 = -1.17 * (1 - ft) / (1 - 0.22 * ft - 0.19 * ft**2)
    s = np.sqrt(nui)
    return (
        (a0 + 0.25 * (1 - ft**2) * s) / (1 + 0.5 * s) + 0.315 * nui**2 * ft**6
    ) / (1 + 0.15 * nui**2 * ft**6)


def _published_alpha_redl(ft, nui, zeff):
    """Redl et al., Phys. Plasmas 28, 022502 (2021), Eqs. (20)-(21): the
    1 + 0.18 sqrt(nu*_i) denominator divides alpha0 as well."""
    a0 = (
        -(0.62 + 0.055 * (zeff - 1))
        / (0.53 + 0.17 * (zeff - 1))
        * (1 - ft)
        / (1 - (0.31 - 0.065 * (zeff - 1)) * ft - 0.25 * ft**2)
    )
    s = np.sqrt(nui)
    return (
        (a0 + 0.7 * zeff * np.sqrt(ft) * s) / (1 + 0.18 * s) - 0.002 * nui**2 * ft**6
    ) / (1 + 0.004 * nui**2 * ft**6)


@pytest.mark.parametrize("zeff", [1.0, 2.5, 4.0])
def test_alpha_matches_published_formula(zeff):
    """alpha against the papers' formulas over the banana-plateau-collisional range.

    Built without a Pyro object: get_alpha only reads trapped_fraction, nu_star_i and
    Zeff. At nu*_i = 1 dividing only the nu*_i term instead of the whole numerator
    (the transcription error this pins) moves alpha by 0.1-0.25, far outside rtol."""
    ft, nui = np.meshgrid([0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 1e-3, 0.1, 1.0, 10.0, 100.0])
    for cls, want in (
        (Sauter1999, _published_alpha_sauter(ft, nui)),
        (Redl2021, _published_alpha_redl(ft, nui, zeff)),
    ):
        model = object.__new__(cls)
        model.trapped_fraction, model.nu_star_i, model.Zeff = ft, nui, zeff
        assert_allclose(model.get_alpha(), want, rtol=1e-12, atol=1e-14)

    # collisionless limit: alpha -> alpha0
    model = object.__new__(Sauter1999)
    model.trapped_fraction, model.nu_star_i, model.Zeff = ft[0], np.zeros(5), zeff
    assert_allclose(model.get_alpha(), model.get_alpha0(), rtol=1e-12)
