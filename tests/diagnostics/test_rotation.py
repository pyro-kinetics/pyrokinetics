import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import simpson
from scipy.special import i0

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics


def analytic_sonic_terms(pyro, theta):
    """Analytic phi0, dphi0/dr and dphi0/dtheta for a pure plasma
    (single ion species with z_i = +1).

    Quasineutrality between the two Boltzmann-like responses (Belli & Candy,
    Phys. Plasmas 25, 032301 (2018), Eqs. (8)-(10)) gives the poloidally
    varying potential

        Phi_*(r, theta) = W(r) * (R^2 - R0^2)
        W = omega0^2 / 2 * (m_i T_e - m_e T_i) / (T_i + T_e)

    on top of which calculate_sonic_quasineutrality adds an offset fixed by
    its theta = 0 boundary conditions:

        phi0(0) = phi0_max = m_e omega0^2 R0^2 / (2 z_e)
        dphi0_dr(0) = d(phi0_max)/dr

    R0 is the outboard midplane major radius R(theta=0) and phi0 is
    normalised to tref / e.
    """
    geo = pyro.local_geometry
    species = pyro.local_species

    ion = species[[name for name in species.names if name != "electron"][0]]
    electron = species.electron

    omega0 = electron.omega0
    domega_drho = electron.domega_drho

    R, Z = geo.get_flux_surface(theta)
    dR_dtheta, dR_dr, dZ_dtheta, dZ_dr = geo.get_RZ_derivatives(theta)

    itheta0 = np.argmin(abs(theta))
    R0 = R[itheta0]
    dR0_dr = dR_dr[itheta0]

    mi, me = ion.mass, electron.mass
    Ti, Te = ion.temp, electron.temp
    dTi_dr = -Ti * ion.inverse_lt
    dTe_dr = -Te * electron.inverse_lt

    G = (mi * Te - me * Ti) / (Ti + Te)
    dG_dr = (
        (mi * dTe_dr - me * dTi_dr) * (Ti + Te)
        - (mi * Te - me * Ti) * (dTi_dr + dTe_dr)
    ) / (Ti + Te) ** 2

    W = 0.5 * omega0**2 * G
    dW_dr = omega0 * domega_drho * G + 0.5 * omega0**2 * dG_dr

    X = R**2 - R0**2
    dX_dr = 2 * (R * dR_dr - R0 * dR0_dr)
    dX_dtheta = 2 * R * dR_dtheta

    phi0_max = me * omega0**2 * R0**2 / (2 * electron.z)
    dphi0max_dr = me * omega0 * R0 / electron.z * (omega0 * dR0_dr + R0 * domega_drho)

    phi0 = phi0_max + W * X / electron.z.units
    dphi0_dr = dphi0max_dr + (dW_dr * X + W * dX_dr) / electron.z.units
    dphi0_dtheta = W * dX_dtheta / electron.z.units

    return {"phi0": phi0, "dphi0_dr": dphi0_dr, "dphi0_dtheta": dphi0_dtheta}


def analytic_flux_surface_average_density(pyro, ntheta=1024):
    """Analytic flux surface averaged density for a pure plasma (single ion
    species with z_i = +1).

    Inserting the quasineutral potential of ``analytic_sonic_terms`` into the
    Boltzmann-like response of each species,

        n_s(theta) = n_s(0) exp(chi_s(0) - chi_s(theta))
        chi_s = z_s phi0 / T_s - m_s omega0^2 R^2 / (2 T_s)

    the species dependence cancels and both densities vary identically as
    (Belli & Candy, Phys. Plasmas 25, 032301 (2018))

        n_s(theta) = n_s(0)
                     exp[omega0^2 (m_i + m_e) (R^2 - R0^2) / (2 (T_i + T_e))]

    with R0 = R(theta=0). This is averaged over the flux surface with the
    Jacobian weighting used by flux_surface_average_density.
    """
    geo = pyro.local_geometry
    species = pyro.local_species

    ion = species[[name for name in species.names if name != "electron"][0]]
    electron = species.electron

    theta = np.linspace(-np.pi, np.pi, ntheta + 1)

    R, Z = geo.get_flux_surface(theta)
    dR_dtheta, dR_dr, dZ_dtheta, dZ_dr = geo.get_RZ_derivatives(theta)

    R0 = R[np.argmin(abs(theta))]

    mi, me = ion.mass, electron.mass
    Ti, Te = ion.temp, electron.temp

    dens_shape = np.exp(
        electron.omega0**2 * (mi + me) * (R**2 - R0**2) / (2 * (Ti + Te))
    )

    jacob = R * (dR_dr * dZ_dtheta - dZ_dr * dR_dtheta)
    dens_shape_fsa = simpson((dens_shape * jacob).m, x=theta) / simpson(
        jacob.m, x=theta
    )

    return {name: species[name].dens * dens_shape_fsa for name in species.names}


def circular_pyro(rho, Rmaj, omega0):
    """GS2 template with a concentric circular flux surface of inverse aspect
    ratio rho / Rmaj, rotating at omega0"""
    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")

    geo = pyro.local_geometry
    geo.rho = rho * geo.rho.units
    geo.Rmaj = Rmaj * geo.Rmaj.units
    geo.kappa = 1.0 * geo.kappa.units
    geo.delta = 0.0 * geo.delta.units
    geo.s_kappa = 0.0 * geo.s_kappa.units
    geo.s_delta = 0.0 * geo.s_delta.units
    geo.shift = 0.0 * geo.shift.units
    geo.dZ0dr = 0.0 * geo.dZ0dr.units

    electron = pyro.local_species.electron
    electron.omega0 = omega0 * electron.omega0.units
    electron.domega_drho = 0.0 * electron.domega_drho.units

    return pyro


@pytest.mark.parametrize("omega0", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("domega_drho", [0.0, 0.7, -1.3])
def test_sonic_quasineutrality(omega0, domega_drho):
    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")
    electron = pyro.local_species.electron
    electron.omega0 = omega0 * electron.omega0.units
    electron.domega_drho = domega_drho * electron.domega_drho.units

    diag = Diagnostics(pyro)
    qn_result = diag.calculate_sonic_quasineutrality()

    expected = analytic_sonic_terms(pyro, qn_result["theta_qn"])

    for key in ("phi0", "dphi0_dr", "dphi0_dtheta"):
        numeric = qn_result[key]
        assert_allclose(
            numeric.m,
            expected[key].to(numeric.units).m,
            rtol=1e-8,
            atol=1e-7,
            err_msg=f"{key} at omega0={omega0}, domega_drho={domega_drho}",
        )


@pytest.mark.parametrize("omega0", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("domega_drho", [0.0, 0.7, -1.3])
def test_flux_surface_average_density(omega0, domega_drho):
    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")
    electron = pyro.local_species.electron
    electron.omega0 = omega0 * electron.omega0.units
    electron.domega_drho = domega_drho * electron.domega_drho.units

    diag = Diagnostics(pyro)
    fsa_dens = diag.flux_surface_average_density()

    expected = analytic_flux_surface_average_density(pyro)

    assert sorted(fsa_dens.keys()) == sorted(pyro.local_species.names)

    for name in pyro.local_species.names:
        numeric = fsa_dens[name]
        assert_allclose(
            numeric.m,
            expected[name].to(numeric.units).m,
            rtol=1e-6,
            err_msg=f"{name} at omega0={omega0}, domega_drho={domega_drho}",
        )

    # Quasineutrality is preserved by the flux surface average of a pure plasma
    charge_density = sum(
        pyro.local_species[name].z * fsa_dens[name] for name in pyro.local_species.names
    )
    assert_allclose(charge_density.m, 0.0, atol=1e-12)


def test_flux_surface_average_density_no_rotation():
    """Without rotation the density is uniform on the flux surface, so the
    flux surface average is the density held in local_species"""
    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")

    fsa_dens = Diagnostics(pyro).flux_surface_average_density()

    for name in pyro.local_species.names:
        dens = pyro.local_species[name].dens
        assert_allclose(fsa_dens[name].to(dens.units).m, dens.m, rtol=1e-14)


def test_flux_surface_average_density_large_aspect_ratio():
    r"""Known result in the large aspect ratio limit of a circular flux
    surface.

    With :math:`R = R_0(1 + \epsilon\cos\theta)`, :math:`\epsilon = \rho/R_0`,
    the density variation of a pure plasma becomes, to lowest order in
    :math:`\epsilon`,

    .. math::
        n_s = n_s(0)e^{\lambda(\cos\theta - 1)},\quad
        \lambda = \frac{\Omega_0^2(m_i + m_e)R_0\rho}{T_i + T_e}

    and the Jacobian weighting becomes uniform in :math:`\theta`, so the flux
    surface average is the modified Bessel function of the first kind

    .. math::
        \frac{\langle n_s\rangle}{n_s(0)} = e^{-\lambda}I_0(\lambda)
    """
    lambda_rotation = 1.0
    rho = 1.0
    Rmaj = 100.0

    pyro = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in")
    species = pyro.local_species
    ion = species[[name for name in species.names if name != "electron"][0]]
    electron = species.electron

    # omega0 (in units of vref / lref) that gives the requested lambda
    omega0 = np.sqrt(
        (
            lambda_rotation
            * (ion.temp + electron.temp)
            / ((ion.mass + electron.mass) * Rmaj * rho)
        ).m
    )

    pyro = circular_pyro(rho, Rmaj, omega0)
    fsa_dens = Diagnostics(pyro).flux_surface_average_density()

    expected = np.exp(-lambda_rotation) * i0(lambda_rotation)

    for name in pyro.local_species.names:
        dens = pyro.local_species[name].dens
        assert_allclose(
            fsa_dens[name].to(dens.units).m / dens.m,
            expected,
            rtol=1e-2,
            err_msg=f"{name} in large aspect ratio limit",
        )
