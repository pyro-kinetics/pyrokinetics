import numpy as np
import pytest
from numpy.testing import assert_allclose

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
