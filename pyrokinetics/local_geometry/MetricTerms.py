from cleverdict import CleverDict
from copy import deepcopy
from ..decorators import not_implemented
from ..factory import Factory
from ..constants import pi
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..typing import ArrayLike
from ..equilibrium import Equilibrium
from . import LocalGeometry
import matplotlib.pyplot as plt
from functools import cached_property
import scipy.integrate as integrate
from scipy.interpolate import interp1d


class MetricTerms:  # CleverDict
    r"""
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary
    Attributes
    ----------
    psi_n : Float
        Normalised Psi
    rho : Float
        r/a
    r_minor : Float
        Minor radius of flux surface
    a_minor : Float
        Minor radius of LCFS [m]
    Rmaj : Float
        Normalised Major radius (Rmajor/a_minor)
    Z0 : Float
        Normalised vertical position of midpoint (Zmid / a_minor)
    f_psi : Float
        Torodial field function
    B0 : Float
        Toroidal field at major radius (f_psi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math: `q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear `r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta' = `2 \mu_0 \partial p \partial \rho 1/B0^2`

    R_eq : Array
        Equilibrium R data used for fitting
    Z_eq : Array
        Equilibrium Z data used for fitting
    b_poloidal_eq : Array
        Equilibrium B_poloidal data used for fitting
    theta_eq : Float
        theta values for equilibrium data

    R : Array
        Fitted R data
    Z : Array
        Fitted Z data
    b_poloidal : Array
        Fitted B_poloidal data
    theta : Float
        Fitted theta data

    dRdtheta : Array
        Derivative of fitted `R` w.r.t `\theta`
    dRdr : Array
        Derivative of fitted `R` w.r.t `r`
    dZdtheta : Array
        Derivative of fitted `Z` w.r.t `\theta`
    dZdr : Array
        Derivative of fitted `Z` w.r.t `r`
    """

    def __init__(self, local_geometry: LocalGeometry):
        self.regulartheta = np.linspace(-np.pi, np.pi, 1000)
        self.R, self.Z = local_geometry.get_flux_surface(self.regulartheta)
        (
            self.dRdtheta,
            self.dRdr,
            self.dZdtheta,
            self.dZdr,
        ) = local_geometry.get_RZ_derivatives(
            self.regulartheta, normalised=True
        )  # (local_geometry.theta)
        (
            self.d2Rdtheta2,
            self.d2Rdrdtheta,
            self.d2Zdtheta2,
            self.d2Zdrdtheta,
        ) = local_geometry.get_RZ_second_derivatives(
            self.regulartheta, normalised=True
        )  # (local_geometry.theta)
        self.Jac = self.R * (self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta)
        self.q = local_geometry.q
        self.dpsidr = local_geometry.dpsidr
        self.Y = integrate.trapezoid(self.Jac / (self.R**2), self.regulartheta) / (
            2.0 * np.pi
        )
        self.dqdr = self.q * local_geometry.shat / local_geometry.rho
        print(self.dqdr * 2 * np.pi)
        self.d2psidr2 = 0.0  # Take d2psidr2 = 0. This quantity doesn't appear in any
        # physical quantities, and so is essentially arbitrary
        self.mu_0 = 0.0  # fixme
        self.dPdr = 0.0  # fixme

    @cached_property
    def get_toroidal_system_covariant_metric_comps(self):
        self._g_vrvr = self.dRdr**2 + self.dZdr**2
        self._g_vrvtheta = self.dRdr * self.dRdtheta + self.dZdr * self.dZdtheta
        self._g_vthetavtheta = self.dRdtheta**2 + self.dZdtheta**2
        self._g_zetazeta = self.R**2
        return self._g_vrvr, self._g_vrvtheta, self._g_vthetavtheta, self._g_zetazeta

    @cached_property
    def get_toroidal_system_covariant_metric_derivatives(self):
        # self._partial_g_vrvr_partial_vtheta = 2 * self.dRdr * self.d2Rdrdtheta + 2 * self.dZdr * self.d2Zdrdtheta
        self._partial_g_vrvtheta_partial_vtheta = (
            self.d2Rdrdtheta * self.dRdtheta
            + self.d2Rdtheta2 * self.dRdr
            + self.d2Zdrdtheta * self.dZdtheta
            + self.d2Zdtheta2 * self.dZdr
        )
        self._partial_g_vthetavtheta_partial_vr = 2 * (
            self.dRdtheta * self.d2Rdrdtheta + self.dZdtheta * self.d2Zdrdtheta
        )
        self._partial_Jac_partial_vtheta = (
            self.dRdtheta * self.Jac / self.R
        ) + self.R * (
            self.d2Rdrdtheta * self.dZdtheta
            + self.dRdr * self.d2Zdtheta2
            - self.d2Rdtheta2 * self.dZdr
            - self.dRdtheta * self.d2Zdrdtheta
        )
        check = integrate.cumulative_trapezoid(
            self._partial_Jac_partial_vtheta, self.regulartheta
        )
        check = list(check)
        check.insert(0, 0.0)
        check = np.array(check)
        self._check = check
        return (
            self._partial_g_vrvtheta_partial_vtheta,
            self._partial_g_vthetavtheta_partial_vr,
            self._partial_Jac_partial_vtheta,
            self._check,
        )  # self._partial_g_vrvr_partial_vtheta, self._check

    @cached_property
    def get_B_cov_zeta(self):
        self._B_cov_zeta = self.q * self.dpsidr / self.Y
        return self._B_cov_zeta

    @cached_property
    def get_dB_cov_zeta_dr(self):
        (
            g_vrvr,
            g_vrvtheta,
            g_vthetavtheta,
            g_zetazeta,
        ) = self.get_toroidal_system_covariant_metric_comps
        (
            partial_g_vrvtheta_partial_vtheta,
            partial_g_vthetavtheta_partial_vr,
            partial_Jac_partial_vtheta,
            check,
        ) = self.get_toroidal_system_covariant_metric_derivatives
        B_cov_zeta = self.get_B_cov_zeta
        H = self.Y + ((self.q / self.Y) ** 2) * (
            integrate.trapezoid(
                self.Jac**3 / ((self.R**4) * g_vthetavtheta), self.regulartheta
            )
            / (2.0 * np.pi)
        )
        term1 = self.Y * self.dqdr / self.q
        term2 = -(
            integrate.trapezoid(
                -2.0 * self.Jac * self.dRdr / (self.R**3), self.regulartheta
            )
            / (2.0 * np.pi)
        )
        term3 = -(self.mu_0 * self.dPdr / (self.dpsidr**2)) * (
            integrate.trapezoid(
                self.Jac**3 / ((self.R**2) * g_vthetavtheta), self.regulartheta
            )
            / (2.0 * np.pi)
        )
        to_integrate = (self.Jac / ((self.R**2) * g_vthetavtheta)) * (
            partial_g_vrvtheta_partial_vtheta
            - partial_g_vthetavtheta_partial_vr
            - (g_vrvtheta * partial_Jac_partial_vtheta / self.Jac)
        )
        term4 = integrate.trapezoid(to_integrate, self.regulartheta) / (2.0 * np.pi)
        self._dB_cov_zeta_dr = (B_cov_zeta / H) * (term1 + term2 + term3 + term4)
        return self._dB_cov_zeta_dr

    @cached_property
    def get_dJac_dr(self):
        (
            g_vrvr,
            g_vrvtheta,
            g_vthetavtheta,
            g_zetazeta,
        ) = self.get_toroidal_system_covariant_metric_comps
        (
            partial_g_vrvtheta_partial_vtheta,
            partial_g_vthetavtheta_partial_vr,
            partial_Jac_partial_vtheta,
            check,
        ) = self.get_toroidal_system_covariant_metric_derivatives
        B_cov_zeta = self.get_B_cov_zeta
        dB_cov_zeta_dr = self.get_dB_cov_zeta_dr
        term1 = self.Jac * self.d2psidr2 / self.dpsidr
        term2 = -(self.Jac / g_vthetavtheta) * (
            partial_g_vrvtheta_partial_vtheta
            - partial_g_vthetavtheta_partial_vr
            - (g_vrvtheta * partial_Jac_partial_vtheta / self.Jac)
        )
        term3 = (
            (self.mu_0 * self.dPdr / (self.dpsidr**2))
            * (self.Jac**3)
            / g_vthetavtheta
        )
        term4 = (
            (B_cov_zeta * dB_cov_zeta_dr / (self.dpsidr**2))
            * (self.Jac**3)
            / ((self.R**2) * g_vthetavtheta)
        )
        self._dJac_dr = term1 + term2 + term3 + term4
        return self._dJac_dr

    @cached_property
    def get_dalpha_dvtheta(self):
        self._dalpha_dvtheta = self.q * self.Jac / ((self.R**2) * self.Y)
        return self._dalpha_dvtheta

    @cached_property
    def get_d2alpha_drdvtheta(self):  # sometimes known as 'local shear'
        B_cov_zeta = self.get_B_cov_zeta
        dB_cov_zeta_dr = self.get_dB_cov_zeta_dr
        dJac_dr = self.get_dJac_dr
        term1 = dB_cov_zeta_dr * self.Jac / (self.dpsidr * (self.R**2))
        term2 = (
            -self.d2psidr2
            * self.Jac
            * B_cov_zeta
            / ((self.dpsidr**2) * (self.R**2))
        )
        term3 = B_cov_zeta * dJac_dr / (self.dpsidr * (self.R**2))
        term4 = -(2.0 * self.dRdr / (self.R**3)) * (
            B_cov_zeta * self.Jac / self.dpsidr
        )
        self._d2alpha_drdvtheta = term1 + term2 + term3 + term4
        return self._d2alpha_drdvtheta

    @cached_property
    def get_dalpha_dr(self):
        d2alpha_drdvtheta = self.get_d2alpha_drdvtheta
        dalpha_dr = integrate.cumulative_trapezoid(d2alpha_drdvtheta, self.regulartheta)
        dalpha_dr = list(dalpha_dr)
        dalpha_dr.insert(0, 0.0)
        dalpha_dr = np.array(dalpha_dr)
        f = interp1d(self.regulartheta, dalpha_dr)
        self._dalpha_dr = dalpha_dr - f(
            0.0
        )  # set dalpha/dr(r,theta=0.0)=0.0, assumed by codes
        return self._dalpha_dr

    @cached_property
    def get_FA_covariant_metric_components(self):
        dalpha_dr = self.get_dalpha_dr
        dalpha_dvtheta = self.get_dalpha_dvtheta
        (
            g_vrvr,
            g_vrvtheta,
            g_vthetavtheta,
            g_zetazeta,
        ) = self.get_toroidal_system_covariant_metric_comps

        self._g_rr = g_vrvr + (dalpha_dr**2) * g_zetazeta
        self._g_ralpha = -dalpha_dr * g_zetazeta
        self._g_rt = g_vrvtheta + dalpha_dr * dalpha_dvtheta * g_zetazeta
        self._g_alphaalpha = g_zetazeta
        self._g_alphat = -dalpha_dvtheta * g_zetazeta
        self._g_tt = g_vthetavtheta + (dalpha_dvtheta**2) * g_zetazeta

        return (
            self._g_rr,
            self._g_ralpha,
            self._g_rt,
            self._g_alphaalpha,
            self._g_alphat,
            self._g_tt,
        )
