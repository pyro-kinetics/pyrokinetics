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
        :math:`\beta' = 2 \mu_0 \partial p \partial \rho 1/B0^2`

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

    # This class uses two (somewhat similar) coordinate systems. These are the 'toroidal system' and the 'field-aligned system'.
    # The toroidal system is defined on page 73, with symbols {\varrho, \vartheta, \zeta}. For the purpose of variable names,
    # we shall use {tr, tt, zeta}. (tr = 'toroidal system radial', tt = 'toroidal system theta').
    # The field-aligned system is defined on page 82, with symbols {r, \alpha, \theta}. For the purpose of variable names,
    # we shall use {fr, alpha, ft}. (fr = 'field-aligned radial', ft = 'field-aligned theta').

    # Even though the transformations are defined such that tr = fr and tt = ft (see equations 3.149-3.151), we change
    # the variable names due to the metric terms, which become ambiguous if the same symbol is used.
    # For example, looking at equation D.87, one finds g_ftft = g_tttt + (q + dG_0dtheta)^2 * g_zetazeta. This would be
    # unclear if both g_ftft and g_tttt were labelled 'g_thetatheta'.

    # Note that for axisymmetric quantities (which is almost all geometrical quantities in tokamaks) partial derivatives
    # with respect to the radial direction and the theta direction in the two systems are identical (see equation D.91).
    # Where this is the case, 'r' and 'theta' will be used in the variable names, to keep more in line with Pyrokinetics convention.

    # Almost all calculations are done using normalised quantities, for which the variable names have a subscript '_N'. The
    # relevant normalising quantities are the minor radius 'a' for the length, and the magnetic field B0 = q dpsidr / (R0 * < Jac g^zetazeta >)
    # Both partial derivatives and derivatives of univariate functions will be written using 'd'. (Little information is lost here).

    # The \varrho (= 'tr') used in the thesis has dimensions of length, and is NOT the normalised rho = r / a defined by Pyrokinetics.
    # In fact, rho = tr / a. Such is the nature of bringing together two works after they have been mainly written.

    def __init__(self, local_geometry: LocalGeometry):
        self.regulartheta = np.linspace(-np.pi, np.pi, 1000)  # theta grid

        self.R_N, self.Z_N = local_geometry.get_flux_surface(
            self.regulartheta, normalised=True
        )  # R/a, Z/a
        (
            self.dRdtheta_N,  # (dR/dtheta)/a
            self.dRdr_N,  # dR/dr, already normalised
            self.dZdtheta_N,  # (dZ/dtheta)/a
            self.dZdr_N,  # dZ/dr, already normalised
        ) = local_geometry.get_RZ_derivatives(self.regulartheta, normalised=True)
        (
            self.d2Rdtheta2_N,  # (d2R/dtheta2)/a
            self.d2Rdrdtheta_N,  # d2R/drdtheta, already normalised
            self.d2Zdtheta2_N,  # (d2Z/dtheta2)/a
            self.d2Zdrdtheta_N,  # d2Z/drdtheta, already normalised
        ) = local_geometry.get_RZ_second_derivatives(self.regulartheta, normalised=True)

        self.Jac_N = self.R_N * (
            self.dRdr_N * self.dZdtheta_N - self.dZdr_N * self.dRdtheta_N
        )  # Jac / a^2, equation D.35
        # NOTE: The Jacobians of the toroidal system and the
        # field-aligned system are the same
        self.g_cont_zetazeta_N = (
            1 / self.R_N**2
        )  # a^2 * g^zetazeta = 1 / (R / a)^2, equation D.34 ('cont' for contravariant)
        self.q = local_geometry.q  # safety factor
        self.Y = integrate.trapezoid(
            self.Jac_N * self.g_cont_zetazeta_N, self.regulartheta
        ) / (
            2.0 * np.pi
        )  # frequently occuring quantity, already normalised.
        # poloidal average of Jac * g^zetazeta: <Jac * g^zetazeta>_P (see equation 3.133 for average definition)
        self.dpsidr_N = (
            self.Y * local_geometry.Rmaj / self.q
        )  # This defines the reference magnetic field as B0:
        # dpsidr_N = dpsidr / (B0 * a) = <Jac * g^zetazeta>_P * (R0 / a) / q
        self.dqdr_N = (
            self.q * local_geometry.shat / local_geometry.rho
        )  # safety factor derivative, dqdr_N = a * dqdr = q * shat / (r / a)
        self.d2psidr2_N = 0.0  # Take d2psidr2 = 0. This quantity doesn't appear in any
        # physical quantities, and so is essentially arbitrary. d2psidr2_N = d2psidr2 / B0

        self.mu0dPdr_N = (
            local_geometry.beta_prime / 2.0
        )  # mu0_N = mu0 * n_ref * T_ref / B0^2 = beta / 2 (normalised mu0)
        # dPdr_N = (a / (n_ref * T_ref)) * dPdr (normalised pressure gradient)
        # mu0dPdr_N = (a / B0^2) * mu0 * dPdr = beta_prime / 2 (normalised product)
        self.sigma_alpha = 1  # either 1 or -1, affects field-aligned metric components (included after completion of thesis).
        # Defined via alpha = sigma_alpha * (q \vartheta - \zeta + G_0) (equation 3.150)
        # If 1, then {r, alpha, theta} forms a right-handed system (as in thesis, more 'x,y,z' style)
        # If -1, then {r, theta, alpha} forms a right-handed system (as in CGYRO)
        # In both cases, theta increases in the anti-clockwise direction

    @cached_property
    def get_toroidal_system_covariant_metric_comps(self):
        self.g_trtr_N = (
            self.dRdr_N**2 + self.dZdr_N**2
        )  # eq D.30, already normalised
        self.g_trtt_N = (
            self.dRdr_N * self.dRdtheta_N + self.dZdr_N * self.dZdtheta_N
        )  # eq D.31, g_trtt / a
        self.g_tttt_N = (
            self.dRdtheta_N**2 + self.dZdtheta_N**2
        )  # eq D.32, g_tttt / a^2
        self.g_zetazeta_N = self.R_N**2  # eq D.33, g_zetazeta / a^2
        return self.g_trtr_N, self.g_trtt_N, self.g_tttt_N, self.g_zetazeta_N

    @cached_property
    def get_toroidal_system_covariant_metric_derivatives(self):
        self.dg_trtt_dtheta_N = (
            self.d2Rdrdtheta_N * self.dRdtheta_N
            + self.d2Rdtheta2_N * self.dRdr_N
            + self.d2Zdrdtheta_N * self.dZdtheta_N
            + self.d2Zdtheta2_N * self.dZdr_N
        )  # differentiate eq D.31 w.r.t theta, dg_trtt_dtheta / a
        self.dg_tttt_dr_N = 2 * (
            self.dRdtheta_N * self.d2Rdrdtheta_N + self.dZdtheta_N * self.d2Zdrdtheta_N
        )  # differentiate eq D.32 w.r.t r, dg_tttt_dr / a
        self.dJac_dtheta_N = (self.dRdtheta_N * self.Jac_N / self.R_N) + self.R_N * (
            self.d2Rdrdtheta_N * self.dZdtheta_N
            + self.dRdr_N * self.d2Zdtheta2_N
            - self.d2Rdtheta2_N * self.dZdr_N
            - self.dRdtheta_N * self.d2Zdrdtheta_N
        )  # differentiate eq D.35 w.r.t theta, dJac_dtheta / a^2
        return self.dg_trtt_dtheta_N, self.dg_tttt_dr_N, self.dJac_dtheta_N

    @cached_property  # calculate contravariant metric components of toroidal system using covariant components and equation C.50
    def get_toroidal_system_contravariant_metric_components(self):
        (
            g_trtr_N,
            g_trtt_N,
            g_tttt_N,
            g_zetazeta_N,
        ) = self.get_toroidal_system_covariant_metric_comps

        self.g_cont_trtr_N = g_tttt_N * g_zetazeta_N / (self.Jac_N**2)
        self.g_cont_trtt_N = -g_trtt_N * g_zetazeta_N / (self.Jac_N**2)
        self.g_cont_tttt_N = g_trtr_N * g_zetazeta_N / (self.Jac_N**2)
        # g_cont_zetazeta_N already calculated
        return (
            self.g_cont_trtr_N,
            self.g_cont_trtt_N,
            self.g_cont_tttt_N,
            self.g_cont_zetazeta_N,
        )

    @cached_property
    def get_B_cov_zeta(self):
        self.B_zeta_N = (
            self.q * self.dpsidr_N / self.Y
        )  # equation 3.132, B_zeta / (B0 * a). Note B_zeta is a covariant component
        # and does NOT have dimensions of magnetic field, but of (magnetic field * length).
        # This is also known as the 'current function', I, and is a flux function.
        # Note B0 is defined as B0 = B_zeta / R0, and thus because of our normalisations,
        # self.B_zeta_N should equal R0 / a.
        return self.B_zeta_N

    @cached_property
    def get_dB_cov_zeta_dr(self):  # eq 3.139, dB_zeta_dr / B0
        (
            g_trtr_N,
            g_trtt_N,
            g_tttt_N,
            g_zetazeta_N,
        ) = self.get_toroidal_system_covariant_metric_comps
        (
            dg_trtt_dtheta_N,
            dg_tttt_dr_N,
            dJac_dtheta_N,
        ) = self.get_toroidal_system_covariant_metric_derivatives
        B_zeta_N = self.get_B_cov_zeta

        H = self.Y + ((self.q / self.Y) ** 2) * (
            integrate.trapezoid(
                (self.Jac_N**3) * (self.g_cont_zetazeta_N**2) / g_tttt_N,
                self.regulartheta,
            )
            / (2.0 * np.pi)
        )  # eq 3.140, already normalised.
        # Uses B_zeta / dpsidr = q / Y
        term1 = self.Y * self.dqdr_N / self.q
        term2 = -(
            integrate.trapezoid(
                -2.0 * self.Jac_N * self.dRdr_N / (self.R_N**3), self.regulartheta
            )
            / (2.0 * np.pi)
        )  # uses dg^zetazeta/dr = - (2 / R^3) * dRdr
        term3 = -(self.mu0dPdr_N / (self.dpsidr_N**2)) * (
            integrate.trapezoid(
                (self.Jac_N**3) * self.g_cont_zetazeta_N / g_tttt_N, self.regulartheta
            )
            / (2.0 * np.pi)
        )
        to_integrate = (self.Jac_N * self.g_cont_zetazeta_N / g_tttt_N) * (
            dg_trtt_dtheta_N - dg_tttt_dr_N - (g_trtt_N * dJac_dtheta_N / self.Jac_N)
        )  # integrand of fourth term
        term4 = integrate.trapezoid(to_integrate, self.regulartheta) / (2.0 * np.pi)
        self.dB_zeta_dr_N = (B_zeta_N / H) * (
            term1 + term2 + term3 + term4
        )  # eq 3.139, dB_zeta_dr / B0
        return self.dB_zeta_dr_N

    @cached_property
    def get_dJac_dr(self):  # eq 3.137, uses 3.139. (dJac/dr) / a
        (
            g_trtr_N,
            g_trtt_N,
            g_tttt_N,
            g_zetazeta_N,
        ) = self.get_toroidal_system_covariant_metric_comps
        (
            dg_trtt_dtheta_N,
            dg_tttt_dr_N,
            dJac_dtheta_N,
        ) = self.get_toroidal_system_covariant_metric_derivatives
        B_zeta_N = self.get_B_cov_zeta
        dB_zeta_dr_N = self.dB_zeta_dr_N

        term1 = self.Jac_N * self.d2psidr2_N / self.dpsidr_N
        term2 = -(self.Jac_N / g_tttt_N) * (
            dg_trtt_dtheta_N - dg_tttt_dr_N - (g_trtt_N * dJac_dtheta_N / self.Jac_N)
        )
        term3 = (self.mu0dPdr_N / (self.dpsidr_N**2)) * (self.Jac_N**3) / g_tttt_N
        term4 = (
            (B_zeta_N * dB_zeta_dr_N / (self.dpsidr_N**2))
            * (self.Jac_N**3)
            * self.g_cont_zetazeta_N
            / g_tttt_N
        )
        self.dJac_dr_N = term1 + term2 + term3 + term4  # eq 3.137, (dJac/dr) / a
        return self.dJac_dr_N

    @cached_property
    def get_dalpha_dtheta(self):
        self.dalpha_dtheta_N = self.sigma_alpha * (
            self.q * self.Jac_N * self.g_cont_zetazeta_N / self.Y
        )  # eq D.92, already normalised
        return self.dalpha_dtheta_N

    @cached_property
    def get_d2alpha_drdtheta(
        self,
    ):  # eq D.93, sometimes known as 'local shear', a * d2alpha/drdtheta
        B_zeta_N = self.get_B_cov_zeta
        dB_zeta_dr_N = self.get_dB_cov_zeta_dr
        dJac_dr_N = self.get_dJac_dr

        term1 = dB_zeta_dr_N * self.Jac_N * self.g_cont_zetazeta_N / self.dpsidr_N
        term2 = (
            -self.d2psidr2_N
            * self.Jac_N
            * self.g_cont_zetazeta_N
            * B_zeta_N
            / (self.dpsidr_N**2)
        )
        term3 = B_zeta_N * dJac_dr_N * self.g_cont_zetazeta_N / self.dpsidr_N
        term4 = -(2.0 * self.dRdr_N / (self.R_N**3)) * (
            B_zeta_N * self.Jac_N / self.dpsidr_N
        )
        self.d2alpha_drdtheta_N = self.sigma_alpha * (term1 + term2 + term3 + term4)
        return self.d2alpha_drdtheta_N

    @cached_property
    def get_dalpha_dr(
        self,
    ):  # eq D.94, obtained by integrating D.93 over theta. Calculation in document
        # is bigger as the form of dJac/dr has been written explicitly.
        # a * dalpha/dr
        # inherets correct sigma_alpha from self.get_d2alpha_drdtheta
        d2alpha_drdtheta_N = self.get_d2alpha_drdtheta
        # integrate over theta
        dalpha_dr_N = integrate.cumulative_trapezoid(
            d2alpha_drdtheta_N, self.regulartheta
        )
        dalpha_dr_N = list(dalpha_dr_N)
        dalpha_dr_N.insert(0, 0.0)
        dalpha_dr_N = np.array(dalpha_dr_N)
        f = interp1d(self.regulartheta, dalpha_dr_N)
        self.dalpha_dr_N = dalpha_dr_N - f(
            0.0
        )  # set dalpha/dr(r,theta=0.0)=0.0, assumed by codes
        return self.dalpha_dr_N

    @cached_property
    def get_field_aligned_covariant_metric_components(self):
        dalpha_dr_N = self.get_dalpha_dr
        dalpha_dtheta_N = self.get_dalpha_dtheta
        (
            g_trtr_N,
            g_trtt_N,
            g_tttt_N,
            g_zetazeta_N,
        ) = self.get_toroidal_system_covariant_metric_comps

        self.g_frfr_N = (
            g_trtr_N + (dalpha_dr_N**2) * g_zetazeta_N
        )  # eq D.82, already normalised
        self.g_fralpha_N = (
            -dalpha_dr_N * g_zetazeta_N
        )  # eq D.83, inherets correct sigma_alpha from previous calculation g_fralpha / a
        self.g_frft_N = (
            g_trtt_N + dalpha_dr_N * dalpha_dtheta_N * g_zetazeta_N
        )  # eq D.84, g_frft / a
        self.g_alphaalpha_N = g_zetazeta_N  # eq D.85, g_alphaalpha / a^2
        self.g_alphaft_N = (
            -dalpha_dtheta_N * g_zetazeta_N
        )  # eq D.86, inherets correct sigma_alpha from previous calculation, g_alphaft / a^2
        self.g_ftft_N = (
            g_tttt_N + (dalpha_dtheta_N**2) * g_zetazeta_N
        )  # eq D.87, g_ftft / a^2
        return (
            self.g_frfr_N,
            self.g_fralpha_N,
            self.g_frft_N,
            self.g_alphaalpha_N,
            self.g_alphaft_N,
            self.g_ftft_N,
        )

    @cached_property
    def get_field_aligned_contravariant_metric_components(
        self,
    ):  # use covariant components and equation C.50 to get contravariant components g^{ij}, defined on page 196.
        # Some are simpler to obtain by dotting LHS's of equations D.79-D.81.
        (
            g_cont_trtr_N,
            g_cont_trtt_N,
            g_cont_tttt_N,
            g_cont_zetazeta_N,
        ) = self.get_toroidal_system_contravariant_metric_components
        (
            g_frfr_N,
            g_fralpha_N,
            g_frft_N,
            g_alphaalpha_N,
            g_alphaft_N,
            g_ftft_N,
        ) = self.get_field_aligned_covariant_metric_components
        dalpha_dtheta_N = self.get_dalpha_dtheta
        dalpha_dr_N = self.get_dalpha_dr

        self.g_cont_frfr_N = g_cont_trtr_N  # g^frfr, already normalised
        self.g_cont_frft_N = g_cont_trtt_N  # g^frft * a
        self.g_cont_ftft_N = g_cont_tttt_N  # g^ftft * a^2
        self.g_cont_fralpha_N = (
            dalpha_dr_N * g_cont_trtr_N + dalpha_dtheta_N * g_cont_trtt_N
        )  # g^fralpha * a
        self.g_cont_ftalpha_N = (
            dalpha_dr_N * g_cont_trtt_N + dalpha_dtheta_N * g_cont_tttt_N
        )  # g^ftalpha * a^2
        self.g_cont_alphaalpha_N = (g_frfr_N * g_ftft_N - (g_frft_N**2)) / (
            self.Jac_N**2
        )  # g^alphaalpha * a^2

        return (
            self.g_cont_frfr_N,
            self.g_cont_frft_N,
            self.g_cont_ftft_N,
            self.g_cont_fralpha_N,
            self.g_cont_ftalpha_N,
            self.g_cont_alphaalpha_N,
        )
