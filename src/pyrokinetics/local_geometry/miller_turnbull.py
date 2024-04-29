from typing import Tuple

import numpy as np
from scipy.optimize import least_squares  # type: ignore

from ..constants import pi
from ..typing import ArrayLike
from ..units import ureg as units
from .local_geometry import LocalGeometry, default_inputs


def default_miller_turnbull_inputs():
    """Default args to build a LocalGeometryMillerTurnbull

    Uses a function call to avoid the user modifying these values
    """

    base_defaults = default_inputs()
    miller_defaults = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        "local_geometry": "MillerTurnbull",
    }

    return {**base_defaults, **miller_defaults}


class LocalGeometryMillerTurnbull(LocalGeometry):
    r"""Local equilibrium representation defined as in:

    - `Phys. Plasmas, Vol. 5, No. 4, April 1998 Miller et al
      <https://doi.org/10.1063/1.872666>`_

    - `Physics of Plasmas 6, 1113 (1999); Turnbull et al
      <https://doi.org/10.1063/1.873380>`_

    .. math::
        \begin{align}
        R(r, \theta) &= R_{major}(r) + r \cos(\theta + \arcsin(\delta(r) \sin(\theta)) \\
        Z(r, \theta) &= Z_0(r) + r \kappa(r) \sin(\theta + \zeta(r) \sin(2\theta)) \\
        r &= (\max(R) - \min(R)) / 2
        \end{align}

    Data stored in a CleverDict Object

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
        Toroidal field at major radius (Fpsi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math:`\frac{\partial \psi}{\partial r}`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear :math:`r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta = 2 \mu_0 \partial p \partial \rho 1/B0^2`

    kappa : Float
        Elongation
    delta : Float
        Triangularity
    zeta : Float
        Squareness
    s_kappa : Float
        Shear in Elongation :math:`r/\kappa \partial \kappa/\partial r`
    s_delta : Float
        Shear in Triangularity :math:`r/\sqrt{1 - \delta^2} \partial \delta/\partial r`
    s_zeta : Float
        Shear in Squareness :math:`r/ \partial \zeta/\partial r`
    shift : Float
        Shafranov shift
    dZ0dr : Float
        Shear in midplane elevation

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
        Derivative of fitted :math:`R` w.r.t :math:`\theta`
    dRdr : Array
        Derivative of fitted :math:`R` w.r.t :math:`r`
    dZdtheta : Array
        Derivative of fitted :math:`Z` w.r.t :math:`\theta`
    dZdr : Array
        Derivative of fitted :math:`Z` w.r.t :math:`r`

    d2Rdtheta2 : Array
        Second derivative of fitted :math:`R` w.r.t :math:`\theta`
    d2Rdrdtheta : Array
        Derivative of fitted :math:`R` w.r.t :math:`r` and :math:`\theta`
    d2Zdtheta2 : Array
        Second derivative of fitted :math:`Z` w.r.t :math:`\theta`
    d2Zdrdtheta : Array
        Derivative of fitted :math:`Z` w.r.t :math:`r` and :math:`\theta`

    """

    def __init__(self, *args, **kwargs):
        s_args = list(args)

        if (
            args
            and not isinstance(args[0], LocalGeometryMillerTurnbull)
            and isinstance(args[0], dict)
        ):
            super().__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates MillerTurnbull shaping coefficients from R, Z and b_poloidal

        Parameters
        ----------
        R : Array
            R for the given flux surface
        Z : Array
            Z for the given flux surface
        b_poloidal : Array
            :math:`b_\theta` for the given flux surface
        verbose : Boolean
            Controls verbosity
        shift : Float
            Initial guess for shafranov shift
        """

        kappa = (max(Z) - min(Z)) / (2 * self.rho)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = self.Rmaj / self.rho - R_upper / self.rho

        normalised_height = (Z - Zmid) / (kappa * self.rho)

        self.kappa = kappa
        self.delta = delta
        self.Z0 = Zmid

        R_pi4 = self.Rmaj + self.rho * np.cos(
            pi / 4 + np.arcsin(delta) * np.sin(pi / 4)
        )

        R_gt_0 = np.where(Z > 0, R, 0.0)
        Z_pi4 = Z[np.argmin(np.abs(R_gt_0 - R_pi4))]

        zeta = np.arcsin((Z_pi4 - Zmid) / (kappa * self.rho)) - pi / 4

        self.zeta = zeta

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta_guess = np.arcsin(normalised_height)
        theta = self._get_theta_from_squareness(theta_guess)

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        self.theta = theta
        self.theta_eq = theta

        self.R, self.Z = self.get_flux_surface(theta=self.theta)

        s_kappa_fit = 0.0
        s_delta_fit = 0.0
        s_zeta_fit = 0.0
        shift_fit = shift
        dZ0dr_fit = 0.0

        params = [
            s_kappa_fit,
            s_delta_fit,
            s_zeta_fit,
            shift_fit,
            dZ0dr_fit,
        ]

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in MillerTurnbull::_set_shape_coefficients failed with message : {fits.message}"
            )

        if verbose:
            print(
                f"MillerTurnbull :: Fit to Bpoloidal obtained with residual {fits.cost}"
            )

        if fits.cost > 1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in MillerTurnbull::_set_shape_coefficients is poor with residual of {fits.cost}"
            )

        self.s_kappa = fits.x[0] * units.dimensionless
        self.s_delta = fits.x[1] * units.dimensionless
        self.s_zeta = fits.x[2] * units.dimensionless
        self.shift = fits.x[3] * units.dimensionless
        self.dZ0dr = fits.x[4] * units.dimensionless

    def get_flux_surface(
        self,
        theta: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generates :math:`(R,Z)` of a flux surface given a set of MillerTurnbull fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface
        normalised : Boolean
            Control whether or not to return normalised flux surface

        Returns
        -------
        :math:`R` : Array
            :math:`R(\theta)` values for this flux surface (if not normalised then in [m])
        :math:`Z` : Array
            :math:`Z(\theta)` Values for this flux surface (if not normalised then in [m])
        """

        R = self.Rmaj + self.rho * np.cos(theta + np.arcsin(self.delta) * np.sin(theta))
        Z = self.Z0 + self.kappa * self.rho * np.sin(
            theta + self.zeta * np.sin(2 * theta)
        )

        return R, Z

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        r"""Calculates the derivatives of :math:`R(r, \theta)` and
        :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`, used
        in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use params = [s_kappa_fit,s_delta_fit,s_zeta_fit, shift_fit,dZ0dr_fit] when calculating
            derivatives, otherwise will use object attributes
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        dRdr : Array
            Derivative of :math:`R` w.r.t :math:`r`
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`

        """

        if params is not None:
            s_kappa = params[0]
            s_delta = params[1]
            s_zeta = params[2]
            shift = params[3]
            dZ0dr = params[4]
        else:
            s_kappa = self.s_kappa
            s_delta = self.s_delta
            s_zeta = self.s_zeta
            shift = self.shift
            dZ0dr = self.dZ0dr

        dZdtheta = self.get_dZdtheta(theta)
        dZdr = self.get_dZdr(theta, dZ0dr, s_kappa, s_zeta)
        dRdtheta = self.get_dRdtheta(theta)
        dRdr = self.get_dRdr(theta, shift, s_delta)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
    ) -> np.ndarray:
        r"""Calculates the second derivatives of :math:`R(r, \theta)`
        and :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`,
        used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        d2Rdtheta2 : Array
                        Second derivative of :math:`R` w.r.t :math:`\theta`
        d2Rdrdtheta : Array
                        Second derivative of :math:`R` w.r.t :math:`r` and :math:`\theta`
        d2Zdtheta2 : Array
                        Second derivative of :math:`Z` w.r.t :math:`\theta`
        d2Zdrdtheta : Array
                        Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`

        """

        d2Zdtheta2 = self.get_d2Zdtheta2(theta)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.s_kappa, self.s_zeta)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.s_delta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_dZdtheta(self, theta):
        r"""
        Calculates the derivatives of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on

        Returns
        -------
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`
        """

        return (
            self.kappa
            * self.rho
            * (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.cos(theta + self.zeta * np.sin(2 * theta))
        )

    def get_d2Zdtheta2(self, theta):
        r"""
        Calculates the second derivative of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdtheta2 on

        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of :math:`Z` w.r.t :math:`\theta`
        """

        return (
            self.kappa
            * self.rho
            * (
                -4
                * self.zeta
                * np.sin(2 * theta)
                * np.cos(theta + self.zeta * np.sin(2 * theta))
                - ((1 + 2 * self.zeta * np.cos(2 * theta)) ** 2)
                * np.sin(theta + self.zeta * np.sin(2 * theta))
            )
        )

    def get_dZdr(self, theta, dZ0dr, s_kappa, s_zeta):
        r"""
        Calculates the derivatives of :math:`Z(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Shear in midplane elevation
        s_kappa : Float
            Shear in Elongation
        s_zeta : Float
            Shear in Squareness

        Returns
        -------
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`
        """

        return (
            dZ0dr
            + self.kappa * np.sin(theta + self.zeta * np.sin(2 * theta))
            + s_kappa * self.kappa * np.sin(theta + self.zeta * np.sin(2 * theta))
            + self.kappa
            * s_zeta
            * np.sin(2 * theta)
            * np.cos(theta + self.zeta * np.sin(2 * theta))
        )

    def get_d2Zdrdtheta(self, theta, s_kappa, s_zeta):
        r"""
        Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdrdtheta on
        s_kappa : Float
            Shear in Elongation
        s_zeta : Float
            Shear in Squareness

        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """

        return (
            (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.cos(theta + self.zeta * np.sin(2 * theta))
            * (s_kappa * self.kappa + self.kappa)
            + self.kappa
            * np.cos(theta + self.zeta * np.sin(2 * theta))
            * 2
            * np.cos(2 * theta)
            * s_zeta
            - self.kappa
            * s_zeta
            * np.sin(2 * theta)
            * (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.sin(theta + self.zeta * np.sin(2 * theta))
        )

    def get_dRdtheta(self, theta):
        r"""
        Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        """
        x = np.arcsin(self.delta)

        return -self.rho * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    def get_d2Rdtheta2(self, theta):
        r"""
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdtheta2 on

        Returns
        -------
        d2Rdtheta2 : Array
            Second derivative of :math:`R` w.r.t :math:`\theta`
        """
        x = np.arcsin(self.delta)

        return -self.rho * (
            ((1 + x * np.cos(theta)) ** 2) * np.cos(theta + x * np.sin(theta))
            - x * np.sin(theta) * np.sin(theta + x * np.sin(theta))
        )

    def get_dRdr(self, theta, shift, s_delta):
        r"""
        Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdr on
        shift : Float
            Shafranov shift
        s_delta : Float
            Shear in Triangularity

        Returns
        -------
        dRdr : Array
            Derivative of :math:`R` w.r.t :math:`r`
        """
        x = np.arcsin(self.delta)

        return (
            shift
            + np.cos(theta + x * np.sin(theta))
            - np.sin(theta + x * np.sin(theta)) * np.sin(theta) * s_delta
        )

    def get_d2Rdrdtheta(self, theta, s_delta):
        r"""
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdrdtheta on
        s_delta : Float
            Shear in Triangularity

        Returns
        -------
        d2Rdrdtheta : Array
            Second derivative of :math:`R` w.r.t :math:`r` and :math:`\theta`
        """
        x = np.arcsin(self.delta)

        return (
            -(1 + x * np.cos(theta)) * np.sin(theta + x * np.sin(theta))
            - s_delta * np.cos(theta) * np.sin(theta + x * np.sin(theta))
            - s_delta
            * np.sin(theta)
            * (1 + x * np.cos(theta))
            * np.cos(theta + x * np.sin(theta))
        )

    def _get_theta_from_squareness(self, theta):
        """
        Performs least square fitting to get theta for a given flux surface from the equation for Z

        Parameters
        ----------
        theta

        Returns
        -------

        """
        fits = least_squares(self._minimise_theta_from_squareness, theta.m)

        return fits.x * theta.units

    def _minimise_theta_from_squareness(self, theta):
        """
        Calculate theta in MillerTurnbull by re-arranging equation for Z and changing theta such that the function gets
        minimised
        Parameters
        ----------
        theta : Array
            Guess for theta
        Returns
        -------
        sum_diff : Array
            Minimisation difference
        """
        normalised_height = (self.Z_eq - self.Z0) / (self.kappa * self.rho)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta_func = np.arcsin(normalised_height)
        sum_diff = np.sum(np.abs(theta_func - theta - self.zeta * np.sin(2 * theta)))
        return sum_diff

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMillerTurnbull, self).__init__(
            default_miller_turnbull_inputs()
        )

    def _generate_shape_coefficients_units(self, norms):
        """
        Units for MillerTurnbull parameters
        """

        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "delta": units.dimensionless,
            "s_delta": units.dimensionless,
            "zeta": units.dimensionless,
            "s_zeta": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
        }
