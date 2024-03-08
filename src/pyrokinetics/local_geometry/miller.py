from typing import Tuple

import numpy as np
from scipy.optimize import least_squares  # type: ignore

from ..typing import ArrayLike
from .local_geometry import LocalGeometry, default_inputs


def default_miller_inputs():
    """Default args to build a LocalGeometryMiller

    Uses a function call to avoid the user modifying these values
    """
    base_defaults = default_inputs()
    miller_defaults = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        "pressure": 1.0,
        "local_geometry": "Miller",
    }

    return {**base_defaults, **miller_defaults}


class LocalGeometryMiller(LocalGeometry):
    r"""Local equilibrium representation defined as in: `Phys Plasmas, Vol 5,
    No 4, April 1998 Miller et al <https://doi.org/10.1063/1.872666>`_

    .. math::
        \begin{align}
        R_s(r, \theta) &= R_0(r) + r \cos[\theta + (\sin^{-1}\delta) \sin(\theta)] \\
        Z_s(r, \theta) &= Z_0(r) + r \kappa \sin(\theta) \\
        r &= (\max(R) - \min(R)) / 2
        \end{align}

    Data stored in a CleverDict Object

    Attributes
    ----------
    psi_n : Float
        Normalised poloidal flux :math:`\psi_n=\psi_{surface}/\psi_{LCFS}`
    rho : Float
        Normalised minor radius :math:`\rho=r/a`
    r_minor : Float
        Minor radius of flux surface
    a_minor : Float
        Minor radius [m] :math:`a` (if 2D equilibrium exists)
    Rmaj : Float
        Normalised major radius :math:`R_{maj}/a`
    Z0 : Float
        Normalised vertical position of midpoint (Zmid / a_minor)
    f_psi : Float
        Torodial field function
    B0 : Float
        Normalising magnetic field :math:`B_0 = f / R_{maj}` [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field to B0 :math:`B_{unit} =
        \frac{q}{r}\frac{\partial\psi}{\partial r}` [T]
    dpsidr : Float
        :math:`\frac{\partial \psi}{\partial r}`
    q : Float
        Safety factor :math:`q`
    shat : Float
        Magnetic shear :math:`\hat{s} = \frac{\rho}{q}\frac{\partial q}{\partial\rho}`
    beta_prime : Float
        Pressure gradient :math:`\beta'=\frac{8\pi 10^{-7}}{B_0^2}
        \frac{\partial p}{\partial\rho}`
    kappa : Float
        Elongation :math:`\kappa`
    delta : Float
        Triangularity :math:`\delta`
    s_kappa : Float
        Shear in Elongation :math:`\frac{\rho}{\kappa}
        \frac{\partial\kappa}{\partial\rho}`
    s_delta : Float
        Shear in Triangularity :math:`\frac{\rho}{\sqrt{1-\delta^2}}
        \frac{\partial\delta}{\partial\rho}`
    shift : Float
        Shafranov shift :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`
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
            and not isinstance(args[0], LocalGeometryMiller)
            and isinstance(args[0], dict)
        ):
            super().__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates Miller shaping coefficients from R, Z and b_poloidal

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

        kappa = (max(Z) - min(Z)) / (2 * self.r_minor)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = self.Rmaj / self.rho - R_upper / self.r_minor

        normalised_height = (Z - Zmid) / (kappa * self.r_minor)

        self.kappa = kappa
        self.delta = delta
        self.Zmid = Zmid

        self.Z0 = float(Zmid / self.a_minor)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        self.theta = theta
        self.theta_eq = theta

        self.R, self.Z = self.get_flux_surface(theta=self.theta, normalised=True)

        s_kappa_fit = 0.0
        s_delta_fit = 0.0
        shift_fit = shift
        dZ0dr_fit = 0.0

        params = [
            s_kappa_fit,
            s_delta_fit,
            shift_fit,
            dZ0dr_fit,
        ]

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in Miller::from_global_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"Miller :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in Miller::from_global_eq is poor with residual of {fits.cost}"
            )

        self.s_kappa = fits.x[0]
        self.s_delta = fits.x[1]
        self.shift = fits.x[2]
        self.dZ0dr = fits.x[3]

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generates :math:`(R,Z)` of a flux surface given a set of Miller fits

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
        Z = self.Z0 + self.kappa * self.rho * np.sin(theta)

        if not normalised:
            R *= self.a_minor
            Z *= self.a_minor

        return R, Z

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
        normalised=False,
    ) -> np.ndarray:
        r"""Calculates the derivatives of :math:`R(r, \theta)` and :math:`Z(r,
        \theta)` w.r.t :math:`r` and :math:`\theta`, used in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use ``params = [s_kappa_fit,
            s_delta_fit, shift_fit, dZ0dr_fit]`` when calculating
            derivatives, otherwise will use object attributes
        normalised : bool
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
            shift = params[2]
            dZ0dr = params[3]
        else:
            s_kappa = self.s_kappa
            s_delta = self.s_delta
            shift = self.shift
            dZ0dr = self.dZ0dr

        dZdtheta = self.get_dZdtheta(theta, normalised)
        dZdr = self.get_dZdr(theta, dZ0dr, s_kappa)
        dRdtheta = self.get_dRdtheta(theta, normalised)
        dRdr = self.get_dRdr(theta, shift, s_delta)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
        normalised=False,
    ) -> np.ndarray:
        r"""
        Calculates the second derivatives of :math:`R(r, \theta)` and :math:`Z(r,
        \theta)` w.r.t :math:`r` and :math:`\theta`, used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        normalised : Boolean
            Control whether or not to return normalised values

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

        d2Zdtheta2 = self.get_d2Zdtheta2(theta, normalised)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.s_kappa)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta, normalised)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.s_delta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_dZdtheta(self, theta, normalised=False):
        r"""Calculates the derivatives of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`

        """

        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor

        return self.kappa * rmin * np.cos(theta)

    def get_d2Zdtheta2(self, theta, normalised=False):
        r"""Calculates the second derivative of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdtheta2 on
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of :math:`Z` w.r.t :math:`\theta`

        """

        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor

        return self.kappa * rmin * -np.sin(theta)

    def get_dZdr(self, theta, dZ0dr, s_kappa):
        r"""Calculates the derivatives of :math:`Z(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Shear in midplane elevation
        s_kappa : Float
            Shear in Elongation
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`
        """

        return dZ0dr + self.kappa * np.sin(theta) + s_kappa * self.kappa * np.sin(theta)

    def get_d2Zdrdtheta(self, theta, s_kappa):
        r"""Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdrdtheta on
        s_kappa : Float
            Shear in Elongation
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """

        return np.cos(theta) * (self.kappa + s_kappa * self.kappa)

    def get_dRdtheta(self, theta, normalised=False):
        r"""Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        """
        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor
        x = np.arcsin(self.delta)

        return -rmin * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    def get_d2Rdtheta2(self, theta, normalised=False):
        r"""Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdtheta2 on
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        d2Rdtheta2 : Array
            Second derivative of :math:`R` w.r.t :math:`\theta`
        """
        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor
        x = np.arcsin(self.delta)

        return -rmin * (
            ((1 + x * np.cos(theta)) ** 2) * np.cos(theta + x * np.sin(theta))
            - x * np.sin(theta) * np.sin(theta + x * np.sin(theta))
        )

    def get_dRdr(self, theta, shift, s_delta):
        r"""Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdr on
        shift : Float
            Shafranov shift
        s_delta : Float
            Shear in Triangularity
        normalised : Boolean
            Control whether or not to return normalised values

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
        r"""Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdrdtheta on
        s_delta : Float
            Shear in Triangularity
        normalised : Boolean
            Control whether or not to return normalised values

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

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMiller, self).__init__(default_miller_inputs())
