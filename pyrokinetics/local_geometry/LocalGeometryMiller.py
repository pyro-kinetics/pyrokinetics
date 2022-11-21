import numpy as np
from typing import Tuple
from scipy.optimize import least_squares  # type: ignore
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import ArrayLike
from .LocalGeometry import default_inputs


def default_miller_inputs():
    # Return default args to build a LocalGeometryMiller
    # Uses a function call to avoid the user modifying these values

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
        "pressure": 1.0,
        "local_geometry": "Miller",
    }

    return {**base_defaults, **miller_defaults}


class LocalGeometryMiller(LocalGeometry):
    r"""
    Local equilibrium representation defined as in:
    Phys. Plasmas, Vol. 5, No. 4, April 1998 Miller et al.
    Physics of Plasmas 6, 1113 (1999); Turnbull et al ;  https://doi.org/10.1063/1.873380
    Miller

    R(r, theta) = Rmajor(r) + r * cos(theta + arcsin(delta(r)) * sin(theta)
    Z(r, theta) = Z0(r) + r * kappa(r) * sin(theta + zeta(r) * sin(2*theta)

    r = (max(R) - min(R)) / 2

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
        Toroidal field at major radius (f_psi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear `r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta' = `2 \mu_0 \partial p \partial \rho 1/B0^2`

    kappa : Float
        Elongation
    delta : Float
        Triangularity
    zeta : Float
        Squareness
    s_kappa : Float
        Shear in Elongation
    s_delta : Float
        Shear in Triangularity
    s_zeta : Float
        Shear in Squareness
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
        Derivative of fitted `R` w.r.t `\theta`
    dRdr : Array
        Derivative of fitted `R` w.r.t `r`
    dZdtheta : Array
        Derivative of fitted `Z` w.r.t `\theta`
    dZdr : Array
        Derivative of fitted `Z` w.r.t `r`
    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if (
            args
            and not isinstance(args[0], LocalGeometryMiller)
            and isinstance(args[0], dict)
        ):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def from_global_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, show_fit=False
    ):
        r"""
        Loads Miller object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local Miller parameters
        verbose : Boolean
            Controls verbosity
        show_fit : Boolean
            Controls whether fit vs equilibrium is plotted
        """

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().from_global_eq(
            eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, show_fit=show_fit
        )

    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates Miller shaping coefficients from R, Z and b_poloidal

        Parameters
        ----------
        R : Array
            R for the given flux surface
        Z : Array
            Z for the given flux surface
        b_poloidal : Array
            `b_\theta` for the given flux surface
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

        R_pi4 = (
            self.Rmaj + self.rho * np.cos(pi / 4 + np.arcsin(delta) * np.sin(pi / 4))
        ) * self.a_minor

        R_gt_0 = np.where(Z > 0, R, 0.0)
        Z_pi4 = Z[np.argmin(np.abs(R_gt_0 - R_pi4))]

        zeta = np.arcsin((Z_pi4 - Zmid) / (kappa * self.r_minor)) - pi / 4

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

        self.R, self.Z = self.get_flux_surface(theta=self.theta, normalised=True)

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
        self.s_zeta = fits.x[2]
        self.shift = fits.x[3]
        self.dZ0dr = fits.x[4]

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates `(R,Z)` of a flux surface given a set of Miller fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface
        normalised : Boolean
            Control whether or not to return normalised flux surface
        Returns
        -------
        `R` : Array
            `R(\theta)` values for this flux surface (if not normalised then in [m])
        `Z` : Array
            `Z(\theta)` Values for this flux surface (if not normalised then in [m])
        """

        R = self.Rmaj + self.rho * np.cos(theta + np.arcsin(self.delta) * np.sin(theta))
        Z = self.Z0 + self.kappa * self.rho * np.sin(
            theta + self.zeta * np.sin(2 * theta)
        )

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
        """
        Calculates the derivatives of `R(r, \theta)` and `Z(r, \theta)` w.r.t `r` and `\theta`, used in B_poloidal calc

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
            Derivative of `R` w.r.t `\theta`
        dRdr : Array
            Derivative of `R` w.r.t `r`
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        dZdr : Array
            Derivative of `Z` w.r.t `r`
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

        dZdtheta = self.get_dZdtheta(theta, normalised)
        dZdr = self.get_dZdr(theta, dZ0dr, s_kappa, s_zeta, normalised)
        dRdtheta = self.get_dRdtheta(theta, normalised)
        dRdr = self.get_dRdr(theta, shift, s_delta)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_dZdtheta(self, theta, normalised=False):
        """
        Calculates the derivatives of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        """

        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor

        return (
            self.kappa
            * rmin
            * (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.cos(theta + self.zeta * np.sin(2 * theta))
        )

    def get_dZdr(self, theta, dZ0dr, s_kappa, s_zeta, normalised=False):
        """
        Calculates the derivatives of `Z(r, \theta)` w.r.t `r`

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
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """
        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor

        return (
            dZ0dr
            + self.kappa * np.sin(theta + self.zeta * np.sin(2 * theta))
            + s_kappa * self.kappa * np.sin(theta + self.zeta * np.sin(2 * theta))
            + self.kappa
            * rmin
            * s_zeta
            * np.sin(2 * theta)
            * np.cos(theta + self.zeta * np.sin(2 * theta))
        )

    def get_dRdtheta(self, theta, normalised=False):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        dRdtheta : Array
            Derivative of `R` w.r.t `\theta`
        """
        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor
        x = np.arcsin(self.delta)

        return -rmin * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    def get_dRdr(self, theta, shift, s_delta):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `r`

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
            Derivative of `R` w.r.t `r`
        """
        x = np.arcsin(self.delta)

        return (
            shift
            + np.cos(theta + x * np.sin(theta))
            - np.sin(theta + x * np.sin(theta)) * np.sin(theta) * s_delta
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
        fits = least_squares(self._minimise_theta_from_squareness, theta)

        return fits.x

    def _minimise_theta_from_squareness(self, theta):
        """
        Calculate theta in Miller by re-arranging equation for Z and changing theta such that the function gets
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
        theta_func = np.arcsin((self.Z_eq - self.Zmid) / (self.kappa * self.r_minor))

        sum_diff = np.sum(np.abs(theta_func - theta - self.zeta * np.sin(2 * theta)))
        return sum_diff

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMiller, self).__init__(default_miller_inputs())
