import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
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
    Miller Object representing local Miller fit parameters

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
    Rgeo : Float
        Normalisd major radius of normalising field (Rreference/a)
    Z0 : Float
        Normalised vertical position of midpoint (Zmid / a_minor)
    f_psi : Float
        Torodial field function
    B0 : Float
        Toroidal field at major radius (f_psi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    kappa : Float
        Elongation
    delta : Float
        Triangularity
    s_kappa : Float
        Shear in Elongation
    s_delta : Float
        Shear in Triangularity
    shift : Float
        Shafranov shift
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear
    beta_prime : Float
        :math:`\beta' = \beta * a/L_p`

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

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        miller = cls()
        miller.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return miller

    def load_from_eq(
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

        """

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().load_from_eq(
            eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, show_fit=show_fit
        )

    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):

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
                f"Least squares fitting in Miller::load_from_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"Miller :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in Miller::load_from_eq is poor with residual of {fits.cost}"
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
        Generates (R,Z) of a flux surface given a set of Miller fits

        Parameters
        ----------
        kappa : Float
            Elongation
        delta : Float
            Triangularity
        Rcen : Float
            Major radius of flux surface [m]
        rmin : Float
            Minor radius of flux surface [m]
        Zmid : Float
            Vertical midpoint of flux surface [m]
        theta : Array
            Values of theta to evaluate flux surface

        Returns
        -------
        R : Array
            R values for this flux surface [m]
        Z : Array
            Z Values for this flux surface [m]
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
        Miller definition of grad r from
        Miller, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Parameters
        ----------
        kappa: Scalar
            Miller elongation
        delta: Scalar
            Miller triangularity
        s_kappa: Scalar
            Radial derivative of Miller elongation
        s_delta: Scalar
            Radial derivative of Miller triangularity
        shift: Scalar
            Shafranov shift
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        grad_r : Array
            grad_r(theta)
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

        if normalised:
            rmin = self.rho
        else:
            rmin = self.r_minor
        x = np.arcsin(self.delta)

        return -rmin * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    def get_dRdr(self, theta, shift, s_delta):

        x = np.arcsin(self.delta)

        return (
            shift
            + np.cos(theta + x * np.sin(theta))
            - np.sin(theta + x * np.sin(theta)) * np.sin(theta) * s_delta
        )

    def _get_theta_from_squareness(self, theta):

        fits = least_squares(self._minimise_theta_from_squareness, theta)

        return fits.x

    def _minimise_theta_from_squareness(self, params):

        theta = params
        theta_func = np.arcsin((self.Z_eq - self.Zmid) / (self.kappa * self.r_minor))

        sum_diff = np.sum(np.abs(theta_func - theta - self.zeta * np.sin(2 * theta)))
        return sum_diff

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMiller, self).__init__(default_miller_inputs())
