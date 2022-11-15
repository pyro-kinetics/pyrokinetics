import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
from .LocalGeometry import default_inputs


def default_fourier_inputs(n_moments=32):
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    fourier_defaults = {
        "dZ0dr": 0.0,
        "shift": 0.0,
        "cN": np.array([0.5, *[0.0] * (n_moments - 1)]),
        "sN": np.zeros(n_moments),
        "dcNdr": np.array([1.0, *[0.0] * (n_moments - 1)]),
        "dsNdr": np.zeros(n_moments),
        "local_geometry": "Fourier",
    }

    return {**base_defaults, **fourier_defaults}


class LocalGeometryFourier(LocalGeometry):
    r"""
    Fourier Object representing local fourier fit parameters
    Uses method in Plasma Phys. Control. Fusion 63 (2021) 012001 (5pp)
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
            and not isinstance(args[0], LocalGeometryFourier)
            and isinstance(args[0], dict)
        ):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    @classmethod
    def from_gk_data(cls, params: Dict[str, Any]):
        """
        Initialise from data gathered from GKCode object, and additionally set
        bunit_over_b0
        """
        # TODO change __init__ to take necessary parameters by name. It shouldn't
        # be possible to have a fourier object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        fourier = cls(params)
        fourier.bunit_over_b0 = fourier.get_bunit_over_b0()
        return fourier

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        fourier = cls()
        fourier.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return fourier

    def load_from_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=32, show_fit=False
    ):
        r"""
        Loads fourier object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local fourier parameters
        verbose : Boolean
            Controls verbosity

        """

        self.n_moments = n_moments

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().load_from_eq(
            eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, show_fit=False
        )

    def load_from_local_geometry(
        self, local_geometry: LocalGeometry, verbose=False, n_moments=32, show_fit=False
    ):
        r"""
        Loads mxh object from a LocalGeometry Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        local_geometry : LocalGeometry
            LocalGeometry object
        verbose : Boolean
            Controls verbosity
        n_moments: Int
            Number of moments to fit with

        """

        self.n_moments = n_moments

        super().load_from_local_geometry(
            local_geometry=local_geometry, verbose=verbose, show_fit=show_fit
        )

    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""

        Parameters
        ----------
        R
        Z
        b_poloidal
        verbose

        Returns
        -------

        """

        R_major = self.Rmaj * self.a_minor
        Zmid = self.Z0 * self.a_minor

        R_diff = R - R_major
        Z_diff = Z - Zmid

        dot_product = R_diff * np.roll(R_diff, 1) + Z_diff * np.roll(Z_diff, 1)
        magnitude = np.sqrt(R_diff**2 + Z_diff**2)
        arc_angle = dot_product / (magnitude * np.roll(magnitude, 1))

        theta_diff = np.arccos(arc_angle)

        if Z[1] > Z[0]:
            theta = np.cumsum(theta_diff) - theta_diff[0]
        else:
            theta = -np.cumsum(theta_diff) - theta_diff[0]

        self.theta_eq = theta

        # Interpolate to evenly spaced theta
        theta_new = np.linspace(0, 2 * np.pi, len(theta))
        R = np.interp(theta_new, theta, R)
        Z = np.interp(theta_new, theta, Z)
        b_poloidal = np.interp(theta_new, theta, b_poloidal)
        theta = theta_new

        aN = np.sqrt((R - R_major) ** 2 + (Z - Zmid) ** 2) / self.a_minor

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        cN = simpson(aN[None, :] * np.cos(ntheta), theta, axis=1) / np.pi
        sN = simpson(aN[None, :] * np.sin(ntheta), theta, axis=1) / np.pi

        cN[0] *= 0.5
        sN[0] *= 0.5

        self.cN = cN
        self.sN = sN

        self.theta = theta
        self.R, self.Z = self.get_flux_surface(theta)

        # Need evenly spaced bpol to fit to
        self.b_poloidal_even_space = b_poloidal

        dZ0dr = 0.0
        params = [shift, dZ0dr, 1.0, *[0.0] * (self.n_moments * 2 - 1)]

        fits = least_squares(
            self.minimise_b_poloidal, params, kwargs={"even_space_theta": "True"}
        )

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in Fourier::load_from_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"Fourier :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in Fourier::load_from_eq is poor with residual of {fits.cost}"
            )

        self.shift = fits.x[0]
        self.dZ0dr = fits.x[1]
        self.dcNdr = fits.x[2 : self.n_moments + 2]
        self.dsNdr = fits.x[self.n_moments + 2 :]

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta,
        )

    def get_grad_r(
        self,
        theta: ArrayLike,
        params=None,
        normalised=False,
    ) -> np.ndarray:
        """
        fourier definition of grad r from
        fourier, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Parameters
        ----------
        kappa: Scalar
            fourier elongation
        shift: Scalar
            Shafranov shift
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        grad_r : Array
            grad_r(theta)
        """

        if params is None:
            shift = self.shift
            dZ0dr = self.dZ0dr
            dcNdr = self.dcNdr
            dsNdr = self.dsNdr
        else:
            shift = params[0]
            dZ0dr = params[1]
            dcNdr = params[2 : self.n_moments + 2]
            dsNdr = params[self.n_moments + 2 :]

        n_moments = len(self.cN)
        n = np.linspace(0, n_moments - 1, n_moments)
        ntheta = n[:, None] * theta[None, :]

        aN = np.sum(
            self.cN[:, None] * np.cos(ntheta) + self.sN[:, None] * np.sin(ntheta),
            axis=0,
        )
        daNdr = np.sum(
            dcNdr[:, None] * np.cos(ntheta) + dsNdr[:, None] * np.sin(ntheta), axis=0
        )
        daNdtheta = np.sum(
            -self.cN[:, None] * n[:, None] * np.sin(ntheta)
            + self.sN[:, None] * n[:, None] * np.cos(ntheta),
            axis=0,
        )

        dZdtheta = aN * np.cos(theta) + daNdtheta * np.sin(theta)

        dZdr = dZ0dr + daNdr * np.sin(theta)

        dRdtheta = -aN * np.sin(theta) + daNdtheta * np.cos(theta)

        dRdr = shift + daNdr * np.cos(theta)

        g_tt = dRdtheta**2 + dZdtheta**2

        grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

        return grad_r

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of fourier fits

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
        thetaR : Array
            Values of thetaR to evaluate flux surface

        Returns
        -------
        R : Array
            R values for this flux surface [m]
        Z : Array
            Z Values for this flux surface [m]
        """
        n_moments = len(self.cN)
        n = np.linspace(0, n_moments - 1, n_moments)

        ntheta = n[:, None] * theta[None, :]
        aN = np.sum(
            self.cN[:, None] * np.cos(ntheta) + self.sN[:, None] * np.sin(ntheta),
            axis=0,
        )

        R = self.Rmaj + aN * np.cos(theta)
        Z = self.Z0 + aN * np.sin(theta)

        if not normalised:
            R *= self.a_minor
            Z *= self.a_minor

        return R, Z

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryFourier, self).__init__(default_fourier_inputs())
