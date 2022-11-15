import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
from .LocalGeometry import default_inputs


def default_mxh_inputs(n_moments=4):
    # Return default args to build a LocalGeometryMXH
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    mxh_defaults = {
        "asym_coeff": np.zeros(n_moments),
        "dasym_dr": np.zeros(n_moments),
        "sym_coeff": np.zeros(n_moments),
        "dsym_dr": np.zeros(n_moments),
        "local_geometry": "MXH",
    }

    return {**base_defaults, **mxh_defaults}


class LocalGeometryMXH(LocalGeometry):
    r"""
    MXH Object representing local mxh fit parameters
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
            and not isinstance(args[0], LocalGeometryMXH)
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
        # be possible to have a mxh object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        mxh = cls(params)
        mxh.bunit_over_b0 = mxh.get_bunit_over_b0()
        return mxh

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        mxh = cls()
        mxh.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return mxh

    def load_from_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=4, show_fit=False
    ):
        r"""
        Loads mxh object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local mxh parameters
        verbose : Boolean
            Controls verbosity

        """

        self.n_moments = n_moments

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().load_from_eq(
            eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, show_fit=show_fit
        )

    def load_from_local_geometry(
        self, local_geometry: LocalGeometry, verbose=False, n_moments=4, show_fit=False
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

        kappa = (max(Z) - min(Z)) / (2 * self.r_minor)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        normalised_height = (Z - Zmid) / (kappa * self.r_minor)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        normalised_radius = (R - self.Rmaj * self.a_minor) / self.r_minor

        normalised_radius = np.where(
            np.isclose(normalised_radius, 1.0), 1.0, normalised_radius
        )
        normalised_radius = np.where(
            np.isclose(normalised_radius, -1.0), -1.0, normalised_radius
        )

        thetaR = np.arccos(normalised_radius)

        theta = np.where(R < R_upper, np.pi - theta, theta)
        theta = np.where((R >= R_upper) & (Z < 0), 2 * np.pi + theta, theta)
        thetaR = np.where(Z < 0, 2 * np.pi - thetaR, thetaR)

        self.theta_eq = theta

        # Ensure theta start from zero
        theta = np.roll(theta, -np.argmin(theta))
        thetaR = np.roll(thetaR, -np.argmin(thetaR))

        theta_diff = thetaR - theta

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        asym_coeff = (
            simpson(theta_diff[None, :] * np.cos(ntheta), theta, axis=1) / np.pi
        )
        sym_coeff = simpson(theta_diff[None, :] * np.sin(ntheta), theta, axis=1) / np.pi

        self.kappa = kappa
        self.sym_coeff = sym_coeff
        self.asym_coeff = asym_coeff

        self.theta = theta
        self.thetaR = self.get_thetaR(self.theta)
        self.dthetaR_dtheta = self.get_dthetaR_dtheta(self.theta)

        self.R, self.Z = self.get_flux_surface(self.theta)

        dkap_dr_init = 0.0
        params = [shift, 0.0, dkap_dr_init, *[0.0] * self.n_moments * 2]

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in MXH::load_from_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"MXH :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in MXH::load_from_eq is poor with residual of {fits.cost}"
            )

        self.shift = fits.x[0]
        self.dkapdr = fits.x[1]
        self.s_kappa = self.r_minor / self.kappa * self.dkapdr
        self.dZ0dr = fits.x[2]
        self.dasym_dr = fits.x[3 : self.n_moments + 3]
        self.dsym_dr = fits.x[self.n_moments + 3 :]

        self.dthetaR_dr = self.get_dthetaR_dr(self.theta, self.dasym_dr, self.dsym_dr)

        self.get_b_poloidal(theta=self.theta)

    def get_thetaR(self, theta):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        thetaR = theta + np.sum(
            (
                self.asym_coeff[:, None] * np.cos(ntheta)
                + self.sym_coeff[:, None] * np.sin(ntheta)
            ),
            axis=0,
        )

        return thetaR

    def get_dthetaR_dtheta(self, theta):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        dthetaR_dtheta = 1.0 + np.sum(
            (
                -self.asym_coeff[:, None] * n[:, None] * np.sin(ntheta)
                + self.sym_coeff[:, None] * n[:, None] * np.cos(ntheta)
            ),
            axis=0,
        )

        return dthetaR_dtheta

    def get_dthetaR_dr(self, theta, dasym_dr, dsym_dr):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        dthetaR_dr = np.sum(
            (dasym_dr[:, None] * np.cos(ntheta) + dsym_dr[:, None] * np.sin(ntheta)),
            axis=0,
        )

        return dthetaR_dr

    def get_grad_r(
        self,
        theta: ArrayLike,
        params=None,
        normalised=False,
    ) -> np.ndarray:
        """
        MXH definition of grad r from
        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Parameters
        ----------
        kappa: Scalar
            elongation
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
            dkapdr = self.dkapdr
            dZ0dr = self.dZ0dr
            dasym_dr = self.dasym_dr
            dsym_dr = self.dsym_dr
        else:
            shift = params[0]
            dkapdr = params[1]
            dZ0dr = params[2]
            dasym_dr = params[3 : self.n_moments + 3]
            dsym_dr = params[self.n_moments + 3 :]

        thetaR = self.get_thetaR(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, dasym_dr, dsym_dr)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)

        dZdtheta = self.kappa * self.rho * np.cos(theta)

        # Assumes dZ0/dr = 0
        dZdr = dZ0dr + self.kappa * np.sin(theta) + dkapdr * self.rho * np.sin(theta)

        dRdtheta = -self.rho * np.sin(thetaR) * dthetaR_dtheta

        dRdr = shift + np.cos(thetaR) - self.rho * np.sin(thetaR) * dthetaR_dr

        g_tt = dRdtheta**2 + dZdtheta**2

        grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

        return grad_r

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of MXH fits

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

        thetaR = self.get_thetaR(theta)

        R = self.Rmaj + self.rho * np.cos(thetaR)
        Z = self.Z0 + self.kappa * self.rho * np.sin(theta)

        if not normalised:
            R *= self.a_minor
            Z *= self.a_minor

        return R, Z

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMXH, self).__init__(default_mxh_inputs())
