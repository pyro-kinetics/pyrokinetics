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


def grad_r(
    kappa: Scalar,
    r: Scalar,
    shift: Scalar,
    dkapdr: Scalar,
    dZ0dr: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    dthetaR_dtheta: ArrayLike,
    dthetaR_dr: ArrayLike,
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

    dZdtheta = kappa * r * np.cos(theta)

    # Assumes dZ0/dr = 0
    dZdr = dZ0dr + kappa * np.sin(theta) + dkapdr * r * np.sin(theta)

    dRdtheta = -r * np.sin(thetaR) * dthetaR_dtheta

    dRdr = shift + np.cos(thetaR) - r * np.sin(thetaR) * dthetaR_dr

    g_tt = dRdtheta ** 2 + dZdtheta ** 2

    grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    return grad_r


def flux_surface(
    kappa: Scalar,
    Rcen: Scalar,
    rmin: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    Zmid: Scalar,
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
    R = Rcen + rmin * np.cos(thetaR)
    Z = Zmid + kappa * rmin * np.sin(theta)

    return R, Z


def get_b_poloidal(
    kappa: Scalar,
    R: ArrayLike,
    r: Scalar,
    shift: Scalar,
    dZ0dr: Scalar,
    dpsidr: Scalar,
    dkapdr: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    dthetaR_dtheta: ArrayLike,
    dthetaR_dr: ArrayLike,
) -> np.ndarray:
    r"""
    Returns mxh prediction for get_b_poloidal given flux surface parameters

    Parameters
    ----------
    kappa: Scalar
        mxh elongation
    delta: Scalar
        mxh triangularity
    s_kappa: Scalar
        Radial derivative of mxh elongation
    s_delta: Scalar
        Radial derivative of mxh triangularity
    shift: Scalar
        Shafranov shift
    dpsidr: Scalar
        :math: `\partial \psi / \partial r`
    R: ArrayLike
        Major radius
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    mxh_b_poloidal : Array
        Array of get_b_poloidal from mxh fit
    """

    return (
        dpsidr
        / R
        * grad_r(
            kappa, r, shift, dkapdr, dZ0dr, theta, thetaR, dthetaR_dtheta, dthetaR_dr
        )
    )


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
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=4, **kwargs
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

        super().load_from_eq(eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, **kwargs)

    def load_from_lg(self, lg: LocalGeometry, verbose=False, n_moments=4, **kwargs):
        r"""
        Loads mxh object from a LocalGeometry Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        lg : LocalGeometry
            LocalGeometry object
        verbose : Boolean
            Controls verbosity
        n_moments: Int
            Number of moments to fit with

        """

        self.n_moments = n_moments

        super().load_from_lg(lg=lg, verbose=verbose, **kwargs)

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

        # Ensure theta start from zero and remove any repeats
        theta = np.roll(theta, -np.argmin(theta))
        thetaR = np.roll(thetaR, -np.argmin(thetaR))
        thetaR = thetaR[np.where(np.diff(theta) != 0.0)]
        self.R = self.R[np.where(np.diff(theta) != 0.0)]
        self.Z = Z[np.where(np.diff(theta) != 0.0)]
        self.b_poloidal = b_poloidal[np.where(np.diff(theta) != 0.0)]
        theta = theta[np.where(np.diff(theta) != 0.0)]

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
        dkap_dr = fits.x[1]
        self.s_kappa = self.r_minor / self.kappa * dkap_dr
        self.dZ0dr = fits.x[2]
        self.dasym_dr = fits.x[3 : self.n_moments + 3]
        self.dsym_dr = fits.x[self.n_moments + 3 :]

        self.dthetaR_dr = self.get_dthetaR_dr(self.theta, self.dasym_dr, self.dsym_dr)

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

    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between mxh and equilibrium get_b_poloidal

        """

        shift = params[0]
        dkapdr = params[1]
        dZ0dr = params[2]
        dasym_dr = params[3 : self.n_moments + 3]
        dsym_dr = params[self.n_moments + 3 :]
        dthetaR_dr = self.get_dthetaR_dr(self.theta, dasym_dr, dsym_dr)

        return self.b_poloidal - get_b_poloidal(
            kappa=self.kappa,
            r=self.r_minor,
            shift=shift,
            dpsidr=self.dpsidr,
            dkapdr=dkapdr,
            dZ0dr=dZ0dr,
            R=self.R,
            theta=self.theta,
            thetaR=self.thetaR,
            dthetaR_dtheta=self.dthetaR_dtheta,
            dthetaR_dr=dthetaR_dr,
        )

    def test_safety_factor(self):
        r"""
        Calculate safety fractor from mxh Object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from mxh B_poloidal
        """

        R = self.R
        Z = self.Z

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        b_poloidal = self.get_b_poloidal

        f = self.f_psi

        integral = np.sum(f * dL / (R ** 2 * b_poloidal))

        q = integral / (2 * pi)

        return q

    def get_bunit_over_b0(self):
        r"""
        Get Bunit/B0 using q and loop integral of Bp

        :math:`\frac{B_{unit}}{B_0} = \frac{R_0}{2\pi r_{minor}} \oint \frac{a}{R} \frac{dl_N}{\nabla r}`

        where :math:`dl_N = \frac{dl}{a_{minor}}` coming from the normalising a_minor

        Returns
        -------
        bunit_over_b0 : Float
             :math:`\frac{B_{unit}}{B_0}`

        """

        theta = np.linspace(0, 2 * pi, 256)

        thetaR = self.get_thetaR(theta)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, self.dasym_dr, self.dsym_dr)

        R, Z = flux_surface(
            kappa=self.kappa,
            Rcen=self.Rmaj,
            rmin=self.rho,
            Zmid=self.Z0,
            theta=theta,
            thetaR=thetaR,
        )

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        R_grad_r = R * grad_r(
            self.kappa,
            self.r_minor,
            self.shift,
            self.s_kappa * self.kappa / self.r_minor,
            self.dZ0dr,
            theta,
            thetaR,
            dthetaR_dtheta,
            dthetaR_dr,
        )

        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def plot_fits(self):
        import matplotlib.pyplot as plt

        R_fit, Z_fit, = flux_surface(
            kappa=self.kappa,
            Rcen=self.Rmaj * self.a_minor,
            rmin=self.r_minor,
            theta=self.theta,
            thetaR=self.thetaR,
            Zmid=self.Z0 * self.a_minor,
        )

        plt.plot(self.R, self.Z, label="Data")
        plt.plot(R_fit, Z_fit, "--", label="Fit")
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.title("Fit to flux surface for MXH")
        plt.legend()
        plt.grid()
        plt.show()

        bpol_fit = get_b_poloidal(
            kappa=self.kappa,
            r=self.r_minor,
            shift=self.shift,
            dkapdr=self.s_kappa * self.kappa / self.r_minor,
            dpsidr=self.dpsidr,
            dZ0dr=self.dZ0dr,
            R=self.R,
            theta=self.theta,
            thetaR=self.thetaR,
            dthetaR_dtheta=self.dthetaR_dtheta,
            dthetaR_dr=self.dthetaR_dr,
        )

        plt.plot(self.theta, self.b_poloidal, label="Data")
        plt.plot(self.theta, bpol_fit, "--", label=f"N moments={self.n_moments}")
        plt.legend()
        plt.xlabel("theta")
        plt.title("Fit to poloidal field for MXH")
        plt.ylabel("Bpol")
        plt.grid()
        plt.show()

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMXH, self).__init__(default_mxh_inputs())
