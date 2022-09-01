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
        "cN": np.zeros(n_moments),
        "sN": np.zeros(n_moments),
        "dcNdr": np.zeros(n_moments),
        "dsNdr": np.zeros(n_moments),
        "local_geometry": "fourier",
    }

    return {**base_defaults, **fourier_defaults}


def grad_r(
    theta: ArrayLike,
    dZ0dr: Scalar,
    cN: ArrayLike,
    sN: ArrayLike,
    dcNdr: ArrayLike,
    dsNdr: ArrayLike,
    shift: Scalar,
    Rmaj: Scalar,
    Z0: Scalar,
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

    n_moments = len(cN)
    n = np.linspace(0, n_moments - 1, n_moments)
    ntheta = n[:, None] * theta[None, :]

    cN[0] *= 0.5
    sN[0] *= 0.5

    aN = np.sum(cN[:, None] * np.cos(ntheta) + sN[:, None] * np.sin(ntheta), axis=0)
    daNdr = np.sum(
        dcNdr[:, None] * np.cos(ntheta) + dsNdr[:, None] * np.sin(ntheta), axis=0
    )
    daNdtheta = np.sum(
        -cN[:, None] * n[:, None] * np.sin(ntheta)
        + sN[:, None] * n[:, None] * np.cos(ntheta),
        axis=0,
    )

    cN[0] *= 2.0
    sN[0] *= 2.0

    dZdtheta = aN * np.cos(theta) + daNdtheta * np.sin(theta)

    # Assumes dZ0/dr = 0
    dZdr = dZ0dr + aN * np.sin(theta) + Z0 * daNdr * np.sin(theta)

    dRdtheta = -aN * np.sin(theta) + daNdtheta * np.cos(theta)

    dRdr = shift + aN * np.cos(theta) + Rmaj * daNdr * np.sin(theta)

    g_tt = dRdtheta ** 2 + dZdtheta ** 2

    grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    return grad_r


def flux_surface(
    theta: ArrayLike,
    cN: ArrayLike,
    sN: ArrayLike,
    a_minor: Scalar,
    R_major: Scalar,
    Zmid: Scalar,
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
    n_moments = len(cN)
    n = np.linspace(0, n_moments - 1, n_moments)

    cN[0] *= 0.5
    sN[0] *= 0.5

    ntheta = n[:, None] * theta[None, :]
    aN = (
        np.sum(cN[:, None] * np.cos(ntheta) + sN[:, None] * np.sin(ntheta), axis=0)
        * a_minor
    )

    cN[0] *= 2.0
    sN[0] *= 2.0

    R = R_major + aN * np.cos(theta)
    Z = Zmid + aN * np.sin(theta)

    return R, Z


def get_b_poloidal(
    theta: ArrayLike,
    cN: ArrayLike,
    sN: ArrayLike,
    dcNdr: ArrayLike,
    dsNdr: ArrayLike,
    Rmaj: Scalar,
    Z0: Scalar,
    R: ArrayLike,
    shift: Scalar,
    dZ0dr: Scalar,
    dpsidr: Scalar,
) -> np.ndarray:
    r"""
    Returns fourier prediction for b_poloidal given flux surface parameters

    Parameters
    ----------
    kappa: Scalar
        fourier elongation
    delta: Scalar
        fourier triangularity
    s_kappa: Scalar
        Radial derivative of fourier elongation
    s_delta: Scalar
        Radial derivative of fourier triangularity
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
    fourier_b_poloidal : Array
        Array of b_poloidal from fourier fit
    """

    return (
        dpsidr
        / R
        * grad_r(
            theta=theta,
            dZ0dr=dZ0dr,
            cN=cN,
            sN=sN,
            dcNdr=dcNdr,
            dsNdr=dsNdr,
            Rmaj=Rmaj,
            Z0=Z0,
            shift=shift,

        )
    )


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

    def load_from_eq(self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=32, **kwargs):
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

        super().load_from_eq(eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, **kwargs)


    def load_from_lg(self, lg: LocalGeometry, verbose=False, n_moments=32, **kwargs):
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
        Z = np.roll(Z, -np.argmax(R))
        R = np.roll(R, -np.argmax(R))

        R_major = self.Rmaj * self.a_minor
        Zmid = self.Z0 * self.a_minor

        R_diff = R - R_major
        Z_diff = Z - Zmid

        dot_product = R_diff * np.roll(R_diff, 1) + Z_diff * np.roll(Z_diff, 1)
        magnitude = np.sqrt(R_diff ** 2 + Z_diff ** 2)
        arc_angle = dot_product / (magnitude * np.roll(magnitude, 1))

        theta_diff = np.arccos(arc_angle)

        if Z[1] > Z[0]:
            theta = np.cumsum(theta_diff) - theta_diff[0]
        else:
            theta = -np.cumsum(theta_diff) - theta_diff[0]

        # Remove same points
        R = R[np.where(np.diff(theta) != 0.0)]
        Z = Z[np.where(np.diff(theta) != 0.0)]
        b_poloidal = b_poloidal[np.where(np.diff(theta) != 0.0)]
        theta = theta[np.where(np.diff(theta) != 0.0)]

        # Ensure final point matches first point
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
        b_poloidal = np.append(b_poloidal, b_poloidal[0])
        theta = np.append(theta, 2 * np.pi + theta[0])

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

        self.cN = cN
        self.sN = sN

        self.theta = theta
        self.R = R
        self.Z = Z
        self.b_poloidal = b_poloidal

        dpsi_dr_init = 1.0
        dZ0dr = 0.0
        params = [shift, dZ0dr, dpsi_dr_init, *[0.0] * self.n_moments * 2]

        fits = least_squares(self.minimise_b_poloidal, params)

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
                f"Warning Fit to Fourier in Miller::load_from_eq is poor with residual of {fits.cost}"
            )

        self.shift = fits.x[0]
        self.dZ0dr = fits.x[1]
        self.dpsidr = fits.x[2]
        self.dcNdr = fits.x[3 : self.n_moments + 3]
        self.dsNdr = fits.x[self.n_moments + 3 :]


    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between fourier and equilibrium b_poloidal

        """

        shift = params[0]
        dZ0dr = params[1]
        dpsidr = params[2]
        dcNdr = params[3 : self.n_moments + 3]
        dsNdr = params[self.n_moments + 3 :]

        return self.b_poloidal - get_b_poloidal(
            theta=self.theta,
            cN=self.cN,
            sN=self.sN,
            dcNdr=dcNdr,
            dsNdr=dsNdr,
            Rmaj=self.Rmaj,
            Z0=self.Z0,
            R=self.R,
            shift=shift,
            dZ0dr=dZ0dr,
            dpsidr=dpsidr,
        )

    def test_safety_factor(self):
        r"""
        Calculate safety fractor from fourier Object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from fourier B_poloidal
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

        R, Z = flux_surface(
            theta, self.cN, self.sN, self.a_minor, self.Rmaj, self.Z0
        )

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        R_grad_r = R * grad_r(
            theta,
            self.dZ0dr,
            self.cN,
            self.sN,
            self.dcNdr,
            self.dsNdr,
            self.shift,
            self.Rmaj,
            self.Z0,
        )

        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def plot_fits(self):
        import matplotlib.pyplot as plt

        R_fit, Z_fit = flux_surface(self.theta, self.cN, self.sN, self.a_minor, self.Rmaj * self.a_minor,
                                    self.Z0 * self.a_minor)

        plt.plot(self.R, self.Z, label='Data')
        plt.plot(R_fit, Z_fit, '--', label='Fit')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("Fit to flux surface for GENE Fourier")
        plt.legend()
        plt.grid()
        plt.show()

        bpol_fit = get_b_poloidal(
            theta=self.theta,
            cN=self.cN,
            sN=self.sN,
            dcNdr=self.dcNdr,
            dsNdr=self.dsNdr,
            Rmaj=self.Rmaj,
            Z0=self.Z0,
            R=self.R,
            shift=self.shift,
            dZ0dr=self.dZ0dr,
            dpsidr=self.dpsidr,
        )

        plt.plot(self.theta, self.b_poloidal, label="Data")
        plt.plot(self.theta, bpol_fit, "--", label=f"N moments={self.n_moments}")
        plt.legend()
        plt.xlabel("theta")
        plt.title("Fit to poloidal field for GENE Fourier")
        plt.ylabel("Bpol")
        plt.grid()
        plt.show()


    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryFourier, self).__init__(default_fourier_inputs())
