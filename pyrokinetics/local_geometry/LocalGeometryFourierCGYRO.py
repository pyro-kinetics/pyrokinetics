import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
from .LocalGeometry import default_inputs


def default_fourier_cgyro_inputs(n_moments=16):
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    fourier_cgyro_defaults = {
        "aR": np.zeros(n_moments),
        "aZ": np.zeros(n_moments),
        "bR": np.zeros(n_moments),
        "bZ": np.zeros(n_moments),
        "daRdr": np.zeros(n_moments),
        "daZdr": np.zeros(n_moments),
        "dbRdr": np.zeros(n_moments),
        "dbZdr": np.zeros(n_moments),
        "local_geometry": "fourier_cgyro",
    }

    return {**base_defaults, **fourier_cgyro_defaults}


def grad_r(
    theta: ArrayLike,
    aR: ArrayLike,
    aZ: ArrayLike,
    bR: ArrayLike,
    bZ: ArrayLike,
    daRdr: ArrayLike,
    daZdr: ArrayLike,
    dbRdr: ArrayLike,
    dbZdr: ArrayLike,
) -> np.ndarray:
    """
    fourier_cgyro definition of grad r from
    fourier_cgyro, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
    Physics of Plasmas 5.4 (1998): 973-978.

    Parameters
    ----------
    kappa: Scalar
        fourier_cgyro elongation
    shift: Scalar
        Shafranov shift
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    grad_r : Array
        grad_r(theta)
    """

    n_moments = len(aR)
    n = np.linspace(0, n_moments - 1, n_moments)
    ntheta = n[:, None] * theta[None, :]

    aR[0] *= 0.5
    aZ[0] *= 0.5

    dZdtheta = np.sum(
        n[:, None] * (-aZ[:, None] * np.sin(ntheta) + bZ[:, None] * np.cos(ntheta)),
        axis=0,
    )

    dZdr = np.sum(
        daZdr[:, None] * np.cos(ntheta) + dbZdr[:, None] * np.sin(ntheta), axis=0
    )

    dRdtheta = np.sum(
        n[:, None] * (-aR[:, None] * np.sin(ntheta) + bR[:, None] * np.cos(ntheta)),
        axis=0,
    )

    dRdr = np.sum(
        daRdr[:, None] * np.cos(ntheta) + dbRdr[:, None] * np.sin(ntheta), axis=0
    )

    aR[0] *= 2.0
    aZ[0] *= 2.0

    g_tt = dRdtheta ** 2 + dZdtheta ** 2

    grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    return grad_r


def flux_surface(
    theta: ArrayLike,
    aR: ArrayLike,
    aZ: ArrayLike,
    bR: ArrayLike,
    bZ: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (R,Z) of a flux surface given a set of fourier_cgyro fits

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

    aR[0] *= 0.5
    aZ[0] *= 0.5

    n_moments = len(aR)
    n = np.linspace(0, n_moments - 1, n_moments)
    ntheta = n[:, None] * theta[None, :]
    R = np.sum(aR[:, None] * np.cos(ntheta) + bR[:, None] * np.sin(ntheta), axis=0)
    Z = np.sum(aZ[:, None] * np.cos(ntheta) + bZ[:, None] * np.sin(ntheta), axis=0)

    aR[0] *= 2.0
    aZ[0] *= 2.0

    return R, Z


def get_b_poloidal(
    dpsidr: Scalar,
    R: ArrayLike,
    theta: ArrayLike,
    aR: ArrayLike,
    aZ: ArrayLike,
    bR: ArrayLike,
    bZ: ArrayLike,
    daRdr: ArrayLike,
    daZdr: ArrayLike,
    dbRdr: ArrayLike,
    dbZdr: ArrayLike,
) -> np.ndarray:
    r"""
    Returns fourier_cgyro prediction for b_poloidal given flux surface parameters

    Parameters
    ----------
    kappa: Scalar
        fourier_cgyro elongation
    delta: Scalar
        fourier_cgyro triangularity
    s_kappa: Scalar
        Radial derivative of fourier_cgyro elongation
    s_delta: Scalar
        Radial derivative of fourier_cgyro triangularity
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
    fourier_cgyro_b_poloidal : Array
        Array of b_poloidal from fourier_cgyro fit
    """

    return dpsidr / R * grad_r(theta, aR, aZ, bR, bZ, daRdr, daZdr, dbRdr, dbZdr)


class LocalGeometryFourierCGYRO(LocalGeometry):
    r"""
    Fourier Object representing local fourier_cgyro fit parameters
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
            and not isinstance(args[0], LocalGeometryFourierCGYRO)
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
        # be possible to have a fourier_cgyro object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        fourier_cgyro = cls(params)
        fourier_cgyro.bunit_over_b0 = fourier_cgyro.get_bunit_over_b0()
        return fourier_cgyro

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        fourier_cgyro = cls()
        fourier_cgyro.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return fourier_cgyro

    def load_from_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=16, **kwargs
    ):
        r"""
        Loads fourier_cgyro object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local fourier_cgyro parameters
        verbose : Boolean
            Controls verbosity

        """

        self.n_moments = n_moments

        super().load_from_eq(eq=eq, psi_n=psi_n, verbose=verbose, **kwargs)

    def load_from_lg(self, lg: LocalGeometry, verbose=False, n_moments=16, **kwargs):
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

    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False):
        r"""
        Calculate Shaping coefficients

        Parameters
        ----------
        verbose

        Returns
        -------

        """

        b_poloidal = np.roll(b_poloidal, -np.argmax(R))
        Z = np.roll(Z, -np.argmax(R))
        R = np.roll(R, -np.argmax(R))

        Zmid = (max(Z) + min(Z)) / 2

        R_major = (max(R) + min(R)) / 2

        R_diff = R - R_major
        Z_diff = Z - Zmid

        dot_product = R_diff * np.roll(R_diff, 1) + Z_diff * np.roll(Z_diff, 1)
        magnitude = np.sqrt(R_diff**2 + Z_diff**2)
        arc_angle = dot_product / (magnitude * np.roll(magnitude, 1))
        theta_diff = np.arccos(arc_angle)

        if Z[1] > Z[0]:
            theta = np.cumsum(theta_diff)
        else:
            theta = -np.cumsum(theta_diff)

        theta = theta - theta[0]

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

        self.R = R
        self.Z = Z
        self.b_poloidal = b_poloidal
        self.theta = theta

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * self.theta[None, :]
        aR = (
            simpson(
                self.R[
                    None,
                    :,
                ]
                * np.cos(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        aZ = (
            simpson(
                self.Z[
                    None,
                    :,
                ]
                * np.cos(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        bR = (
            simpson(
                self.R[
                    None,
                    :,
                ]
                * np.sin(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        bZ = (
            simpson(
                self.Z[
                    None,
                    :,
                ]
                * np.sin(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )

        self.Z0 = float(Zmid / self.a_minor)
        self.aR = aR
        self.aZ = aZ
        self.bR = bR
        self.bZ = bZ

        dpsidr_init = [0.0]

        # Roughly a cosine wave
        daRdr_init = [0.0, 1.0, *[0.0] * (self.n_moments - 2)]

        # Rougly a sine wave
        dbZdr_init = [0.0, 1.0, *[0.0] * (self.n_moments - 2)]

        daZdr_init = [*[0.0] * self.n_moments]
        dbRdr_init = [*[0.0] * self.n_moments]

        params = dpsidr_init + daRdr_init + daZdr_init + dbRdr_init + dbZdr_init

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

        self.dpsidr = fits.x[0]
        self.daRdr = fits.x[1 : self.n_moments + 1]
        self.daZdr = fits.x[self.n_moments + 1 : 2 * self.n_moments + 1]
        self.dbRdr = fits.x[2 * self.n_moments + 1 : 3 * self.n_moments + 1]
        self.dbZdr = fits.x[3 * self.n_moments + 1 :]

    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between fourier_cgyro and equilibrium b_poloidal

        """

        dpsidr = params[0]
        daRdr = params[1 : self.n_moments + 1]
        daZdr = params[self.n_moments + 1 : 2 * self.n_moments + 1]
        dbRdr = params[2 * self.n_moments + 1 : 3 * self.n_moments + 1]
        dbZdr = params[3 * self.n_moments + 1 :]

        return self.b_poloidal - get_b_poloidal(
            dpsidr=dpsidr,
            R=self.R,
            theta=self.theta,
            aR=self.aR,
            aZ=self.aZ,
            bR=self.bR,
            bZ=self.bZ,
            daRdr=daRdr,
            daZdr=daZdr,
            dbRdr=dbRdr,
            dbZdr=dbZdr,
        )

    def test_safety_factor(self):
        r"""
        Calculate safety fractor from fourier_cgyro Object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from fourier_cgyro B_poloidal
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
        bunit_over_b0 : Float
             :math:`\frac{B_{unit}}{B_0}`

        """

        theta = np.linspace(0, 2 * pi, 256)

        R, Z = flux_surface(
            theta=theta,
            aR=self.aR,
            aZ=self.aZ,
            bR=self.bR,
            bZ=self.bZ,
        )

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        R_grad_r = R * grad_r(
            theta=theta,
            aR=self.aR,
            aZ=self.aZ,
            bR=self.bR,
            bZ=self.bZ,
            daRdr=self.daRdr,
            daZdr=self.daZdr,
            dbRdr=self.dbRdr,
            dbZdr=self.dbZdr,
        )

        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def plot_fits(self):
        import matplotlib.pyplot as plt

        R_fit, Z_fit = flux_surface(
            self.theta, aR=self.aR, aZ=self.aZ, bR=self.bR, bZ=self.bZ
        )
        plt.plot(self.R, self.Z, label="Data")
        plt.plot(R_fit, Z_fit, "--", label="Fit")
        ax = plt.gca()

        ax.set_aspect("equal")
        plt.title("Fit to flux surface for CGYRO Fourier")
        plt.legend()
        plt.grid()
        plt.show()

        bpol_fit = get_b_poloidal(
            dpsidr=self.dpsidr,
            R=self.R,
            theta=self.theta,
            aR=self.aR,
            aZ=self.aZ,
            bR=self.bR,
            bZ=self.bZ,
            daRdr=self.daRdr,
            daZdr=self.daZdr,
            dbRdr=self.dbRdr,
            dbZdr=self.dbZdr,
        )

        plt.plot(self.theta, self.b_poloidal, label="Data")
        plt.plot(self.theta, bpol_fit, "--", label=f"N moments={self.n_moments}")
        plt.legend()
        plt.xlabel("theta")
        plt.title("Fit to poloidal field for CGYRO Fourier")
        plt.ylabel("Bpol")
        plt.grid()
        plt.show()

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryFourierCGYRO, self).__init__(default_fourier_cgyro_inputs())
