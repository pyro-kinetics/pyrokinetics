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
        "aR": np.array([1.0, *[0.0] * (n_moments - 1)]),
        "aZ": np.array([1.0, *[0.0] * (n_moments - 1)]),
        "bR": np.zeros(n_moments),
        "bZ": np.zeros(n_moments),
        "daRdr": np.zeros(n_moments),
        "daZdr": np.zeros(n_moments),
        "dbRdr": np.zeros(n_moments),
        "dbZdr": np.zeros(n_moments),
        "local_geometry": "Fourier_cgyro",
    }

    return {**base_defaults, **fourier_cgyro_defaults}

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
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=16, show_fit=False):
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

        super().load_from_eq(eq=eq, psi_n=psi_n, verbose=verbose, show_fit=show_fit)

    def load_from_local_geometry(self, local_geometry: LocalGeometry, verbose=False, n_moments=16, show_fit=False):
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

        super().load_from_local_geometry(local_geometry=local_geometry, verbose=verbose, show_fit=show_fit)

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

        dR = R - np.roll(R, 1)
        dZ = Z - np.roll(Z, 1)

        dl = np.sqrt(dR**2 + dZ**2)

        full_length = np.sum(dl)

        theta = np.cumsum(dl) * 2 * pi / full_length
        theta = theta - theta[0]

        self.theta_eq = theta

        Zmid = (max(Z) + min(Z)) / 2

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

        self.theta = theta
        self.b_poloidal_even_space = b_poloidal

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * self.theta[None, :]
        aR = (
            simpson(
                R[
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
                Z[
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
                R[
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
                Z[
                    None,
                    :,
                ]
                * np.sin(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )

        aR[0] *= 0.5
        aZ[0] *= 0.5

        self.Z0 = float(Zmid / self.a_minor)
        self.aR = aR
        self.aZ = aZ
        self.bR = bR
        self.bZ = bZ

        self.R, self.Z = self.get_flux_surface(self.theta)

        # Roughly a cosine wave
        daRdr_init = [0.0, 1.0, *[0.0] * (self.n_moments - 2)]

        # Rougly a sine wave
        dbZdr_init = [0.0, 1.0, *[0.0] * (self.n_moments - 2)]

        daZdr_init = [*[0.0] * self.n_moments]
        dbRdr_init = [*[0.0] * self.n_moments]

        params = daRdr_init + daZdr_init + dbRdr_init + dbZdr_init

        fits = least_squares(self.minimise_b_poloidal, params, kwargs={"even_space_theta":"True"})

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in Fourier::load_from_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"FourierCGYRO :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in FourierCGYRO::load_from_eq is poor with residual of {fits.cost}"
            )

        self.daRdr = fits.x[0 : self.n_moments]
        self.daZdr = fits.x[self.n_moments: 2 * self.n_moments]
        self.dbRdr = fits.x[2 * self.n_moments: 3 * self.n_moments]
        self.dbZdr = fits.x[3 * self.n_moments:]

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta)


    def get_grad_r(self,
            theta: ArrayLike,
            params=None,
            normalised=False,
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

        if params is None:
            daRdr = self.daRdr
            daZdr = self.daZdr
            dbRdr = self.dbRdr
            dbZdr = self.dbZdr
        else:
            daRdr = params[0: self.n_moments]
            daZdr = params[self.n_moments: 2 * self.n_moments]
            dbRdr = params[2 * self.n_moments: 3 * self.n_moments]
            dbZdr = params[3 * self.n_moments:]

        n_moments = len(self.aR)
        n = np.linspace(0, n_moments - 1, n_moments)
        ntheta = n[:, None] * theta[None, :]

        dZdtheta = np.sum(
            n[:, None] * (-self.aZ[:, None] * np.sin(ntheta) + self.bZ[:, None] * np.cos(ntheta)),
            axis=0,
        )

        dZdr = np.sum(
            daZdr[:, None] * np.cos(ntheta) + dbZdr[:, None] * np.sin(ntheta), axis=0
        )

        dRdtheta = np.sum(
            n[:, None] * (-self.aR[:, None] * np.sin(ntheta) + self.bR[:, None] * np.cos(ntheta)),
            axis=0,
        )

        dRdr = np.sum(
            daRdr[:, None] * np.cos(ntheta) + dbRdr[:, None] * np.sin(ntheta), axis=0
        )

        g_tt = dRdtheta ** 2 + dZdtheta ** 2

        grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

        return grad_r

    def get_flux_surface(self,
            theta: ArrayLike,
            normalised=True
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

        n_moments = len(self.aR)
        n = np.linspace(0, n_moments - 1, n_moments)
        ntheta = n[:, None] * theta[None, :]
        R = np.sum(self.aR[:, None] * np.cos(ntheta) + self.bR[:, None] * np.sin(ntheta), axis=0)
        Z = np.sum(self.aZ[:, None] * np.cos(ntheta) + self.bZ[:, None] * np.sin(ntheta), axis=0)

        if normalised:
            R *= 1 / self.a_minor
            Z *= 1 / self.a_minor

        return R, Z


    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryFourierCGYRO, self).__init__(default_fourier_cgyro_inputs())
