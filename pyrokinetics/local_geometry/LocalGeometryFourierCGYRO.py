import numpy as np
from typing import Tuple
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import ArrayLike
from .LocalGeometry import default_inputs


def default_fourier_cgyro_inputs(n_moments=16):
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    fourier_cgyro_defaults = {
        "n_moments": n_moments,
        "aR": np.array([3.0, 0.5, *[0.0] * (n_moments - 2)]),
        "aZ": np.array([0.0, 0.5, *[0.0] * (n_moments - 2)]),
        "bR": np.zeros(n_moments),
        "bZ": np.zeros(n_moments),
        "daRdr": np.zeros(n_moments),
        "daZdr": np.zeros(n_moments),
        "dbRdr": np.zeros(n_moments),
        "dbZdr": np.zeros(n_moments),
        "a_minor": 1.0,
        "local_geometry": "FourierCGYRO",
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

    def from_global_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=16, show_fit=False
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

        super().from_global_eq(eq=eq, psi_n=psi_n, verbose=verbose, show_fit=show_fit)

    def from_local_geometry(
        self, local_geometry: LocalGeometry, verbose=False, n_moments=16, show_fit=False
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

        super().from_local_geometry(
            local_geometry=local_geometry, verbose=verbose, show_fit=show_fit
        )

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

        # Interpolate to evenly spaced theta
        theta_new = np.linspace(0, 2 * np.pi, len(theta))
        R = np.interp(theta_new, theta, R)
        Z = np.interp(theta_new, theta, Z)
        b_poloidal = np.interp(theta_new, theta, b_poloidal)
        theta = theta_new

        self.theta = theta
        self.b_poloidal_even_space = b_poloidal

        ntheta = np.outer(self.n, theta)
        aR = (
            simpson(
                R * np.cos(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        aZ = (
            simpson(
                Z * np.cos(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        bR = (
            simpson(
                R * np.sin(ntheta),
                self.theta,
                axis=1,
            )
            / np.pi
        )
        bZ = (
            simpson(
                Z * np.sin(ntheta),
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

        fits = least_squares(
            self.minimise_b_poloidal, params, kwargs={"even_space_theta": "True"}
        )

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in Fourier::from_global_eq failed with message : {fits.message}"
            )

        if verbose:
            print(
                f"FourierCGYRO :: Fit to Bpoloidal obtained with residual {fits.cost}"
            )

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in FourierCGYRO::from_global_eq is poor with residual of {fits.cost}"
            )

        self.daRdr = fits.x[0 : self.n_moments]
        self.daZdr = fits.x[self.n_moments : 2 * self.n_moments]
        self.dbRdr = fits.x[2 * self.n_moments : 3 * self.n_moments]
        self.dbZdr = fits.x[3 * self.n_moments :]

    @property
    def n(self):
        pass

    @n.getter
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    def get_RZ_derivatives(
        self,
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
            daRdr = params[0 : self.n_moments]
            daZdr = params[self.n_moments : 2 * self.n_moments]
            dbRdr = params[2 * self.n_moments : 3 * self.n_moments]
            dbZdr = params[3 * self.n_moments :]

        dZdtheta = self.get_dZdtheta(theta)

        dZdr = self.get_dZdr(theta, daZdr, dbZdr)

        dRdtheta = self.get_dRdtheta(theta)

        dRdr = self.get_dRdr(theta, daRdr, dbRdr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_dZdtheta(self, theta):

        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-self.aZ * np.sin(ntheta) + self.bZ * np.cos(ntheta)),
            axis=1,
        )

    def get_dZdr(self, theta, daZdr, dbZdr):

        ntheta = np.outer(theta, self.n)

        return np.sum(daZdr * np.cos(ntheta) + dbZdr * np.sin(ntheta), axis=1)

    def get_dRdtheta(self, theta):

        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-self.aR * np.sin(ntheta) + self.bR * np.cos(ntheta)),
            axis=1,
        )

    def get_dRdr(self, theta, daRdr, dbRdr):

        ntheta = np.outer(theta, self.n)

        return np.sum(daRdr * np.cos(ntheta) + dbRdr * np.sin(ntheta), axis=1)

    def get_flux_surface(
        self, theta: ArrayLike, normalised=True
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

        ntheta = np.outer(theta, self.n)

        R = np.sum(
            self.aR * np.cos(ntheta) + self.bR * np.sin(ntheta),
            axis=1,
        )
        Z = np.sum(
            self.aZ * np.cos(ntheta) + self.bZ * np.sin(ntheta),
            axis=1,
        )

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
