import numpy as np
from typing import Tuple
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..typing import ArrayLike
from .LocalGeometry import default_inputs


def default_fourier_cgyro_inputs():
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    n_moments = 16
    fourier_cgyro_defaults = {
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
    Local equilibrium representation defined as in:
    Plasma Phys. Control. Fusion 51 (2009) 105009 J Candy https://doi.org/10.1088/0741-3335/51/10/105009
    FourierCGYRO

    R(r, theta) = 0.5 aR_0(r) + sum_n=1^N [aR_n(r) * cos(n*theta) + bR_n(r) * sin(n*theta)]
    Z(r, theta) = 0.5 aZ_0(r) + sum_n=1^N [aZ_n(r) * cos(n*theta) + bZ_n(r) * sin(n*theta)]

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
    Rgeo : Float
        Normalisd major radius of normalising field (Rreference/a)
    Z0 : Float
        Normalised vertical position of midpoint (Zmid / a_minor)
    f_psi : Float
        Torodial field function
    B0 : Float
        Toroidal field at major radius (Fpsi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear
    beta_prime : Float
        :math:`\beta' = \beta * a/L_p`

    aR : ArrayLike
        cosine moments of R
    aZ : ArrayLike
        cosine moments of Z
    bR : ArrayLike
        sine moments of R
    bZ : ArrayLike
        sine moments of Z
    daRdr : ArrayLike
        Derivative of aR w.r.t r
    daZdr : ArrayLike
        Derivative of aZ w.r.t r
    dbRdr : ArrayLike
        Derivative of bR w.r.t r
    dbZdr : ArrayLike
        Derivative of bZ w.r.t r

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
            and not isinstance(args[0], LocalGeometryFourierCGYRO)
            and isinstance(args[0], dict)
        ):
            super().__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False):
        r"""
        Calculates FourierCGYRO shaping coefficients from R, Z and b_poloidal

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
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    @property
    def n_moments(self):
        return 16

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
            If given then will use params = [daRdr[nmoments], daZdr[nmoments], dbRdr[nmoments], dbZdr[nmoments] ] when calculating
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

        dZdtheta = self.get_dZdtheta(theta, normalised)

        dZdr = self.get_dZdr(theta, daZdr, dbZdr)

        dRdtheta = self.get_dRdtheta(theta, normalised)

        dRdr = self.get_dRdr(theta, daRdr, dbRdr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
        normalised=False,
    ) -> np.ndarray:
        """
        Calculates the second derivatives of `R(r, \theta)` and `Z(r, \theta)` w.r.t `r` and `\theta`, used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        d2Rdtheta2 : Array
                        Second derivative of `R` w.r.t `\theta`
        d2Rdrdtheta : Array
                        Second derivative of `R` w.r.t `r` and `\theta`
        d2Zdtheta2 : Array
                        Second derivative of `Z` w.r.t `\theta`
        d2Zdrdtheta : Array
                        Second derivative of `Z` w.r.t `r` and `\theta`
        """

        d2Zdtheta2 = self.get_d2Zdtheta2(theta, normalised)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.daZdr, self.dbZdr)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta, normalised)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.daRdr, self.dbRdr)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_dZdtheta(self, theta, normalised=False):
        """
        Calculates the derivatives of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        Returns
        -------
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        """
        ntheta = np.outer(theta, self.n)

        if normalised:
            fac = 1.0 / self.a_minor
        else:
            fac = 1.0

        return fac * np.sum(
            self.n * (-self.aZ * np.sin(ntheta) + self.bZ * np.cos(ntheta)),
            axis=1,
        )

    def get_d2Zdtheta2(self, theta, normalised=False):
        """
        Calculates the second derivative of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of `Z` w.r.t `\theta`
        """

        ntheta = np.outer(theta, self.n)

        if normalised:
            fac = 1.0 / self.a_minor
        else:
            fac = 1.0

        return fac * np.sum(
            -(self.n**2) * (self.aZ * np.cos(ntheta) + self.bZ * np.sin(ntheta)),
            axis=1,
        )

    def get_dZdr(self, theta, daZdr, dbZdr):
        """
        Calculates the derivatives of `Z(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daZdr : ArrayLike
            Derivative in aZ w.r.t r
        dbZdr : ArrayLike
            Derivative of bZ w.r.t r
        Returns
        -------
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(daZdr * np.cos(ntheta) + dbZdr * np.sin(ntheta), axis=1)

    def get_d2Zdrdtheta(self, theta, daZdr, dbZdr):
        """
        Calculates the second derivative of `Z(r, \theta)` w.r.t `r` and `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daZdr : ArrayLike
            Derivative in aZ w.r.t r
        dbZdr : ArrayLike
            Derivative of bZ w.r.t r
        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of `Z` w.r.t `r` and `\theta`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-daZdr * np.sin(ntheta) + dbZdr * np.cos(ntheta)), axis=1
        )

    def get_dRdtheta(self, theta, normalised=False):
        """
        Calculates the derivatives of `R(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on
        Returns
        -------
        dRdtheta : Array
            Derivative of `R` w.r.t `\theta`
        """
        ntheta = np.outer(theta, self.n)

        if normalised:
            fac = 1.0 / self.a_minor
        else:
            fac = 1.0

        return fac * np.sum(
            self.n * (-self.aR * np.sin(ntheta) + self.bR * np.cos(ntheta)),
            axis=1,
        )

    def get_d2Rdtheta2(self, theta, normalised=False):
        """
        Calculates the second derivative of `R(r, \theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on
        Returns
        -------
        d2Rdtheta2 : Array
            Second derivative of `R` w.r.t `\theta`
        """
        ntheta = np.outer(theta, self.n)

        if normalised:
            fac = 1.0 / self.a_minor
        else:
            fac = 1.0

        return fac * np.sum(
            -(self.n**2) * (self.aR * np.cos(ntheta) + self.bR * np.sin(ntheta)),
            axis=1,
        )

    def get_dRdr(self, theta, daRdr, dbRdr):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daRdr : ArrayLike
            Derivative in aR w.r.t r
        dbRdr : ArrayLike
            Derivative of bR w.r.t r
        Returns
        -------
        dRdr : Array
            Derivative of `R` w.r.t `r`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(daRdr * np.cos(ntheta) + dbRdr * np.sin(ntheta), axis=1)

    def get_d2Rdrdtheta(self, theta, daRdr, dbRdr):
        """
        Calculate the second derivative of `R(r, \theta)` w.r.t `r` and `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daRdr : ArrayLike
            Derivative in aR w.r.t r
        dbRdr : ArrayLike
            Derivative of bR w.r.t r
        Returns
        -------
        d2Rdrdtheta : Array
            Second derivative of R w.r.t `r` and `\theta`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-daRdr * np.sin(ntheta) + dbRdr * np.cos(ntheta)), axis=1
        )

    def get_flux_surface(
        self, theta: ArrayLike, normalised=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of FourierCGYRO fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface
        normalised : Boolean
            Control whether or not to return normalised flux surface
        Returns
        -------
        R : Array
            R values for this flux surface (if not normalised then in [m])
        Z : Array
            Z Values for this flux surface (if not normalised then in [m])
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
