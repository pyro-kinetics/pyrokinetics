import numpy as np
from typing import Tuple
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import ArrayLike
from .LocalGeometry import default_inputs


def default_fourier_gene_inputs():
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    n_moments = 32
    fourier_defaults = {
        "dZ0dr": 0.0,
        "shift": 0.0,
        "cN": np.array([0.5, *[0.0] * (n_moments - 1)]),
        "sN": np.zeros(n_moments),
        "dcNdr": np.array([1.0, *[0.0] * (n_moments - 1)]),
        "dsNdr": np.zeros(n_moments),
        "local_geometry": "FourierGENE",
    }

    return {**base_defaults, **fourier_defaults}


class LocalGeometryFourierGENE(LocalGeometry):
    r"""
    Local equilibrium representation defined as in:
    Fourier representation used in GENE https://gitlab.mpcdf.mpg.de/GENE/gene/-/blob/release-2.0/doc/gene.tex
    FourierGENE

    aN(r, theta) = sqrt( (R(r, theta) - R_0)**2 - (Z(r, theta) - Z_0)**2 ) / Lref
                 =  sum_n=0^N [cn(r) * cos(n*theta) + sn(r) * sin(n*theta)]

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
        Toroidal field at major radius (f_psi / Rmajor) [T]
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

    shift : Float
        Shafranov shift
    dZ0dr : Float
        Shear in midplane elevation
    cN : ArrayLike
        cosine moments of aN
    sN : ArrayLike
        sine moments of aN
    dcNdr : ArrayLike
        Derivative of cosine moments w.r.t r
    dsNdr : ArrayLike
        Derivative of sine moments w.r.t r
    aN : ArrayLike
        aN values at theta
    daNdtheta : ArrayLike
        Derivative of aN w.r.t theta at theta
    daNdr : ArrayLike
        Derivative of aN w.r.t r at theta

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
            and not isinstance(args[0], LocalGeometryFourierGENE)
            and isinstance(args[0], dict)
        ):
            super().__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates FourierGENE shaping coefficients from R, Z and b_poloidal

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

        ntheta = np.outer(self.n, theta)

        cN = simpson(aN * np.cos(ntheta), theta, axis=1) / np.pi
        sN = simpson(aN * np.sin(ntheta), theta, axis=1) / np.pi

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
                f"Least squares fitting in Fourier::from_global_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"Fourier :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in Fourier::from_global_eq is poor with residual of {fits.cost}"
            )

        self.shift = fits.x[0]
        self.dZ0dr = fits.x[1]
        self.dcNdr = fits.x[2 : self.n_moments + 2]
        self.dsNdr = fits.x[self.n_moments + 2 :]

        ntheta = np.outer(theta, self.n)

        self.aN = np.sum(
            self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
            axis=1,
        )
        self.daNdr = np.sum(
            self.dcNdr * np.cos(ntheta) + self.dsNdr * np.sin(ntheta),
            axis=1,
        )
        self.daNdtheta = np.sum(
            -self.cN * self.n * np.sin(ntheta) + self.sN * self.n * np.cos(ntheta),
            axis=1,
        )

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    @property
    def n_moments(self):
        return 32

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
            If given then will use params = [shift, dZ0dr, , dcNdr[nmoments], dsNdr[nmoments] ] when calculating
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
            shift = self.shift
            dZ0dr = self.dZ0dr
            dcNdr = self.dcNdr
            dsNdr = self.dsNdr
        else:
            shift = params[0]
            dZ0dr = params[1]
            dcNdr = params[2 : self.n_moments + 2]
            dsNdr = params[self.n_moments + 2 :]

        ntheta = np.outer(theta, self.n)

        aN = np.sum(
            self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
            axis=1,
        )
        daNdr = np.sum(dcNdr * np.cos(ntheta) + dsNdr * np.sin(ntheta), axis=1)
        daNdtheta = np.sum(
            -self.cN * self.n * np.sin(ntheta) + self.sN * self.n * np.cos(ntheta),
            axis=1,
        )

        dZdtheta = self.get_dZdtheta(theta, aN, daNdtheta)

        dZdr = self.get_dZdr(theta, dZ0dr, daNdr)

        dRdtheta = self.get_dRdtheta(theta, aN, daNdtheta)

        dRdr = self.get_dRdr(theta, shift, daNdr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_dZdtheta(self, theta, aN, daNdtheta):
        """
        Calculates the derivatives of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        aN: ArrayLike
            aN for those theta points
        daNdtheta: ArrayLike
            Derivative of aN at theta w.r.t theta
        Returns
        -------
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        """
        return aN * np.cos(theta) + daNdtheta * np.sin(theta)

    def get_dZdr(self, theta, dZ0dr, daNdr):
        """
        Calculates the derivatives of `Z(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Derivative in midplane elevation w.r.t r
        daNdr : ArrayLike
            Derivative of aN w.r.t r
        Returns
        -------
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """
        return dZ0dr + daNdr * np.sin(theta)

    def get_dRdtheta(self, theta, aN, daNdtheta):
        """
        Calculates the derivatives of `R(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on
        aN: ArrayLike
            aN for those theta points
        daNdtheta: ArrayLike
            Derivative of aN at theta w.r.t theta
        Returns
        -------
        dRdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        """

        return -aN * np.sin(theta) + daNdtheta * np.cos(theta)

    def get_dRdr(self, theta, shift, daNdr):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        shift : Float
            Derivative in major radius w.r.t r
        daNdr : ArrayLike
            Derivative of aN w.r.t r
        Returns
        -------
        dRdr : Array
            Derivative of `R` w.r.t `r`
        """
        return shift + daNdr * np.cos(theta)

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of FourierGENE fits

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

        aN = np.sum(
            self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
            axis=1,
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
        super(LocalGeometryFourierGENE, self).__init__(default_fourier_gene_inputs())
