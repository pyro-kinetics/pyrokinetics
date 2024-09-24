from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.optimize import least_squares  # type: ignore

from ..constants import pi
from ..typing import ArrayLike
from ..units import ureg as units
from .local_geometry import LocalGeometry


class LocalGeometryFourierCGYRO(LocalGeometry):
    r"""Local equilibrium representation defined as in: `PPCF 51 (2009) 105009 J
    Candy <https://doi.org/10.1088/0741-3335/51/10/105009>`_

    FourierCGYRO

    .. math::
        \begin{align}
        R(r, \theta) &= 0.5 aR_0(r) + \sum_{n=1}^N [aR_n(r) \cos(n \theta) + bR_n(r) \sin(n \theta)] \\
        Z(r, \theta) &= 0.5 aZ_0(r) + \sum_{n=1}^N [aZ_n(r) \cos(n \theta) + bZ_n(r) \sin(n \theta)] \\
        r = (\max(R) - \min(R)) / 2
        \end{align}

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
        :math:`\frac{\partial \psi}{\partial r}`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear
    beta_prime : Float
        :math:`\beta = \beta a/L_p`

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
        Derivative of fitted :math:`R` w.r.t :math:`\theta`
    dRdr : Array
        Derivative of fitted :math:`R` w.r.t :math:`r`
    dZdtheta : Array
        Derivative of fitted :math:`Z` w.r.t :math:`\theta`
    dZdr : Array
        Derivative of fitted :math:`Z` w.r.t :math:`r`

    """

    DEFAULT_N_MOMENTS: ClassVar[int] = 16
    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "aR": np.array([3.0, 0.5, *[0.0] * (DEFAULT_N_MOMENTS - 2)]),
        "aZ": np.array([0.0, 0.5, *[0.0] * (DEFAULT_N_MOMENTS - 2)]),
        "bR": np.zeros(DEFAULT_N_MOMENTS),
        "bZ": np.zeros(DEFAULT_N_MOMENTS),
        "daRdr": np.zeros(DEFAULT_N_MOMENTS),
        "daZdr": np.zeros(DEFAULT_N_MOMENTS),
        "dbRdr": np.zeros(DEFAULT_N_MOMENTS),
        "dbZdr": np.zeros(DEFAULT_N_MOMENTS),
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "FourierCGYRO"

    def __init__(
        self,
        psi_n: float = DEFAULT_INPUTS["psi_n"],
        rho: float = DEFAULT_INPUTS["rho"],
        Rmaj: float = DEFAULT_INPUTS["Rmaj"],
        Z0: float = DEFAULT_INPUTS["Z0"],
        a_minor: float = DEFAULT_INPUTS["a_minor"],
        Fpsi: float = DEFAULT_INPUTS["Fpsi"],
        B0: float = DEFAULT_INPUTS["B0"],
        q: float = DEFAULT_INPUTS["q"],
        shat: float = DEFAULT_INPUTS["shat"],
        beta_prime: float = DEFAULT_INPUTS["beta_prime"],
        dpsidr: float = DEFAULT_INPUTS["dpsidr"],
        bt_ccw: float = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: float = DEFAULT_INPUTS["ip_ccw"],
        aR: NDArray[np.float64] = DEFAULT_INPUTS["aR"],
        aZ: NDArray[np.float64] = DEFAULT_INPUTS["aZ"],
        bR: NDArray[np.float64] = DEFAULT_INPUTS["bR"],
        bZ: NDArray[np.float64] = DEFAULT_INPUTS["bZ"],
        daRdr: Optional[NDArray[np.float64]] = None,
        daZdr: Optional[NDArray[np.float64]] = None,
        dbRdr: Optional[NDArray[np.float64]] = None,
        dbZdr: Optional[NDArray[np.float64]] = None,
    ):
        if daRdr is None:
            daRdr = np.zeros_like(aR)
        if daZdr is None:
            daZdr = np.zeros_like(aZ)
        if dbRdr is None:
            dbRdr = np.zeros_like(bR)
        if dbZdr is None:
            dbZdr = np.zeros_like(bZ)

        super().__init__(
            psi_n,
            rho,
            Rmaj,
            Z0,
            a_minor,
            Fpsi,
            B0,
            q,
            shat,
            beta_prime,
            dpsidr,
            bt_ccw,
            ip_ccw,
        )
        self.aR = aR
        self.aZ = aZ
        self.bR = bR
        self.bZ = bZ
        self.daRdr = daRdr
        self.daZdr = daZdr
        self.dbRdr = dbRdr
        self.dbZdr = dbZdr

        # Error checking on array inputs
        arrays = ("aR", "aZ", "bR", "bZ", "daRdr", "daZdr", "dbRdr", "dbZdr")
        for name in arrays:
            if self[name].ndim != 1:
                msg = f"LocalGeometryFourierCGYRO input {name} should be 1D"
                raise ValueError(msg)
        if len(set(len(self[x]) for x in arrays)) != 1:
            msg = "Array inputs to LocalGeometryFourierCGYRO must have same length"
            raise ValueError(msg)

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
            :math:`b_\theta` for the given flux surface
        verbose : Boolean
            Controls verbosity
        """
        length_unit = R.units
        field_unit = b_poloidal.units

        b_poloidal = np.roll(b_poloidal.m, -np.argmax(R)) * field_unit
        Z = np.roll(Z.m, -np.argmax(R)) * length_unit
        R = np.roll(R.m, -np.argmax(R)) * length_unit

        dR = R - np.roll(R.m, 1) * length_unit
        dZ = Z - np.roll(Z.m, 1) * length_unit

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

        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(self.n, theta)
        aR = (
            simpson(
                R.magnitude * np.cos(ntheta),
                x=self.theta,
                axis=1,
            )
            / np.pi
        )
        aZ = (
            simpson(
                Z.magnitude * np.cos(ntheta),
                x=self.theta,
                axis=1,
            )
            / np.pi
        )
        bR = (
            simpson(
                R.magnitude * np.sin(ntheta),
                x=self.theta,
                axis=1,
            )
            / np.pi
        )
        bZ = (
            simpson(
                Z.magnitude * np.sin(ntheta),
                x=self.theta,
                axis=1,
            )
            / np.pi
        )

        aR[0] *= 0.5
        aZ[0] *= 0.5

        self.Z0 = Zmid
        self.aR = aR * length_unit
        self.aZ = aZ * length_unit
        self.bR = bR * length_unit
        self.bZ = bZ * length_unit

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
                f"Least squares fitting in Fourier::_set_shape_coefficients failed with message : {fits.message}"
            )

        if verbose:
            print(
                f"FourierCGYRO :: Fit to Bpoloidal obtained with residual {fits.cost}"
            )

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in FourierCGYRO::_set_shape_coefficients is poor with residual of {fits.cost}"
            )

        self.daRdr = fits.x[0 : self.n_moments] * units.dimensionless
        self.daZdr = fits.x[self.n_moments : 2 * self.n_moments] * units.dimensionless
        self.dbRdr = (
            fits.x[2 * self.n_moments : 3 * self.n_moments] * units.dimensionless
        )
        self.dbZdr = fits.x[3 * self.n_moments :] * units.dimensionless

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    @property
    def n_moments(self):
        return len(self.aR)

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        r"""Calculates the derivatives of :math:`R(r, \theta)` and
        :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`, used
        in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use params = [daRdr[nmoments], daZdr[nmoments], dbRdr[nmoments], dbZdr[nmoments] ] when calculating
            derivatives, otherwise will use object attributes

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        dRdr : Array
            Derivative of :math:`R` w.r.t :math:`r`
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`

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

        if hasattr(theta, "units"):
            theta = theta.m

        dZdtheta = self.get_dZdtheta(theta)

        dZdr = self.get_dZdr(theta, daZdr, dbZdr)

        dRdtheta = self.get_dRdtheta(theta)

        dRdr = self.get_dRdr(theta, daRdr, dbRdr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
    ) -> np.ndarray:
        r"""Calculates the second derivatives of :math:`R(r, \theta)`
        and :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`,
        used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        normalised : Boolean
            Control whether or not to return normalised values

        Returns
        -------
        d2Rdtheta2 : Array
                        Second derivative of :math:`R` w.r.t :math:`\theta`
        d2Rdrdtheta : Array
                        Second derivative of :math:`R` w.r.t :math:`r` and :math:`\theta`
        d2Zdtheta2 : Array
                        Second derivative of :math:`Z` w.r.t :math:`\theta`
        d2Zdrdtheta : Array
                        Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`

        """

        d2Zdtheta2 = self.get_d2Zdtheta2(theta)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.daZdr, self.dbZdr)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.daRdr, self.dbRdr)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_dZdtheta(self, theta):
        r"""
        Calculates the derivatives of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on

        Returns
        -------
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`
        """
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-self.aZ * np.sin(ntheta) + self.bZ * np.cos(ntheta)),
            axis=1,
        )

    def get_d2Zdtheta2(self, theta):
        r"""
        Calculates the second derivative of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on

        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of :math:`Z` w.r.t :math:`\theta`
        """
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(theta, self.n)

        return np.sum(
            -(self.n**2) * (self.aZ * np.cos(ntheta) + self.bZ * np.sin(ntheta)),
            axis=1,
        )

    def get_dZdr(self, theta, daZdr, dbZdr):
        r"""
        Calculates the derivatives of :math:`Z(r, \theta)` w.r.t :math:`r`

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
            Derivative of :math:`Z` w.r.t :math:`r`
        """
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(theta, self.n)

        return np.sum(daZdr * np.cos(ntheta) + dbZdr * np.sin(ntheta), axis=1)

    def get_d2Zdrdtheta(self, theta, daZdr, dbZdr):
        r"""
        Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`

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
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-daZdr * np.sin(ntheta) + dbZdr * np.cos(ntheta)), axis=1
        )

    def get_dRdtheta(self, theta):
        r"""
        Calculates the derivatives of :math:`R(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-self.aR * np.sin(ntheta) + self.bR * np.cos(ntheta)),
            axis=1,
        )

    def get_d2Rdtheta2(self, theta):
        r"""
        Calculates the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on

        Returns
        -------
        d2Rdtheta2 : Array
            Second derivative of :math:`R` w.r.t :math:`\theta`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(
            -(self.n**2) * (self.aR * np.cos(ntheta) + self.bR * np.sin(ntheta)),
            axis=1,
        )

    def get_dRdr(self, theta, daRdr, dbRdr):
        r"""
        Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`r`

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
            Derivative of :math:`R` w.r.t :math:`r`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(daRdr * np.cos(ntheta) + dbRdr * np.sin(ntheta), axis=1)

    def get_d2Rdrdtheta(self, theta, daRdr, dbRdr):
        r"""
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r` and :math:`\theta`

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
            Second derivative of R w.r.t :math:`r` and :math:`\theta`
        """
        ntheta = np.outer(theta, self.n)

        return np.sum(
            self.n * (-daRdr * np.sin(ntheta) + dbRdr * np.cos(ntheta)), axis=1
        )

    def get_flux_surface(
        self,
        theta: ArrayLike,
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

        if hasattr(theta, "units"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        R = np.sum(
            self.aR * np.cos(ntheta) + self.bR * np.sin(ntheta),
            axis=1,
        )
        Z = np.sum(
            self.aZ * np.cos(ntheta) + self.bZ * np.sin(ntheta),
            axis=1,
        )

        return R, Z

    def _generate_shape_coefficients_units(self, norms):
        """
        Set shaping coefficients to lref normalisations
        """

        return {
            "aR": norms.lref,
            "bR": norms.lref,
            "aZ": norms.lref,
            "bZ": norms.lref,
            "daRdr": units.dimensionless,
            "dbRdr": units.dimensionless,
            "daZdr": units.dimensionless,
            "dbZdr": units.dimensionless,
        }

    @staticmethod
    def _shape_coefficient_names():
        """
        List of shape coefficient names used for printing
        """
        return [
            "aR",
            "bR",
            "aZ",
            "bZ",
            "daRdr",
            "dbRdr",
            "daZdr",
            "dbZdr",
        ]
