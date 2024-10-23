import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint
from numpy.typing import NDArray
from scipy.integrate import simpson

from ..constants import pi
from ..typing import ArrayLike
from ..units import Array, Float
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, ShapeParams

DEFAULT_CGYRO_MOMENTS = 16


@dataclasses.dataclass(frozen=True)
class FourierCGYROShapeParams(ShapeParams):
    aR: Array
    aZ: Array
    bR: Array
    bZ: Array
    daRdr: Array
    daZdr: Array
    dbRdr: Array
    dbZdr: Array
    FIT_PARAMS: ClassVar[List[str]] = ["daRdr", "daZdr", "dbRdr", "dbZdr"]


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

    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "aR": np.array([3.0, 0.5, *[0.0] * (DEFAULT_CGYRO_MOMENTS - 2)]),
        "aZ": np.array([0.0, 0.5, *[0.0] * (DEFAULT_CGYRO_MOMENTS - 2)]),
        "bR": np.zeros(DEFAULT_CGYRO_MOMENTS),
        "bZ": np.zeros(DEFAULT_CGYRO_MOMENTS),
        "daRdr": np.zeros(DEFAULT_CGYRO_MOMENTS),
        "daZdr": np.zeros(DEFAULT_CGYRO_MOMENTS),
        "dbRdr": np.zeros(DEFAULT_CGYRO_MOMENTS),
        "dbZdr": np.zeros(DEFAULT_CGYRO_MOMENTS),
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "FourierCGYRO"

    ShapeParams: ClassVar[Type] = FourierCGYROShapeParams

    def __init__(
        self,
        psi_n: Float = DEFAULT_INPUTS["psi_n"],
        rho: Float = DEFAULT_INPUTS["rho"],
        Rmaj: Float = DEFAULT_INPUTS["Rmaj"],
        Z0: Float = DEFAULT_INPUTS["Z0"],
        a_minor: Float = DEFAULT_INPUTS["a_minor"],
        Fpsi: Float = DEFAULT_INPUTS["Fpsi"],
        FF_prime: Float = DEFAULT_INPUTS["FF_prime"],
        B0: Float = DEFAULT_INPUTS["B0"],
        q: Float = DEFAULT_INPUTS["q"],
        shat: Float = DEFAULT_INPUTS["shat"],
        beta_prime: Float = DEFAULT_INPUTS["beta_prime"],
        dpsidr: Float = DEFAULT_INPUTS["dpsidr"],
        bt_ccw: int = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: int = DEFAULT_INPUTS["ip_ccw"],
        theta: Optional[Array] = None,
        overwrite_dpsidr: bool = True,
        aR: Array = DEFAULT_INPUTS["aR"],
        aZ: Array = DEFAULT_INPUTS["aZ"],
        bR: Array = DEFAULT_INPUTS["bR"],
        bZ: Array = DEFAULT_INPUTS["bZ"],
        daRdr: Optional[Array] = None,
        daZdr: Optional[Array] = None,
        dbRdr: Optional[Array] = None,
        dbZdr: Optional[Array] = None,
    ):
        if daRdr is None:
            daRdr = np.zeros_like(aR)
        if daZdr is None:
            daZdr = np.zeros_like(aZ)
        if dbRdr is None:
            dbRdr = np.zeros_like(bR)
        if dbZdr is None:
            dbZdr = np.zeros_like(bZ)

        # Error checking on array inputs
        arrays = {
            "aR": aR,
            "aZ": aZ,
            "bR": bR,
            "bZ": bZ,
            "daRdr": daRdr,
            "daZdr": daZdr,
            "dbRdr": dbRdr,
            "dbZdr": dbZdr,
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                msg = f"LocalGeometryFourierCGYRO input {name} should be 1D"
                raise ValueError(msg)
        if len(set(len(array) for array in arrays.values())) != 1:
            msg = "Array inputs to LocalGeometryFourierCGYRO must have same length"
            raise ValueError(msg)

        self._init_with_shape_params(
            psi_n=psi_n,
            rho=rho,
            Rmaj=Rmaj,
            Z0=Z0,
            a_minor=a_minor,
            Fpsi=Fpsi,
            FF_prime=FF_prime,
            B0=B0,
            q=q,
            shat=shat,
            beta_prime=beta_prime,
            dpsidr=dpsidr,
            bt_ccw=bt_ccw,
            ip_ccw=ip_ccw,
            theta=theta,
            overwrite_dpsidr=overwrite_dpsidr,
            aR=aR,
            aZ=aZ,
            bR=bR,
            bZ=bZ,
            daRdr=daRdr,
            daZdr=daZdr,
            dbRdr=dbRdr,
            dbZdr=dbZdr,
        )

    @classmethod
    def _fit_shape_params(
        cls,
        R: Array,
        Z: Array,
        b_poloidal: Array,
        Rmaj: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        verbose: bool = False,
        n_moments: int = DEFAULT_CGYRO_MOMENTS,
    ) -> ShapeParams:
        r"""
        Calculates FourierCGYRO shaping coefficients

        Parameters
        ----------
        R
            R for the given flux surface
        Z
            Z for the given flux surface
        b_poloidal
            :math:`B_\theta` for the given flux surface
        Rmaj
            Major radius of the centre of the flux surface
        Z0
            Vertical height of the centre of the flux surface
        rho
            Normalised minor radius of the flux surface
        dpsidr
            :math:`\partial \psi / \partial r`
        verbose
            Controls verbosity
        n_moments
            Number of Fourier terms to include
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

        # Interpolate to evenly spaced theta, as this improves the fit
        theta_new = np.linspace(0, 2 * np.pi, len(theta))
        R = np.interp(theta_new, theta, R)
        Z = np.interp(theta_new, theta, Z)
        b_poloidal = np.interp(theta_new, theta, b_poloidal)
        theta = theta_new

        # TODO Numpy outer doesn't work on pint=0.23 quantities
        ntheta = np.outer(np.arange(n_moments), theta)
        aR = (
            simpson(
                R.magnitude * np.cos(ntheta),
                x=theta,
                axis=1,
            )
            / np.pi
        )
        aZ = (
            simpson(
                Z.magnitude * np.cos(ntheta),
                x=theta,
                axis=1,
            )
            / np.pi
        )
        bR = (
            simpson(
                R.magnitude * np.sin(ntheta),
                x=theta,
                axis=1,
            )
            / np.pi
        )
        bZ = (
            simpson(
                Z.magnitude * np.sin(ntheta),
                x=theta,
                axis=1,
            )
            / np.pi
        )

        aR[0] *= 0.5
        aZ[0] *= 0.5

        # Set up starting parameters
        params = cls.ShapeParams(
            aR=aR * length_unit,
            aZ=aZ * length_unit,
            bR=bR * length_unit,
            bZ=bZ * length_unit,
            daRdr=np.zeros(n_moments),
            daZdr=np.zeros(n_moments),
            dbRdr=np.zeros(n_moments),
            dbZdr=np.zeros(n_moments),
        )
        # Roughly a cosine wave
        params.daRdr[1] = 1.0
        # Rougly a sine wave
        params.dbZdr[1] = 1.0

        return cls._fit_params(
            theta, b_poloidal, params, Rmaj, Z0, rho, dpsidr, verbose=verbose
        )

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    @property
    def n_moments(self):
        return len(self.aR)

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: ShapeParams
    ) -> Tuple[Array, Array]:
        del R0, Z0, rho  # unused variables
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.aR)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)

        R = np.sum(
            params.aR * np.cos(ntheta) + params.bR * np.sin(ntheta),
            axis=1,
        )
        Z = np.sum(
            params.aZ * np.cos(ntheta) + params.bZ * np.sin(ntheta),
            axis=1,
        )
        return R, Z

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: ShapeParams
    ) -> Derivatives:
        del rho  # unused variable
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.aR)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)
        dZdtheta = cls._dZdtheta(n, ntheta, params.aZ, params.bZ)
        dZdr = cls._dZdr(ntheta, params.daZdr, params.dbZdr)
        dRdtheta = cls._dRdtheta(n, ntheta, params.aR, params.bR)
        dRdr = cls._dRdr(ntheta, params.daRdr, params.dbRdr)
        return Derivatives(dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr)

    @staticmethod
    def _dZdtheta(n: NDArray, ntheta: NDArray, aZ: Array, bZ: Array) -> Array:
        return np.sum(
            n * (aZ * np.sin(ntheta) + bZ * np.cos(ntheta)),
            axis=1,
        )

    @staticmethod
    def _dZdr(ntheta: NDArray, daZdr: Array, dbZdr: Array) -> Array:
        return np.sum(daZdr * np.cos(ntheta) + dbZdr * np.sin(ntheta), axis=1)

    @staticmethod
    def _dRdtheta(n: NDArray, ntheta: NDArray, aR: Array, bR: Array) -> Array:
        return np.sum(
            n * (-aR * np.sin(ntheta) + bR * np.cos(ntheta)),
            axis=1,
        )

    @staticmethod
    def _dRdr(ntheta: NDArray, daRdr: Array, dbRdr: Array) -> Array:
        return np.sum(daRdr * np.cos(ntheta) + dbRdr * np.sin(ntheta), axis=1)

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

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
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
