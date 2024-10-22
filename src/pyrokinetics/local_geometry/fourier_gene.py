from typing import Any, ClassVar, Dict, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson

from ..typing import ArrayLike
from ..units import Array, Float
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, shape_params

DEFAULT_GENE_MOMENTS = 32


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

    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "cN": np.array([0.5, *[0.0] * (DEFAULT_GENE_MOMENTS - 1)]),
        "sN": np.zeros(DEFAULT_GENE_MOMENTS),
        "dcNdr": np.array([1.0, *[0.0] * (DEFAULT_GENE_MOMENTS - 1)]),
        "dsNdr": np.zeros(DEFAULT_GENE_MOMENTS),
        **LocalGeometry.DEFAULT_INPUTS,
    }

    @shape_params(fit=["dcNdr", "dsNdr"])
    class ShapeParams(NamedTuple):
        cN: NDArray[np.float64]
        sN: NDArray[np.float64]
        dcNdr: NDArray[np.float64]
        dsNdr: NDArray[np.float64]

    local_geometry: ClassVar[str] = "FourierGENE"

    def __init__(
        self,
        psi_n: float = DEFAULT_INPUTS["psi_n"],
        rho: float = DEFAULT_INPUTS["rho"],
        Rmaj: float = DEFAULT_INPUTS["Rmaj"],
        Z0: float = DEFAULT_INPUTS["Z0"],
        a_minor: float = DEFAULT_INPUTS["a_minor"],
        Fpsi: float = DEFAULT_INPUTS["Fpsi"],
        FF_prime: float = DEFAULT_INPUTS["FF_prime"],
        B0: float = DEFAULT_INPUTS["B0"],
        q: float = DEFAULT_INPUTS["q"],
        shat: float = DEFAULT_INPUTS["shat"],
        beta_prime: float = DEFAULT_INPUTS["beta_prime"],
        dpsidr: float = DEFAULT_INPUTS["dpsidr"],
        bt_ccw: float = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: float = DEFAULT_INPUTS["ip_ccw"],
        theta: Optional[NDArray[np.float64]] = None,
        overwrite_dpsidr: bool = True,
        cN: NDArray[np.float64] = DEFAULT_INPUTS["cN"],
        sN: NDArray[np.float64] = DEFAULT_INPUTS["sN"],
        dcNdr: Optional[NDArray[np.float64]] = None,
        dsNdr: Optional[NDArray[np.float64]] = None,
    ):
        if dcNdr is None:
            dcNdr = np.zeros_like(cN)
            dcNdr[0] = 1.0
        if dsNdr is None:
            dsNdr = np.zeros_like(sN)

        # Error checking on array inputs
        arrays = {"cN": cN, "sN": sN, "dcNdr": dcNdr, "dsNdr": dsNdr}
        for name, array in arrays.items():
            if array.ndim != 1:
                msg = f"LocalGeometryFourierGENE input {name} should be 1D"
                raise ValueError(msg)
        if len(set(len(array) for array in arrays.values())) != 1:
            msg = "Array inputs to LocalGeometryFourierGENE must have same length"
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
            cN=cN,
            sN=sN,
            dcNdr=dcNdr,
            dsNdr=dsNdr,
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
        n_moments: int = DEFAULT_GENE_MOMENTS,
    ) -> ShapeParams:
        r"""
        Calculates FourierGENE shaping coefficients

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

        R_major = Rmaj
        Zmid = Z0

        R_diff = R - R_major
        Z_diff = Z - Zmid

        length_unit = R_major.units

        dot_product = (
            R_diff * np.roll(R_diff.m, 1) + Z_diff * np.roll(Z_diff.m, 1)
        ) * length_unit
        magnitude = np.sqrt(R_diff**2 + Z_diff**2)
        arc_angle = dot_product / (magnitude * np.roll(magnitude.m, 1)) / length_unit

        theta_diff = np.arccos(arc_angle)

        if Z[1] > Z[0]:
            theta = np.cumsum(theta_diff) - theta_diff[0]
        else:
            theta = -np.cumsum(theta_diff) - theta_diff[0]

        # Interpolate to evenly spaced theta, as this improves the fit
        theta_new = np.linspace(0, 2 * np.pi, len(theta))
        R = np.interp(theta_new, theta, R)
        Z = np.interp(theta_new, theta, Z)
        b_poloidal = np.interp(theta_new, theta, b_poloidal)
        theta = theta_new

        aN = np.sqrt((R - R_major) ** 2 + (Z - Zmid) ** 2)

        theta_dimensionless = units.Quantity(theta).magnitude
        ntheta = np.outer(np.arange(n_moments), theta_dimensionless)

        cN = simpson(aN.m * np.cos(ntheta), x=theta, axis=1) / np.pi * length_unit
        sN = simpson(aN.m * np.sin(ntheta), x=theta, axis=1) / np.pi * length_unit

        cN[0] *= 0.5
        sN[0] *= 0.5

        params = cls.ShapeParams(
            cN=cN, sN=sN, dcNdr=np.zeros(n_moments), dsNdr=np.zeros(n_moments)
        )
        params.dcNdr[0] = 1.0
        return cls._fit_params(theta, b_poloidal, params, Rmaj, Z0, rho, dpsidr)
        # TODO function previously also set the following:
        #
        # ntheta = np.outer(theta, self.n)
        #
        # self.aN = np.sum(
        #     self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
        #     axis=1,
        # )
        # self.daNdr = np.sum(
        #     self.dcNdr * np.cos(ntheta) + self.dsNdr * np.sin(ntheta),
        #     axis=1,
        # )
        # self.daNdtheta = np.sum(
        #     -self.cN * self.n * np.sin(ntheta) + self.sN * self.n * np.cos(ntheta),
        #     axis=1,
        # )

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

    @property
    def n_moments(self):
        return len(self.cN)

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: ShapeParams
    ) -> Tuple[Array, Array]:
        del rho  # unused variable
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.cN)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)
        aN = np.sum(
            params.cN * np.cos(ntheta) + params.sN * np.sin(ntheta),
            axis=1,
        )
        R = R0 + aN * np.cos(theta)
        Z = Z0 + aN * np.sin(theta)
        return R, Z

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: ShapeParams
    ) -> Derivatives:
        del rho  # unused variable
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.cN)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)

        aN = np.sum(
            params.cN * np.cos(ntheta) + params.sN * np.sin(ntheta),
            axis=1,
        )
        daNdr = np.sum(
            params.dcNdr * np.cos(ntheta) + params.dsNdr * np.sin(ntheta), axis=1
        )
        daNdtheta = np.sum(
            -params.cN * n * np.sin(ntheta) + params.sN * n * np.cos(ntheta),
            axis=1,
        )

        dZdtheta = cls._dZdtheta(theta, aN, daNdtheta)
        dZdr = cls._dZdr(theta, daNdr)
        dRdtheta = cls._dRdtheta(theta, aN, daNdtheta)
        dRdr = cls._dRdr(theta, daNdr)
        return Derivatives(dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr)

    @staticmethod
    def _dZdtheta(theta: Array, aN: Array, daNdtheta: Array) -> Array:
        return aN * np.cos(theta) + daNdtheta * np.sin(theta)

    @staticmethod
    def _dZdr(theta: Array, daNdr: Array) -> Array:
        return daNdr * np.sin(theta)

    @staticmethod
    def _dRdtheta(theta: Array, aN: Array, daNdtheta: Array) -> Array:
        return -aN * np.sin(theta) + daNdtheta * np.cos(theta)

    @staticmethod
    def _dRdr(theta: Array, daNdr: Array) -> Array:
        return daNdr * np.cos(theta)

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        """
        Calculates the derivatives of `R(r, \theta)` and `Z(r, \theta)` w.r.t `r` and `\theta`, used in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use params = [dcNdr[nmoments], dsNdr[nmoments] ] when calculating
            derivatives, otherwise will use object attributes
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
            dcNdr = self.dcNdr
            dsNdr = self.dsNdr
        else:
            dcNdr = params[: self.n_moments]
            dsNdr = params[self.n_moments :]

        if hasattr(theta, "magnitude"):
            theta = theta.m

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

        dZdr = self.get_dZdr(theta, daNdr)

        dRdtheta = self.get_dRdtheta(theta, aN, daNdtheta)

        dRdr = self.get_dRdr(theta, daNdr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
    ) -> np.ndarray:
        ntheta = np.outer(theta, self.n)

        aN = np.sum(
            self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
            axis=1,
        )
        daNdr = np.sum(
            self.dcNdr * np.cos(ntheta) + self.dsNdr * np.sin(ntheta), axis=1
        )
        daNdtheta = np.sum(
            -self.cN * self.n * np.sin(ntheta) + self.sN * self.n * np.cos(ntheta),
            axis=1,
        )
        d2aNdtheta2 = np.sum(
            -(self.n**2) * (self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta)),
            axis=1,
        )
        d2aNdrdtheta = np.sum(
            -self.n * self.dcNdr * np.sin(ntheta)
            + self.n * self.dsNdr * np.cos(ntheta),
            axis=1,
        )
        d2Zdtheta2 = self.get_d2Zdtheta2(theta, aN, daNdtheta, d2aNdtheta2)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, daNdr, d2aNdrdtheta)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta, aN, daNdtheta, d2aNdtheta2)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, daNdr, d2aNdrdtheta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

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

    def get_d2Zdtheta2(self, theta, aN, daNdtheta, d2aNdtheta2):

        return (
            daNdtheta * np.cos(theta)
            - aN * np.sin(theta)
            + d2aNdtheta2 * np.sin(theta)
            + daNdtheta * np.cos(theta)
        )

    def get_dZdr(self, theta, daNdr):
        """
        Calculates the derivatives of `Z(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daNdr : ArrayLike
            Derivative of aN w.r.t r
        Returns
        -------
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """
        return daNdr * np.sin(theta)

    def get_d2Zdrdtheta(self, theta, daNdr, d2aNdrdtheta):
        return d2aNdrdtheta * np.sin(theta) + daNdr * np.cos(theta)

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

    def get_d2Rdtheta2(self, theta, aN, daNdtheta, d2aNdtheta2):

        return (
            -daNdtheta * np.sin(theta)
            - aN * np.cos(theta)
            + d2aNdtheta2 * np.cos(theta)
            - daNdtheta * np.sin(theta)
        )

    def get_dRdr(self, theta, daNdr):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        daNdr : ArrayLike
            Derivative of aN w.r.t r
        Returns
        -------
        dRdr : Array
            Derivative of `R` w.r.t `r`
        """
        return daNdr * np.cos(theta)

    def get_d2Rdrdtheta(self, theta, daNdr, d2aNdrdtheta):
        return d2aNdrdtheta * np.cos(theta) - daNdr * np.sin(theta)

    def get_flux_surface(
        self,
        theta: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of FourierGENE fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface
        Returns
        -------
        R : Array
            R values for this flux surface ([m])
        Z : Array
            Z Values for this flux surface ([m])
        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        aN = np.sum(
            self.cN * np.cos(ntheta) + self.sN * np.sin(ntheta),
            axis=1,
        )

        R = self.Rmaj + aN * np.cos(theta)
        Z = self.Z0 + aN * np.sin(theta)

        return R, Z

    def _generate_shape_coefficients_units(self, norms):
        """
        Set shaping coefficients to lref normalisations
        """

        return {
            "cN": norms.lref,
            "sN": norms.lref,
            "aN": norms.lref,
            "daNdtheta": norms.lref,
            "dcNdr": units.dimensionless,
            "dsNdr": units.dimensionless,
            "daNdr": units.dimensionless,
        }

    @staticmethod
    def _shape_coefficient_names():
        """
        List of shape coefficient names used for printing
        """
        return [
            "cN",
            "sN",
            "dcNdr",
            "dsNdr",
        ]
