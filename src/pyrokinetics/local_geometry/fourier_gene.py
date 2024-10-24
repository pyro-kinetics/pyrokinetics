import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint
from scipy.integrate import simpson

from ..typing import ArrayLike
from ..units import Array, Float
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, ShapeParams

DEFAULT_GENE_MOMENTS = 32


@dataclasses.dataclass(frozen=True)
class FourierGENEShapeParams(ShapeParams):
    cN: Array
    sN: Array
    dcNdr: Array
    dsNdr: Array
    FIT_PARAMS: ClassVar[List[str]] = ["dcNdr", "dsNdr"]


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

    local_geometry: ClassVar[str] = "FourierGENE"

    ShapeParams: ClassVar[Type] = FourierGENEShapeParams

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
        bt_ccw: int = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: int = DEFAULT_INPUTS["ip_ccw"],
        dpsidr: Optional[Float] = None,
        theta: Optional[Array] = None,
        cN: Array = DEFAULT_INPUTS["cN"],
        sN: Array = DEFAULT_INPUTS["sN"],
        dcNdr: Optional[Array] = None,
        dsNdr: Optional[Array] = None,
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
            bt_ccw=bt_ccw,
            ip_ccw=ip_ccw,
            dpsidr=dpsidr,
            theta=theta,
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

        R_0 = Rmaj
        Z0 = Z0

        length_unit = getattr(rho, "units", 1.0)

        # Ensure we start at index where Z[0] = 0
        if not np.isclose(Z[0], Z0):
            Z0_index = np.argmin(abs(Z - Z0))
            R = np.roll(R.m, -Z0_index) * length_unit
            Z = np.roll(Z.m, -Z0_index) * length_unit
            b_poloidal = np.roll(b_poloidal.m, -Z0_index) * b_poloidal.units

        R_diff = R - R_0
        Z_diff = Z - Z0
        aN = np.sqrt((R_diff) ** 2 + (Z_diff) ** 2)

        if R[0] < R_0:
            if Z[1] > Z[0]:
                roll_sign = -1
            else:
                roll_sign = 1
        else:
            if Z[1] > Z[0]:
                roll_sign = 1
            else:
                roll_sign = -1

        dot_product = (
            R_diff * np.roll(R_diff.m, roll_sign)
            + Z_diff * np.roll(Z_diff.m, roll_sign)
        ) * length_unit
        magnitude = np.sqrt(R_diff**2 + Z_diff**2)
        arc_angle = (
            dot_product / (magnitude * np.roll(magnitude.m, roll_sign)) / length_unit
        )

        theta0 = np.arcsin(Z_diff[0] / aN[0])

        theta_diff = np.arccos(arc_angle)

        if Z[1] > Z[0]:
            theta = np.cumsum(theta_diff) - theta_diff[0]
        else:
            theta = -np.cumsum(theta_diff) - theta_diff[0]

        theta += theta0

        theta_start = 0
        if not np.isclose(theta[0], 0.0) and theta[0] > 0:
            theta = np.insert(theta, 0, theta[-1] - 2 * np.pi)
            R = np.insert(R, 0, R[-1])
            Z = np.insert(Z, 0, Z[-1])
            b_poloidal = np.insert(b_poloidal, 0, b_poloidal[-1])
            theta_start = 1

        if not np.isclose(theta[-1], 2 * np.pi) and theta[-1] < 2 * np.pi:
            theta = np.append(theta, theta[theta_start] + 2 * np.pi)
            R = np.append(R, R[theta_start])
            Z = np.append(Z, Z[theta_start])
            b_poloidal = np.append(b_poloidal, b_poloidal[theta_start])

        if len(R) < n_moments * 4:
            theta_resolution_scale = 4
        else:
            theta_resolution_scale = 1

        theta_new = (
            np.linspace(
                0, 2 * np.pi, len(theta) * theta_resolution_scale, endpoint=True
            )
            * units.radians
        )

        # Interpolate to evenly spaced theta, as this improves the fit
        theta_new = np.linspace(0, 2 * np.pi, len(theta))
        R = np.interp(theta_new, theta, R)
        Z = np.interp(theta_new, theta, Z)
        b_poloidal = np.interp(theta_new, theta, b_poloidal)
        theta = theta_new

        aN = np.sqrt((R - R_0) ** 2 + (Z - Z0) ** 2)

        theta_dimensionless = units.Quantity(theta).magnitude
        ntheta = np.outer(np.arange(n_moments), theta_dimensionless)

        cN = (
            simpson(aN.m * np.cos(ntheta), x=theta_dimensionless, axis=1)
            / np.pi
            * length_unit
        )
        sN = (
            simpson(aN.m * np.sin(ntheta), x=theta_dimensionless, axis=1)
            / np.pi
            * length_unit
        )

        cN[0] *= 0.5
        sN[0] *= 0.5

        params = cls.ShapeParams(
            cN=cN, sN=sN, dcNdr=np.zeros(n_moments), dsNdr=np.zeros(n_moments)
        )
        params.dcNdr[0] = 1.0
        result = cls._fit_params_to_b_poloidal(
            theta,
            b_poloidal,
            params,
            Rmaj,
            Z0,
            rho,
            dpsidr,
            verbose=verbose,
            max_cost=0.1,
        )

        result.dsNdr[0] = 0.0
        return result

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

    def get_d2Zdtheta2(self, theta, aN, daNdtheta, d2aNdtheta2):
        return (
            daNdtheta * np.cos(theta)
            - aN * np.sin(theta)
            + d2aNdtheta2 * np.sin(theta)
            + daNdtheta * np.cos(theta)
        )

    def get_d2Zdrdtheta(self, theta, daNdr, d2aNdrdtheta):
        return d2aNdrdtheta * np.sin(theta) + daNdr * np.cos(theta)

    def get_d2Rdtheta2(self, theta, aN, daNdtheta, d2aNdtheta2):
        return (
            -daNdtheta * np.sin(theta)
            - aN * np.cos(theta)
            + d2aNdtheta2 * np.cos(theta)
            - daNdtheta * np.sin(theta)
        )

    def get_d2Rdrdtheta(self, theta, daNdr, d2aNdrdtheta):
        return d2aNdrdtheta * np.cos(theta) - daNdr * np.sin(theta)

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
        return {
            "cN": norms.lref,
            "sN": norms.lref,
            "aN": norms.lref,
            "daNdtheta": norms.lref,
            "dcNdr": units.dimensionless,
            "dsNdr": units.dimensionless,
            "daNdr": units.dimensionless,
        }
