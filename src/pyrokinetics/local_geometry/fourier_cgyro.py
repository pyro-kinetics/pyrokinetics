import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint
from numpy.typing import NDArray
from scipy.integrate import simpson

from ..constants import pi
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

    aR: Array
    """Cosine moments of :math:`R`."""

    aZ: Array
    """Cosine moments of :math:`Z`."""

    bR: Array
    """Sine moments of :math:`R`."""

    bZ: Array
    """Sine moments of :math:`Z`."""

    daRdr: Array
    """Derivative of ``aR`` w.r.t :math:`r`."""

    daZdr: Array
    """Derivative of ``aZ`` w.r.t :math:`r`."""

    dbRdr: Array
    """Derivative of ``bR`` w.r.t :math:`r`."""

    dbZdr: Array
    """Derivative of ``bZ`` w.r.t :math:`r`."""

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
        bt_ccw: int = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: int = DEFAULT_INPUTS["ip_ccw"],
        dpsidr: Optional[Float] = None,
        theta: Optional[Array] = None,
        aR: Array = DEFAULT_INPUTS["aR"],
        aZ: Array = DEFAULT_INPUTS["aZ"],
        bR: Array = DEFAULT_INPUTS["bR"],
        bZ: Array = DEFAULT_INPUTS["bZ"],
        daRdr: Optional[Array] = None,
        daZdr: Optional[Array] = None,
        dbRdr: Optional[Array] = None,
        dbZdr: Optional[Array] = None,
    ):
        r"""Local equilibrium representation.

        Ddefined as in: `PPCF 51 (2009) 105009 J Candy
        <https://doi.org/10.1088/0741-3335/51/10/105009>`_

        .. math::
            \begin{align}
            R(r,\theta) &= \frac{1}{2} aR_0(r) + \sum_{n=1}^N [
                aR_n(r)\cos(n\theta) + bR_n(r)\sin(n\theta)] \\
            Z(r,\theta) &= \frac{1}{2} aZ_0(r) + \sum_{n=1}^N [
                aZ_n(r)\cos(n\theta) + bZ_n(r)\sin(n\theta)] \\
            r &= \frac{\max(R) - \min(R)}{2}
            \end{align}

        Parameters
        ----------
        psi_n
            Normalised poloidal flux :math:`\psi_n=\psi_{surface}/\psi_{LCFS}`
        rho
            Minor radius :math:`\rho=r/a`
        Rmaj
            Normalised major radius :math:`R_{maj}/a`
        Z0
            Normalised vertical position of midpoint (Zmid / a_minor)
        a_minor
            Minor radius of the LCFS (if 2D equilibrium exists)
        Fpsi
            Torodial field function :math:`F`
        FF_prime
            :math:`F` multiplied by its derivative w.r.t :math:`r`.
        B0
            Normalising magnetic field :math:`B_0 = f / R_{maj}`
        q
            Safety factor :math:`q`
        shat
            Magnetic shear
            :math:`\hat{s}=\frac{\rho}{q}\frac{\partial q}{\partial\rho}`
        beta_prime
            Pressure gradient :math:`\beta'=\frac{8\pi 10^{-7}}{B_0^2}
            \frac{\partial p}{\partial\rho}`
        bt_ccw
            +1 if :math:`B_\theta` is counter-clockwise, -1 otherwise.
        ip_ccw
            +1 if the plasma current is counter-clockwise, -1 otherwise.
        dpsidr
            :math:`\frac{\partial \psi}{\partial r}`. Should be provided when
            building from a global equilibrium or another local geometry.
        theta
            Grid of :math:`\theta` on which to evaluate the flux surface.
        aR
            Cosine moments of :math:`R`.
        aZ
            Cosine moments of :math:`Z`.
        bR
            Sine moments of :math:`R`.
        bZ
            Sine moments of :math:`Z`.
        daRdr
            Derivative of ``aR`` w.r.t :math:`r`.
        daZdr
            Derivative of ``aZ`` w.r.t :math:`r`.
        dbRdr
            Derivative of ``bR`` w.r.t :math:`r`.
        dbZdr
            Derivative of ``bZ`` w.r.t :math:`r`.
        """
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
            bt_ccw=bt_ccw,
            ip_ccw=ip_ccw,
            dpsidr=dpsidr,
            theta=theta,
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
    ) -> FourierCGYROShapeParams:
        r"""Calculate Fourier CGYRO shaping coefficients.

        Parameters
        ----------
        R
            R for the given flux surface.
        Z
            Z for the given flux surface.
        b_poloidal
            :math:`B_\theta` for the given flux surface.
        Rmaj
            Major radius of the centre of the flux surface.
        Z0
            Vertical height of the centre of the flux surface.
        rho
            Normalised minor radius of the flux surface.
        dpsidr
            :math:`\partial \psi / \partial r`.
        verbose
            Print fitting data if ``True``.
        n_moments
            Number of Fourier terms to include.
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
        params = FourierCGYROShapeParams(
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

        return cls._fit_params_to_b_poloidal(
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

    @property
    def n_moments(self):
        return len(self.aR)

    @classmethod
    def _flux_surface(
        cls,
        theta: Array,
        R0: Float,
        Z0: Float,
        rho: Float,
        params: FourierCGYROShapeParams,
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
        cls, theta: Array, rho: Float, params: FourierCGYROShapeParams
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

    def _d2Zdtheta2(self, theta: Array) -> Array:
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        return np.sum(
            -(n**2) * (self.aZ * np.cos(ntheta) + self.bZ * np.sin(ntheta)),
            axis=1,
        )

    def _d2Zdrdtheta(self, theta: Array) -> Array:
        # TODO Numpy outer doesn't work on pint=0.23 quantities
        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        return np.sum(
            n * (self.dbZdr * np.cos(ntheta) - self.daZdr * np.sin(ntheta)), axis=1
        )

    def _d2Rdtheta2(self, theta: Array) -> Array:
        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        return np.sum(
            -(n**2) * (self.aR * np.cos(ntheta) + self.bR * np.sin(ntheta)),
            axis=1,
        )

    def _d2Rdrdtheta(self, theta: Array) -> Array:
        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        return np.sum(
            n * (self.dbRdr * np.cos(ntheta) - self.daRdr * np.sin(ntheta)), axis=1
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
