import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint

from ..units import Array, Float
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, ShapeParams


@dataclasses.dataclass(frozen=True)
class MillerShapeParams(ShapeParams):
    kappa: Float
    delta: Float
    s_kappa: Float = 0.0
    s_delta: Float = 0.0
    shift: Float = 0.0
    dZ0dr: Float = 0.0
    FIT_PARAMS: ClassVar[List[str]] = ["s_kappa", "s_delta", "shift", "dZ0dr"]


class LocalGeometryMiller(LocalGeometry):

    kappa: Float
    r"""Elongation. :math:`\kappa`"""

    delta: Float
    r"""Triangularity. :math:`\delta`"""

    s_kappa: Float
    r"""Shear in elongation.

    :math:`\frac{\rho}{\kappa}\frac{\partial\kappa}{\partial\rho}`
    """

    s_delta: Float
    r"""Shear in triangularity.

    :math:`\frac{\rho}{\sqrt{1-\delta^2}}\frac{\partial\delta}{\partial\rho}`
    """

    shift: Float
    r"""Shafranov shift.

    :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`
    """

    dZ0dr: Float
    """Shear in midplane elevation"""

    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "Miller"

    ShapeParams: ClassVar[Type] = MillerShapeParams

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
        kappa: Float = DEFAULT_INPUTS["kappa"],
        s_kappa: Float = DEFAULT_INPUTS["s_kappa"],
        delta: Float = DEFAULT_INPUTS["delta"],
        s_delta: Float = DEFAULT_INPUTS["s_delta"],
        shift: Float = DEFAULT_INPUTS["shift"],
        dZ0dr: Float = DEFAULT_INPUTS["dZ0dr"],
    ):
        r"""Local equilibrium representation.

        Defined in: `Phys Plasmas, Vol 5, No 4, April 1998 Miller et al
        <https://doi.org/10.1063/1.872666>`_

        .. math::

            \begin{align}
            R_s(r,\theta) &= R_0(r) + r\cos[
                \theta + (\sin^{-1}\delta)\sin(\theta)]\\
            Z_s(r,\theta) &= Z_0(r) + r\kappa \sin(\theta) \\
            r &= (\max(R) - \min(R)) / 2
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
        kappa
            Elongation :math:`\kappa`
        s_kappa
            Shear in elongation
            :math:`\frac{\rho}{\kappa} \frac{\partial\kappa}{\partial\rho}`
        delta
            Triangularity :math:`\delta`
        s_delta
            Shear in triangularity
            :math:`\frac{\rho}{\sqrt{1-\delta^2}}
            \frac{\partial\delta}{\partial\rho}`
        shift
            Shafranov shift :math:`\Delta=\frac{\partial R_{maj}}{\partial r}`
        dZ0dr
            Shear in midplane elevation
        """
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
            kappa=kappa,
            s_kappa=s_kappa,
            delta=delta,
            s_delta=s_delta,
            shift=shift,
            dZ0dr=dZ0dr,
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
        shift: Float = 0.0,
    ) -> MillerShapeParams:
        r"""Calculate Miller shaping coefficients.

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
        shift
           Initial guess for the Shafranov shift.
        """
        del Z0  # Ignore Z0 passed in, calculate from Z grid

        kappa = (max(Z) - min(Z)) / (2 * rho)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = (Rmaj - R_upper) / rho

        normalised_height = (Z - Zmid) / (kappa * rho)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        params = MillerShapeParams(kappa=kappa, delta=delta, shift=shift)
        return cls._fit_params_to_b_poloidal(
            theta, b_poloidal, params, Rmaj, Zmid, rho, dpsidr, verbose=verbose
        )

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: MillerShapeParams
    ) -> Tuple[Array, Array]:
        R = R0 + rho * np.cos(theta + np.arcsin(params.delta) * np.sin(theta))
        Z = Z0 + params.kappa * rho * np.sin(theta)
        return R, Z

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: MillerShapeParams
    ) -> Derivatives:
        dZdtheta = cls._dZdtheta(theta, rho, params.kappa)
        dZdr = cls._dZdr(theta, params.dZ0dr, params.kappa, params.s_kappa)
        dRdtheta = cls._dRdtheta(theta, rho, params.delta)
        dRdr = cls._dRdr(theta, params.shift, params.delta, params.s_delta)
        return Derivatives(dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr)

    @staticmethod
    def _dZdtheta(theta: Array, rho: Float, kappa: Float) -> Array:
        return kappa * rho * np.cos(theta)

    @staticmethod
    def _dZdr(theta: Array, dZ0dr: Float, kappa: Float, s_kappa: Float) -> Array:
        return dZ0dr + kappa * np.sin(theta) + s_kappa * kappa * np.sin(theta)

    @staticmethod
    def _dRdtheta(theta: Array, rho: Float, delta: Float) -> Array:
        x = np.arcsin(delta)
        return -rho * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    @staticmethod
    def _dRdr(theta: Array, shift: Float, delta: Float, s_delta: Float) -> Array:
        sin_theta = np.sin(theta)
        x = theta + np.arcsin(delta) * sin_theta
        return shift + np.cos(x) - np.sin(x) * sin_theta * s_delta

    def _d2Zdtheta2(self, theta: Array) -> Array:
        return self.kappa * self.rho * -np.sin(theta)

    def _d2Zdrdtheta(self, theta: Array) -> Array:
        return np.cos(theta) * (self.kappa + self.s_kappa * self.kappa)

    def _d2Rdtheta2(self, theta: Array) -> Array:
        x = np.arcsin(self.delta)
        return -self.rho * (
            ((1 + x * np.cos(theta)) ** 2) * np.cos(theta + x * np.sin(theta))
            - x * np.sin(theta) * np.sin(theta + x * np.sin(theta))
        )

    def _d2Rdrdtheta(self, theta: Array) -> Array:
        x = np.arcsin(self.delta)
        return (
            -(1 + x * np.cos(theta)) * np.sin(theta + x * np.sin(theta))
            - self.s_delta * np.cos(theta) * np.sin(theta + x * np.sin(theta))
            - self.s_delta
            * np.sin(theta)
            * (1 + x * np.cos(theta))
            * np.cos(theta + x * np.sin(theta))
        )

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
        del norms  # unused
        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "delta": units.dimensionless,
            "s_delta": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
        }
