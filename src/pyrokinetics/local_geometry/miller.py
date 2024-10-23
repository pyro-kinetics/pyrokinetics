import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint

from ..typing import ArrayLike
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
    r"""Local equilibrium representation defined as in: `Phys Plasmas, Vol 5,
    No 4, April 1998 Miller et al <https://doi.org/10.1063/1.872666>`_

    .. math::
        \begin{align}
        R_s(r, \theta) &= R_0(r) + r \cos[\theta + (\sin^{-1}\delta) \sin(\theta)] \\
        Z_s(r, \theta) &= Z_0(r) + r \kappa \sin(\theta) \\
        r &= (\max(R) - \min(R)) / 2
        \end{align}

    Data stored in a CleverDict Object

    Attributes
    ----------
    psi_n : Float
        Normalised poloidal flux :math:`\psi_n=\psi_{surface}/\psi_{LCFS}`
    rho : Float
        Minor radius :math:`\rho=r/a`
    a_minor : Float
        Minor radius [m] :math:`a` (if 2D equilibrium exists)
    Rmaj : Float
        Normalised major radius :math:`R_{maj}/a`
    Z0 : Float
        Normalised vertical position of midpoint (Zmid / a_minor)
    f_psi : Float
        Torodial field function
    B0 : Float
        Normalising magnetic field :math:`B_0 = f / R_{maj}` [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field to B0 :math:`B_{unit} =
        \frac{q}{r}\frac{\partial\psi}{\partial r}` [T]
    dpsidr : Float
        :math:`\frac{\partial \psi}{\partial r}`
    q : Float
        Safety factor :math:`q`
    shat : Float
        Magnetic shear :math:`\hat{s} = \frac{\rho}{q}\frac{\partial q}{\partial\rho}`
    beta_prime : Float
        Pressure gradient :math:`\beta'=\frac{8\pi 10^{-7}}{B_0^2}
        \frac{\partial p}{\partial\rho}`
    kappa : Float
        Elongation :math:`\kappa`
    delta : Float
        Triangularity :math:`\delta`
    s_kappa : Float
        Shear in Elongation :math:`\frac{\rho}{\kappa}
        \frac{\partial\kappa}{\partial\rho}`
    s_delta : Float
        Shear in Triangularity :math:`\frac{\rho}{\sqrt{1-\delta^2}}
        \frac{\partial\delta}{\partial\rho}`
    shift : Float
        Shafranov shift :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`
    dZ0dr : Float
        Shear in midplane elevation

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

    d2Rdtheta2 : Array
        Second derivative of fitted :math:`R` w.r.t :math:`\theta`
    d2Rdrdtheta : Array
        Derivative of fitted :math:`R` w.r.t :math:`r` and :math:`\theta`
    d2Zdtheta2 : Array
        Second derivative of fitted :math:`Z` w.r.t :math:`\theta`
    d2Zdrdtheta : Array
        Derivative of fitted :math:`Z` w.r.t :math:`r` and :math:`\theta`

    """

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
        theta: Optional[Array] = None,
        overwrite_dpsidr: bool = True,
        kappa: float = DEFAULT_INPUTS["kappa"],
        s_kappa: float = DEFAULT_INPUTS["s_kappa"],
        delta: float = DEFAULT_INPUTS["delta"],
        s_delta: float = DEFAULT_INPUTS["s_delta"],
        shift: float = DEFAULT_INPUTS["shift"],
        dZ0dr: float = DEFAULT_INPUTS["dZ0dr"],
    ):
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
    ) -> ShapeParams:
        r"""
        Calculates Miller shaping coefficients

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
        shift
           Initial guess for the Shafranov shift
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

        params = cls.ShapeParams(kappa=kappa, delta=delta, shift=shift)
        return cls._fit_params_to_b_poloidal(
            theta, b_poloidal, params, Rmaj, Zmid, rho, dpsidr, verbose=verbose
        )

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: ShapeParams
    ) -> Tuple[Array, Array]:
        R = R0 + rho * np.cos(theta + np.arcsin(params.delta) * np.sin(theta))
        Z = Z0 + params.kappa * rho * np.sin(theta)
        return R, Z

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: ShapeParams
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

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
    ) -> np.ndarray:
        r"""
        Calculates the second derivatives of :math:`R(r, \theta)` and :math:`Z(r,
        \theta)` w.r.t :math:`r` and :math:`\theta`, used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

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
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.s_kappa)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.s_delta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_d2Zdtheta2(self, theta):
        r"""Calculates the second derivative of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdtheta2 on

        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of :math:`Z` w.r.t :math:`\theta`

        """

        return self.kappa * self.rho * -np.sin(theta)

    def get_d2Zdrdtheta(self, theta, s_kappa):
        r"""Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdrdtheta on
        s_kappa : Float
            Shear in Elongation

        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """

        return np.cos(theta) * (self.kappa + s_kappa * self.kappa)

    def get_d2Rdtheta2(self, theta):
        r"""Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdtheta2 on

        Returns
        -------
        d2Rdtheta2 : Array
            Second derivative of :math:`R` w.r.t :math:`\theta`
        """

        x = np.arcsin(self.delta)

        return -self.rho * (
            ((1 + x * np.cos(theta)) ** 2) * np.cos(theta + x * np.sin(theta))
            - x * np.sin(theta) * np.sin(theta + x * np.sin(theta))
        )

    def get_d2Rdrdtheta(self, theta, s_delta):
        r"""Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Rdrdtheta on
        s_delta : Float
            Shear in Triangularity

        Returns
        -------
        d2Rdrdtheta : Array
            Second derivative of :math:`R` w.r.t :math:`r` and :math:`\theta`
        """
        x = np.arcsin(self.delta)

        return (
            -(1 + x * np.cos(theta)) * np.sin(theta + x * np.sin(theta))
            - s_delta * np.cos(theta) * np.sin(theta + x * np.sin(theta))
            - s_delta
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
