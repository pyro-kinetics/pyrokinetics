from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint
from scipy.optimize import least_squares  # type: ignore
from typing_extensions import Self

from ..constants import pi
from ..typing import ArrayLike
from ..units import Array, Float
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, ShapeParams

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@dataclasses.dataclass(frozen=True)
class MillerTurnbullShapeParams(ShapeParams):
    kappa: Float
    delta: Float
    zeta: Float
    s_kappa: Float = 0.0
    s_delta: Float = 0.0
    s_zeta: Float = 0.0
    shift: Float = 0.0
    dZ0dr: Float = 0.0
    FIT_PARAMS: ClassVar[List[str]] = ["s_kappa", "s_delta", "s_zeta", "shift", "dZ0dr"]


class LocalGeometryMillerTurnbull(LocalGeometry):
    r"""Local equilibrium representation defined as in:

    - `Phys. Plasmas, Vol. 5, No. 4, April 1998 Miller et al
      <https://doi.org/10.1063/1.872666>`_

    - `Physics of Plasmas 6, 1113 (1999); Turnbull et al
      <https://doi.org/10.1063/1.873380>`_

    .. math::
        \begin{align}
        R(r, \theta) &= R_{major}(r) + r \cos(\theta + \arcsin(\delta(r) \sin(\theta)) \\
        Z(r, \theta) &= Z_0(r) + r \kappa(r) \sin(\theta + \zeta(r) \sin(2\theta)) \\
        r &= (\max(R) - \min(R)) / 2
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
        Magnetic shear :math:`r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta = 2 \mu_0 \partial p \partial \rho 1/B0^2`

    kappa : Float
        Elongation
    delta : Float
        Triangularity
    zeta : Float
        Squareness
    s_kappa : Float
        Shear in Elongation :math:`r/\kappa \partial \kappa/\partial r`
    s_delta : Float
        Shear in Triangularity :math:`r/\sqrt{1 - \delta^2} \partial \delta/\partial r`
    s_zeta : Float
        Shear in Squareness :math:`r/ \partial \zeta/\partial r`
    shift : Float
        Shafranov shift
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
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "MillerTurnbull"

    ShapeParams: ClassVar[Type] = MillerTurnbullShapeParams

    def __init__(
        self,
        psi_n: Float = DEFAULT_INPUTS["psi_n"],
        rho: Float = DEFAULT_INPUTS["rho"],
        Rmaj: Float = DEFAULT_INPUTS["Rmaj"],
        Z0: Float = DEFAULT_INPUTS["Z0"],
        a_minor: float = DEFAULT_INPUTS["a_minor"],
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
        zeta: Float = DEFAULT_INPUTS["zeta"],
        s_zeta: Float = DEFAULT_INPUTS["s_zeta"],
        shift: Float = DEFAULT_INPUTS["shift"],
        dZ0dr: Float = DEFAULT_INPUTS["dZ0dr"],
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
            bt_ccw=bt_ccw,
            ip_ccw=ip_ccw,
            dpsidr=dpsidr,
            theta=theta,
            kappa=kappa,
            s_kappa=s_kappa,
            delta=delta,
            s_delta=s_delta,
            zeta=zeta,
            s_zeta=s_zeta,
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
        Calculates MillerTurnbull shaping coefficients

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

        kappa = (max(Z) - min(Z)) / (2 * rho)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = Rmaj / rho - R_upper / rho

        normalised_height = (Z - Zmid) / (kappa * rho)

        R_pi4 = Rmaj + rho * np.cos(pi / 4 + np.arcsin(delta) * np.sin(pi / 4))

        R_gt_0 = np.where(Z > 0, R, 0.0)
        Z_pi4 = Z[np.argmin(np.abs(R_gt_0 - R_pi4))]

        zeta = np.arcsin((Z_pi4 - Zmid) / (kappa * rho)) - pi / 4

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta_guess = np.arcsin(normalised_height)
        theta = cls._get_theta_from_squareness(theta_guess, Z, Zmid, kappa, rho, zeta)

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        params = cls.ShapeParams(kappa=kappa, delta=delta, zeta=zeta, shift=shift)
        return cls._fit_params_to_b_poloidal(
            theta, b_poloidal, params, Rmaj, Z0, rho, dpsidr, verbose=verbose
        )

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: ShapeParams
    ) -> Tuple[Array, Array]:
        R = R0 + rho * np.cos(theta + np.arcsin(params.delta) * np.sin(theta))
        Z = Z0 + params.kappa * rho * np.sin(theta + params.zeta * np.sin(2 * theta))
        return R, Z

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: ShapeParams
    ) -> Derivatives:
        dZdtheta = cls._dZdtheta(theta, rho, params.kappa, params.zeta)
        dZdr = cls._dZdr(
            theta,
            params.dZ0dr,
            params.kappa,
            params.s_kappa,
            params.zeta,
            params.s_zeta,
        )
        dRdtheta = cls._dRdtheta(theta, rho, params.delta)
        dRdr = cls._dRdr(theta, params.shift, params.delta, params.s_delta)
        return Derivatives(dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr)

    @staticmethod
    def _dZdtheta(theta: Array, rho: Float, kappa: Float, zeta: Float) -> Array:
        return (
            kappa
            * rho
            * (1 + 2 * zeta * np.cos(2 * theta))
            * np.cos(theta + zeta * np.sin(2 * theta))
        )

    @staticmethod
    def _dZdr(
        theta: Array,
        dZ0dr: Float,
        kappa: Float,
        s_kappa: Float,
        zeta: Float,
        s_zeta: Float,
    ) -> Array:
        return (
            dZ0dr
            + kappa * np.sin(theta + zeta * np.sin(2 * theta))
            + s_kappa * kappa * np.sin(theta + zeta * np.sin(2 * theta))
            + kappa
            * s_zeta
            * np.sin(2 * theta)
            * np.cos(theta + zeta * np.sin(2 * theta))
        )

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
        r"""Calculates the second derivatives of :math:`R(r, \theta)`
        and :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`,
        used in geometry terms

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
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.s_kappa, self.s_zeta)
        d2Rdtheta2 = self.get_d2Rdtheta2(theta)
        d2Rdrdtheta = self.get_d2Rdrdtheta(theta, self.s_delta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_d2Zdtheta2(self, theta):
        r"""
        Calculates the second derivative of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdtheta2 on

        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of :math:`Z` w.r.t :math:`\theta`
        """

        return (
            self.kappa
            * self.rho
            * (
                -4
                * self.zeta
                * np.sin(2 * theta)
                * np.cos(theta + self.zeta * np.sin(2 * theta))
                - ((1 + 2 * self.zeta * np.cos(2 * theta)) ** 2)
                * np.sin(theta + self.zeta * np.sin(2 * theta))
            )
        )

    def get_d2Zdrdtheta(self, theta, s_kappa, s_zeta):
        r"""
        Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r`
        and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate d2Zdrdtheta on
        s_kappa : Float
            Shear in Elongation
        s_zeta : Float
            Shear in Squareness

        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """

        return (
            (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.cos(theta + self.zeta * np.sin(2 * theta))
            * (s_kappa * self.kappa + self.kappa)
            + self.kappa
            * np.cos(theta + self.zeta * np.sin(2 * theta))
            * 2
            * np.cos(2 * theta)
            * s_zeta
            - self.kappa
            * s_zeta
            * np.sin(2 * theta)
            * (1 + 2 * self.zeta * np.cos(2 * theta))
            * np.sin(theta + self.zeta * np.sin(2 * theta))
        )

    def get_d2Rdtheta2(self, theta):
        r"""
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

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
        r"""
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r`
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
    def _get_theta_from_squareness(cls, theta, Z, Z0, kappa, rho, zeta):
        """
        Performs least square fitting to get theta for a given flux surface from the equation for Z

        Parameters
        ----------
        theta

        Returns
        -------

        """
        kwargs = {
            "Z": Z,
            "Z0": Z0,
            "kappa": kappa,
            "rho": rho,
            "zeta": zeta,
        }

        fits = least_squares(
            cls._minimise_theta_from_squareness, theta.m, kwargs=kwargs
        )

        return fits.x * theta.units

    @staticmethod
    def _minimise_theta_from_squareness(theta, Z, Z0, kappa, rho, zeta):
        """
        Calculate theta in MillerTurnbull by re-arranging equation for Z and changing theta such that the function gets
        minimised
        Parameters
        ----------
        theta : Array
            Guess for theta
        Returns
        -------
        sum_diff : Array
            Minimisation difference
        """
        normalised_height = (Z - Z0) / (kappa * rho)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta_func = np.arcsin(normalised_height)
        sum_diff = np.sum(np.abs(theta_func - theta - zeta * np.sin(2 * theta)))
        return units.Quantity(sum_diff).magnitude

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
        del norms  # unused
        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "delta": units.dimensionless,
            "s_delta": units.dimensionless,
            "zeta": units.dimensionless,
            "s_zeta": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
        }

    @classmethod
    def from_local_geometry(
        cls,
        local_geometry: Self,
        verbose: bool = False,
        show_fit: bool = False,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
        **kwargs,
    ) -> Self:
        r"""Create a new instance from a :class:`LocalGeometry` or subclass.

        Gradients in shaping parameters are fitted from the poloidal field.
        Unlike :meth:`LocalGeometry.from_local_geometry`, this method performs
        a shortcut if fitting to a plain Miller geometry, as Miller-Turnbull is
        a superset of the Miller geometry.

        Parameters
        ----------
        local_geometry
            ``LocalGeometry`` or subclass to fit to.
        verbose
            Print more data to terminal when performing a fit.
        show_fit
            If ``True``, plots the resulting fit using Matplotlib.
        axes
            Axes on which to plot if ``show_fit`` is ``True``. If supplied, the
            plot will not be shown, and it is up to the user to call
            ``plt.show()``, ``plt.savefig()`` or similar.  If ``axes`` is
            ``None``, a new set of axes are created and the plot is shown to
            the caller.
        """

        if local_geometry.local_geometry == "Miller":
            result = cls(
                psi_n=local_geometry.psi_n,
                rho=local_geometry.rho,
                Rmaj=local_geometry.Rmaj,
                a_minor=local_geometry.a_minor,
                Fpsi=local_geometry.Fpsi,
                B0=local_geometry.B0,
                Z0=local_geometry.Z0,
                q=local_geometry.q,
                shat=local_geometry.shat,
                beta_prime=local_geometry.beta_prime,
                ip_ccw=local_geometry.ip_ccw,
                bt_ccw=local_geometry.bt_ccw,
                dpsidr=local_geometry.dpsidr,
                theta=local_geometry.theta,
                kappa=local_geometry.kappa,
                s_kappa=local_geometry.s_kappa,
                delta=local_geometry.delta,
                s_delta=local_geometry.s_delta,
                shift=local_geometry.shift,
                dZ0dr=local_geometry.dZ0dr,
            )

            if show_fit or axes is not None:
                result.plot_equilibrium_to_local_geometry_fit(
                    show_fit=show_fit, axes=axes
                )
            return result

        return super().from_local_geometry(
            local_geometry, verbose=verbose, axes=axes, show_fit=show_fit, **kwargs
        )
