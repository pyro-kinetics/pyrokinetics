from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pint
from scipy.integrate import simpson
from typing_extensions import Self

from ..units import Array, Float, PyroQuantity
from ..units import ureg as units
from .local_geometry import Derivatives, LocalGeometry, SecondDerivatives, ShapeParams

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

DEFAULT_MXH_MOMENTS = 4


@dataclasses.dataclass(frozen=True)
class MXHShapeParams(ShapeParams):
    kappa: Float
    cn: Array
    sn: Array
    dcndr: Array
    dsndr: Array
    shift: Float = 0.0
    s_kappa: Float = 0.0
    dZ0dr: Float = 0.0
    FIT_PARAMS: ClassVar[List[str]] = ["shift", "s_kappa", "dZ0dr", "dcndr", "dsndr"]


class LocalGeometryMXH(LocalGeometry):

    kappa: Float
    r"""Elongation. :math:`\kappa`"""

    s_kappa: Float
    r"""Shear in elongation.

    :math:`\frac{\rho}{\kappa}\frac{\partial\kappa}{\partial\rho}`
    """

    shift: Float
    r"""Shafranov shift.

    :math:`\Delta = \frac{\partial R_{maj}}{\partial r}`
    """

    dZ0dr: Float
    """Shear in midplane elevation"""

    cn: Array
    """Cosine moments of thetaR"""

    sn: Array
    """Sine moments of thetaR"""

    dcndr: Array
    r"""Shear in cosine moments :math:`\partial c_n/\partial r`"""

    dsndr: Array
    r"""Shear in sine moments :math:`\partial s_n/\partial r`"""

    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        "cn": np.zeros(DEFAULT_MXH_MOMENTS),
        "sn": np.zeros(DEFAULT_MXH_MOMENTS),
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "MXH"

    ShapeParams: ClassVar[Type] = MXHShapeParams

    def __init__(
        self,
        psi_n: Float,
        rho: Float,
        Rmaj: Float,
        Z0: Float,
        a_minor: Float,
        Fpsi: Float,
        FF_prime: Float,
        B0: Float,
        q: Float,
        shat: Float,
        beta_prime: Float,
        kappa: Float,
        s_kappa: Float,
        shift: Float,
        dZ0dr: Float,
        cn: Array,
        sn: Array,
        bt_ccw: int = -1,
        ip_ccw: int = -1,
        dpsidr: Optional[Float] = None,
        theta: Optional[Array] = None,
        dcndr: Optional[Array] = None,
        dsndr: Optional[Array] = None,
        delta: Optional[Float] = None,
        s_delta: Optional[Float] = None,
        zeta: Optional[Float] = None,
        s_zeta: Optional[Float] = None,
    ):
        r"""Miller eXtended Harmonic (MXH) local equilibrium representation

        Defined as in `PPCF 63 (2021) 012001 (5pp)
        <https://doi.org/10.1088/1361-6587/abc63b>`_

        .. math::

            \begin{align}
            R(r, \theta) &= R_{major}(r) + r \cos(\theta_R) \\
            Z(r, \theta) &= Z_0(r) + r \kappa(r) \sin(\theta) \\
            \theta_R &= \theta + c_0(r) + \sum_{n=1}^N [
                c_n(r) \cos(n \theta) + s_n(r) \sin(n \theta)] \\
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
        kappa
            Elongation :math:`\kappa`
        s_kappa
            Shear in elongation
            :math:`\frac{\rho}{\kappa} \frac{\partial\kappa}{\partial\rho}`
        shift
            Shafranov shift :math:`\Delta=\frac{\partial R_{maj}}{\partial r}`
        dZ0dr
            Shear in midplane elevation
        cn
            Cosine moments of thetaR
        sn
            Sine moments of thetaR
        bt_ccw
            +1 if :math:`B_\theta` is counter-clockwise, -1 otherwise.
        ip_ccw
            +1 if the plasma current is counter-clockwise, -1 otherwise.
        dpsidr
            :math:`\frac{\partial \psi}{\partial r}`. Should be provided when
            building from a global equilibrium or another local geometry.
        theta
            Grid of :math:`\theta` on which to evaluate the flux surface.
        dcndr
            Shear in cosine moments :math:`\partial c_n/\partial r`
        dsndr
            Shear in sine moments :math:`\partial s_n/\partial r`
        """
        if dcndr is None:
            dcndr = np.zeros_like(cn)
        if dsndr is None:
            dsndr = np.zeros_like(sn)

        # Error checking on array inputs
        arrays = {"cn": cn, "sn": cn, "dcndr": dcndr, "dsndr": dsndr}
        for name, array in arrays.items():
            if array.ndim != 1:
                msg = f"LocalGeometryFourierMXH input {name} should be 1D"
                raise ValueError(msg)
        if len(set(len(array) for array in arrays.values())) != 1:
            msg = "Array inputs to LocalGeometryMXH must have same length"
            raise ValueError(msg)

        # Check if units are needed on the array inputs
        if hasattr(rho, "units"):
            if not hasattr(cn, "units"):
                cn *= units.dimensionless
            if not hasattr(sn, "units"):
                sn *= units.dimensionless
            if not hasattr(dcndr, "units"):
                dcndr *= 1.0 / rho.units
            if not hasattr(dsndr, "units"):
                dsndr *= 1.0 / rho.units

        # If delta/s_delta/zeta/s_zeta set, these should overwrite array values
        # We assume the inputs have the correct units
        if delta is not None:
            sn[1] = np.arcsin(delta)
        if zeta is not None:
            sn[2] = -zeta
        if s_delta is not None:
            dsndr[1] = s_delta / np.sqrt(1 - np.sin(sn[1]) ** 2) / rho
        if s_zeta is not None:
            dsndr[2] = -s_zeta / rho

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
            shift=shift,
            dZ0dr=dZ0dr,
            cn=cn,
            sn=sn,
            dcndr=dcndr,
            dsndr=dsndr,
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
        n_moments: int = DEFAULT_MXH_MOMENTS,
    ) -> MXHShapeParams:
        r"""Calculate MXH shaping coefficients.

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
        n_moments
            Number of Fourier components to include.
        """

        kappa = (max(Z) - min(Z)) / (2 * rho)

        Zmid = (max(Z) + min(Z)) / 2

        Zind_upper = np.argmax(Z)

        R_upper = R[Zind_upper]

        normalised_height = (Z - Zmid) / (kappa * rho)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        normalised_radius = (R - Rmaj) / rho

        normalised_radius = np.where(
            np.isclose(normalised_radius, 1.0, atol=1e-4), 1.0, normalised_radius
        )
        normalised_radius = np.where(
            np.isclose(normalised_radius, -1.0, atol=1e-4), -1.0, normalised_radius
        )

        thetaR = np.arccos(normalised_radius)

        theta = np.where(R < R_upper, np.pi - theta, theta)
        theta = np.where((R >= R_upper) & (Z <= Zmid), 2 * np.pi + theta, theta)
        thetaR = np.where(Z <= Zmid, 2 * np.pi - thetaR, thetaR)

        # Ensure first point is close to 0 rather than 2pi
        if theta[0] > np.pi:
            theta[0] += -2 * np.pi
            thetaR[0] += -2 * np.pi

        theta_diff = thetaR - theta

        theta_dimensionless = units.Quantity(theta).magnitude
        theta_diff_dimensionless = units.Quantity(theta_diff).magnitude

        ntheta = np.outer(np.arange(n_moments), theta_dimensionless)
        cn = (
            simpson(
                theta_diff_dimensionless * np.cos(ntheta), x=theta_dimensionless, axis=1
            )
            / np.pi
        )
        sn = (
            simpson(
                theta_diff_dimensionless * np.sin(ntheta), x=theta_dimensionless, axis=1
            )
            / np.pi
        )

        length_unit = rho.units if isinstance(rho, PyroQuantity) else 1.0

        params = MXHShapeParams(
            kappa=kappa,
            cn=cn * units.dimensionless,
            sn=sn * units.dimensionless,
            dcndr=np.zeros(n_moments) / length_unit,
            dsndr=np.zeros(n_moments) / length_unit,
            shift=shift,
        )
        fits = cls._fit_params_to_b_poloidal(
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

        # Force dsndr[0] which has no impact on flux surface
        fits.dsndr[0] *= 0.0
        return fits

    @property
    def n_moments(self):
        return len(self.cn)

    @property
    def delta(self):
        return np.sin(self.sn[1])

    @delta.setter
    def delta(self, value):
        self.sn[1] = np.arcsin(value)

    @property
    def s_delta(self):
        return self.dsndr[1] * np.sqrt(1 - self.delta**2) * self.rho

    @s_delta.setter
    def s_delta(self, value):
        self.dsndr[1] = value / np.sqrt(1 - self.delta**2) / self.rho

    @property
    def zeta(self):
        return -self["sn"][2]

    @zeta.setter
    def zeta(self, value):
        self["sn"][2] = -value

    @property
    def s_zeta(self):
        return -self.dsndr[2] * self.rho

    @s_zeta.setter
    def s_zeta(self, value):
        self.dsndr[2] = -value / self.rho

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: MXHShapeParams
    ) -> Tuple[Array, Array]:

        thetaR = cls._thetaR(theta, params)
        R = R0 + rho * np.cos(thetaR)
        Z = Z0 + params.kappa * rho * np.sin(theta)
        return R, Z

    @staticmethod
    def _thetaR(theta: Array, params: MXHShapeParams) -> Array:
        """Poloidal angle used in definition of R"""
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.cn)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)
        thetaR = theta + np.sum(
            (params.cn * np.cos(ntheta) + params.sn * np.sin(ntheta)),
            axis=1,
        )
        return thetaR

    @staticmethod
    def _dthetaR_dr(theta: Array, params: MXHShapeParams) -> Array:
        r""":math:`\theta` derivative of poloidal angle used in R"""
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.cn)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)
        dthetaR_dr = np.sum(
            (params.dcndr * np.cos(ntheta) + params.dsndr * np.sin(ntheta)),
            axis=1,
        )
        return dthetaR_dr

    @staticmethod
    def _dthetaR_dtheta(theta: Array, params: MXHShapeParams) -> Array:
        theta = units.Quantity(theta).magnitude  # strip units
        n_moments = len(params.cn)
        n = np.arange(n_moments, dtype=float)
        ntheta = np.outer(theta, n)
        dthetaR_dtheta = 1.0 + np.sum(
            (-params.cn * n * np.sin(ntheta) + params.sn * n * np.cos(ntheta)),
            axis=1,
        )
        return dthetaR_dtheta

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: MXHShapeParams
    ) -> Derivatives:
        thetaR = cls._thetaR(theta, params)
        dthetaR_dr = cls._dthetaR_dr(theta, params)
        dthetaR_dtheta = cls._dthetaR_dtheta(theta, params)
        dZdtheta = cls._dZdtheta(theta, rho, params.kappa)
        dZdr = cls._dZdr(theta, params.dZ0dr, params.kappa, params.s_kappa)
        dRdtheta = cls._dRdtheta(thetaR, dthetaR_dtheta, rho)
        dRdr = cls._dRdr(thetaR, dthetaR_dr, rho, params.shift)
        return Derivatives(dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr)

    @staticmethod
    def _dZdtheta(theta: Array, rho: Float, kappa: Float) -> Array:
        return kappa * rho * np.cos(theta)

    @staticmethod
    def _dZdr(theta: Array, dZ0dr: Float, kappa: Float, s_kappa: Float) -> Array:
        return dZ0dr + kappa * np.sin(theta) * (1.0 + s_kappa)

    @staticmethod
    def _dRdtheta(thetaR: Array, dthetaR_dtheta: Array, rho: Float) -> Array:
        return -rho * np.sin(thetaR) * dthetaR_dtheta

    @staticmethod
    def _dRdr(thetaR: Array, dthetaR_dr: Array, rho: Float, shift: Float) -> Array:
        return shift + np.cos(thetaR) - rho * np.sin(thetaR) * dthetaR_dr

    def _d2thetaR_dtheta2(self, theta: Array) -> Array:
        if hasattr(theta, "magnitude"):
            theta = theta.m

        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        d2thetaR_dtheta2 = -np.sum(
            (n**2 * (self.cn * np.cos(ntheta) + self.sn * np.sin(ntheta))),
            axis=1,
        )

        return d2thetaR_dtheta2

    def _d2thetaR_drdtheta(self, theta: Array) -> Array:
        if hasattr(theta, "magnitude"):
            theta = theta.m

        n = np.arange(self.n_moments)
        ntheta = np.outer(theta, n)

        d2thetaR_drdtheta = np.sum(
            n * (self.dsndr * np.cos(ntheta) - self.dcndr * np.sin(ntheta)),
            axis=1,
        )

        return d2thetaR_drdtheta

    def get_RZ_second_derivatives(self, theta: Array) -> SecondDerivatives:
        r"""Second derivatives of :math:`R(r,\theta)` and :math:`Z(r,\theta)`

        Overwrites version in the base class to improve efficiency.
        """

        params = self._shape_params
        thetaR = self._thetaR(theta, params)
        dthetaR_dr = self._dthetaR_dr(theta, params)
        dthetaR_dtheta = self._dthetaR_dtheta(theta, params)
        d2thetaR_drdtheta = self._d2thetaR_drdtheta(theta)
        d2thetaR_dtheta2 = self._d2thetaR_dtheta2(theta)

        d2Zdtheta2 = self._d2Zdtheta2(theta)
        d2Zdrdtheta = self._d2Zdrdtheta(theta)
        d2Rdtheta2 = self._d2Rdtheta2(thetaR, dthetaR_dtheta, d2thetaR_dtheta2)
        d2Rdrdtheta = self._d2Rdrdtheta(
            thetaR, dthetaR_dr, dthetaR_dtheta, d2thetaR_drdtheta
        )

        return SecondDerivatives(
            d2Rdtheta2=d2Rdtheta2,
            d2Rdrdtheta=d2Rdrdtheta,
            d2Zdtheta2=d2Zdtheta2,
            d2Zdrdtheta=d2Zdrdtheta,
        )

    def _d2Zdtheta2(self, theta: Array) -> Array:
        return -self.kappa * self.rho * np.sin(theta)

    def _d2Zdrdtheta(self, theta: Array) -> Array:
        return self.kappa * np.cos(theta) * (1 + self.s_kappa)

    def _d2Rdtheta2(
        self, thetaR: Array, dthetaR_dtheta: Array, d2thetaR_dtheta2: Array
    ) -> Array:
        return -self.rho * np.sin(thetaR) * d2thetaR_dtheta2 - self.rho * (
            dthetaR_dtheta**2
        ) * np.cos(thetaR)

    def _d2Rdrdtheta(
        self,
        thetaR: Array,
        dthetaR_dr: Array,
        dthetaR_dtheta: Array,
        d2thetaR_drdtheta: Array,
    ) -> Array:
        return -dthetaR_dtheta * np.sin(thetaR) - self.rho * (
            np.sin(thetaR) * d2thetaR_drdtheta
            + dthetaR_dr * dthetaR_dtheta * np.cos(thetaR)
        )

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
        # TODO: Need to change dcndr and dsndr to pyro norms
        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "cn": units.dimensionless,
            "sn": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
            "dcndr": norms.lref**-1,
            "dsndr": norms.lref**-1,
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
        a shortcut if fitting to a plain Miller geometry, as MXH is a superset
        of the Miller geometry.

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
                FF_prime=local_geometry.FF_prime,
                B0=local_geometry.B0,
                Z0=local_geometry.Z0,
                q=local_geometry.q,
                shat=local_geometry.shat,
                beta_prime=local_geometry.beta_prime,
                kappa=local_geometry.kappa,
                s_kappa=local_geometry.s_kappa,
                shift=local_geometry.shift,
                dZ0dr=local_geometry.dZ0dr,
                cn=np.zeros(DEFAULT_MXH_MOMENTS),
                sn=np.zeros(DEFAULT_MXH_MOMENTS),
                ip_ccw=local_geometry.ip_ccw,
                bt_ccw=local_geometry.bt_ccw,
                dpsidr=local_geometry.dpsidr,
                theta=local_geometry.theta,
                delta=local_geometry.delta,
                s_delta=local_geometry.s_delta,
            )

            if show_fit or axes is not None:
                result.plot_equilibrium_to_local_geometry_fit(
                    show_fit=show_fit, axes=axes
                )
            return result

        return super().from_local_geometry(
            local_geometry, verbose=verbose, show_fit=show_fit, axes=axes, **kwargs
        )
