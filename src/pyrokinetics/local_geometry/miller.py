from typing import Any, ClassVar, Dict, NamedTuple, Tuple

import numpy as np

from ..typing import ArrayLike
from ..units import ureg as units
from .local_geometry import LocalGeometry, shape_params, Float, Array


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

    @shape_params(fit=["s_kappa", "s_delta", "shift", "dZ0dr"])
    class ShapeParams(NamedTuple):
        kappa: float
        delta: float
        s_kappa: float = 0.0
        s_delta: float = 0.0
        shift: float = 0.0
        dZ0dr: float = 0.0

    local_geometry: ClassVar[str] = "Miller"

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
        kappa: float = DEFAULT_INPUTS["kappa"],
        s_kappa: float = DEFAULT_INPUTS["s_kappa"],
        delta: float = DEFAULT_INPUTS["delta"],
        s_delta: float = DEFAULT_INPUTS["s_delta"],
        shift: float = DEFAULT_INPUTS["shift"],
        dZ0dr: float = DEFAULT_INPUTS["dZ0dr"],
    ):
        super().__init__(
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
        )
        self.kappa = kappa
        self.s_kappa = s_kappa
        self.delta = delta
        self.s_delta = s_delta
        self.shift = shift
        self.dZ0dr = dZ0dr

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates Miller shaping coefficients from R, Z and b_poloidal

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
        shift : Float
            Initial guess for shafranov shift
        """

        kappa = (max(Z) - min(Z)) / (2 * self.rho)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = (self.Rmaj - R_upper) / self.rho

        normalised_height = (Z - Zmid) / (kappa * self.rho)

        self.kappa = kappa
        self.delta = delta
        self.Z0 = Zmid

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

        params = self.ShapeParams(kappa=kappa, delta=delta, shift=shift)
        fits = self.fit_params(
            theta, b_poloidal, params, self.Rmaj, self.Z0, self.rho, self.dpsidr
        )
        self.s_kappa = fits.s_kappa
        self.s_delta = fits.s_delta
        self.shift = fits.shift
        self.dZ0dr = fits.dZ0dr

    def get_flux_surface(
        self,
        theta: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generates :math:`(R,Z)` of a flux surface given a set of Miller fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface

        Returns
        -------
        :math:`R` : Array
            :math:`R(\theta)` values for this flux surface (if not normalised then in [m])
        :math:`Z` : Array
            :math:`Z(\theta)` Values for this flux surface (if not normalised then in [m])
        """

        R = self.Rmaj + self.rho * np.cos(theta + np.arcsin(self.delta) * np.sin(theta))
        Z = self.Z0 + self.kappa * self.rho * np.sin(theta)

        return R, Z

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
    ) -> Tuple[Array, Array, Array, Array]:
        dZdtheta = cls._dZdtheta(theta, rho, params.kappa)
        dZdr = cls._dZdr(theta, params.dZ0dr, params.kappa, params.s_kappa)
        dRdtheta = cls._dRdtheta(theta, rho, params.delta)
        dRdr = cls._dRdr(theta, params.shift, params.delta, params.s_delta)
        return dRdtheta, dRdr, dZdtheta, dZdr

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

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        r"""Calculates the derivatives of :math:`R(r, \theta)` and :math:`Z(r,
        \theta)` w.r.t :math:`r` and :math:`\theta`, used in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use ``params = [s_kappa_fit,
            s_delta_fit, shift_fit, dZ0dr_fit]`` when calculating
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

        if params is not None:
            s_kappa = params[0]
            s_delta = params[1]
            shift = params[2]
            dZ0dr = params[3]
        else:
            s_kappa = self.s_kappa
            s_delta = self.s_delta
            shift = self.shift
            dZ0dr = self.dZ0dr

        dZdtheta = self.get_dZdtheta(theta)
        dZdr = self.get_dZdr(theta, dZ0dr, s_kappa)
        dRdtheta = self.get_dRdtheta(theta)
        dRdr = self.get_dRdr(theta, shift, s_delta)

        return dRdtheta, dRdr, dZdtheta, dZdr

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

    def get_dZdtheta(self, theta):
        r"""Calculates the derivatives of :math:`Z(r, theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on

        Returns
        -------
        dZdtheta : Array
            Derivative of :math:`Z` w.r.t :math:`\theta`

        """

        return self.kappa * self.rho * np.cos(theta)

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

    def get_dZdr(self, theta, dZ0dr, s_kappa):
        r"""Calculates the derivatives of :math:`Z(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Shear in midplane elevation
        s_kappa : Float
            Shear in Elongation

        Returns
        -------
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`
        """

        return dZ0dr + self.kappa * np.sin(theta) + s_kappa * self.kappa * np.sin(theta)

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

    def get_dRdtheta(self, theta):
        r"""Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdtheta on

        Returns
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        """
        x = np.arcsin(self.delta)

        return -self.rho * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

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

    def get_dRdr(self, theta, shift, s_delta):
        r"""Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdr on
        shift : Float
            Shafranov shift
        s_delta : Float
            Shear in Triangularity

        Returns
        -------
        dRdr : Array
            Derivative of :math:`R` w.r.t :math:`r`
        """
        x = np.arcsin(self.delta)

        return (
            shift
            + np.cos(theta + x * np.sin(theta))
            - np.sin(theta + x * np.sin(theta)) * np.sin(theta) * s_delta
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

    def _generate_shape_coefficients_units(self, norms):
        """
        Units for Miller parameters
        """

        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "delta": units.dimensionless,
            "s_delta": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
        }

    @staticmethod
    def _shape_coefficient_names():
        """
        List of shape coefficient names used for printing
        """
        return [
            "kappa",
            "s_kappa",
            "delta",
            "s_delta",
            "shift",
            "dZ0dr",
        ]
