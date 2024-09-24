from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.optimize import least_squares  # type: ignore
from typing_extensions import Self

from ..typing import ArrayLike
from ..units import PyroQuantity
from ..units import ureg as units
from .local_geometry import LocalGeometry

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class LocalGeometryMXH(LocalGeometry):
    r"""Local equilibrium representation defined as in: `PPCF 63 (2021) 012001
    (5pp) <https://doi.org/10.1088/1361-6587/abc63b>`_

    Miller eXtended Harmonic (MXH)

    .. math::
        \begin{align}
        R(r, \theta) &= R_{major}(r) + r \cos(\theta_R) \\
        Z(r, \theta) &= Z_0(r) + r \kappa(r) \sin(\theta) \\
        \theta_R &= \theta + c_0(r) + \sum_{n=1}^N [c_n(r) \cos(n \theta) + s_n(r) \sin(n \theta)] \\
        r &= (\max(R) - \min(R)) / 2
        \end{align}

    Data stored in a ordered dictionary

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
    s_kappa : Float
        Shear in Elongation :math:`r/\kappa \partial \kappa/\partial r`
    shift : Float
        Shafranov shift
    dZ0dr : Float
        Shear in midplane elevation
    thetaR : ArrayLike
        thetaR values at theta
    dthetaR_dtheta : ArrayLike
        Derivative of thetaR w.r.t theta at theta
    dthetaR_dr : ArrayLike
        Derivative of thetaR w.r.t r at theta
    cn : ArrayLike
        cosine moments of thetaR
    sn : ArrayLike
        sine moments of thetaR
    dcndr : ArrayLike
        Shear in cosine moments :math:`\partial c_n/\partial r`
    dsndr : ArrayLike
        Shear in sine moments :math:`\partial s_n/\partial r`

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

    d2Rdtheta2 : Array
        Second derivative of fitted :math:`R` w.r.t :math:`\theta`
    d2Rdrdtheta : Array
        Derivative of fitted :math:`R` w.r.t :math:`r` and :math:`\theta`
    d2Zdtheta2 : Array
        Second derivative of fitted :math:`Z` w.r.t :math:`\theta`
    d2Zdrdtheta : Array
        Derivative of fitted :math:`Z` w.r.t :math:`r` and :math:`\theta`

    """

    DEFAULT_N_MOMENTS: ClassVar[int] = 4
    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shift": 0.0,
        "dZ0dr": 0.0,
        "cn": np.zeros(DEFAULT_N_MOMENTS),
        "dcndr": np.zeros(DEFAULT_N_MOMENTS),
        "sn": np.zeros(DEFAULT_N_MOMENTS),
        "dsndr": np.zeros(DEFAULT_N_MOMENTS),
        **LocalGeometry.DEFAULT_INPUTS,
    }

    local_geometry: ClassVar[str] = "MXH"

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
        kappa: float = DEFAULT_INPUTS["kappa"],
        s_kappa: float = DEFAULT_INPUTS["s_kappa"],
        shift: float = DEFAULT_INPUTS["shift"],
        dZ0dr: float = DEFAULT_INPUTS["dZ0dr"],
        cn: NDArray[np.float64] = DEFAULT_INPUTS["cn"],
        sn: NDArray[np.float64] = DEFAULT_INPUTS["sn"],
        dcndr: Optional[NDArray[np.float64]] = None,
        dsndr: Optional[NDArray[np.float64]] = None,
        delta: Optional[float] = None,
        s_delta: Optional[float] = None,
        zeta: Optional[float] = None,
        s_zeta: Optional[float] = None,
    ):
        """TODO: Write docs

        The user must supply either:

        - delta/s_delta, zeta/s_zeta
        - cn/dcndr, sn/dsndr

        If both are supplied, the values for delta/zeta will overwrite those
        given for cn/sn.
        """
        if dcndr is None:
            dcndr = np.zeros_like(cn)
        if dsndr is None:
            dsndr = np.zeros_like(sn)

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
        self.kappa = kappa
        self.s_kappa = s_kappa
        self.shift = shift
        self.dZ0dr = dZ0dr
        self.cn = cn
        self.sn = sn
        self.dcndr = dcndr
        self.dsndr = dsndr

        # Error checking on array inputs
        arrays = ("cn", "sn", "dcndr", "dsndr")
        for name in arrays:
            if self[name].ndim != 1:
                msg = f"LocalGeometryFourierMXH input {name} should be 1D"
                raise ValueError(msg)
        if len(set(len(self[x]) for x in arrays)) != 1:
            msg = "Array inputs to LocalGeometryMXH must have same length"
            raise ValueError(msg)

        # If delta/s_delta/zeta/s_zeta set, these should overwrite the values in
        # sn/dsndr. This is achieved via property setters.
        if delta is not None:
            self.delta = delta
        if s_delta is not None:
            self.s_delta = s_delta
        if zeta is not None:
            self.zeta = zeta
        if s_zeta is not None:
            self.s_zeta = s_zeta

    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):
        r"""
        Calculates MXH shaping coefficients from R, Z and b_poloidal

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

        Zind_upper = np.argmax(Z)

        R_upper = R[Zind_upper]

        normalised_height = (Z - Zmid) / (kappa * self.rho)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        normalised_radius = (R - self.Rmaj) / self.rho

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

        self.theta_eq = theta

        theta_diff = thetaR - theta

        theta_dimensionless = units.Quantity(theta).magnitude
        theta_diff_dimensionless = units.Quantity(theta_diff).magnitude
        ntheta = np.outer(self.n, theta_dimensionless)
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

        self.kappa = kappa
        self.sn = sn * units.dimensionless
        self.cn = cn * units.dimensionless

        self.theta = theta
        self.thetaR = self.get_thetaR(self.theta)
        self.dthetaR_dtheta = self.get_dthetaR_dtheta(self.theta)

        self.R, self.Z = self.get_flux_surface(self.theta)

        s_kappa_init = 0.0
        params = [shift, s_kappa_init, 0.0, *[0.0] * self.n_moments * 2]

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in MXH::_set_shape_coefficients failed with message : {fits.message}"
            )

        if verbose:
            print(f"MXH :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in MXH::_set_shape_coefficients is poor with residual of {fits.cost}"
            )

        if isinstance(self.rho, PyroQuantity):
            length_units = self.rho.units
        else:
            length_units = 1.0

        self.shift = fits.x[0] * units.dimensionless
        self.s_kappa = fits.x[1] * units.dimensionless
        self.dZ0dr = fits.x[2] * units.dimensionless
        self.dcndr = fits.x[3 : self.n_moments + 3] / length_units
        self.dsndr = fits.x[self.n_moments + 3 :] / length_units

        # Force dsndr[0] which has no impact on flux surface
        self.dsndr[0] = 0.0 / length_units

        self.dthetaR_dr = self.get_dthetaR_dr(self.theta, self.dcndr, self.dsndr)

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

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

    def get_thetaR(self, theta):
        """

        Parameters
        ----------
        theta : Array

        Returns
        -------
        thetaR : Array
            Poloidal angle used in definition of R
        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        thetaR = theta + np.sum(
            (self.cn * np.cos(ntheta) + self.sn * np.sin(ntheta)),
            axis=1,
        )

        return thetaR

    def get_dthetaR_dtheta(self, theta):
        """

        Parameters
        ----------
        theta

        Returns
        -------
        dthetaR/dtheta : Array
            theta derivative of poloidal angle used in R
        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        dthetaR_dtheta = 1.0 + np.sum(
            (-self.cn * self.n * np.sin(ntheta) + self.sn * self.n * np.cos(ntheta)),
            axis=1,
        )

        return dthetaR_dtheta

    def get_d2thetaR_dtheta2(self, theta):
        """

        Parameters
        ----------
        theta

        Returns
        -------
        d^2thetaR/dtheta^2 : Array
            second theta derivative of poloidal angle used in R
        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        d2thetaR_dtheta2 = -np.sum(
            ((self.n**2) * (self.cn * np.cos(ntheta) + self.sn * np.sin(ntheta))),
            axis=1,
        )

        return d2thetaR_dtheta2

    def get_dthetaR_dr(self, theta, dcndr, dsndr):
        """

        Parameters
        ----------
        theta : Array
            theta angles
        dcndr : Array
            Asymmetric coefficients in thetaR
        dsndr : Array
            Symmetric coefficients in thetaR

        Returns
        -------

        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        dthetaR_dr = np.sum(
            (dcndr * np.cos(ntheta) + dsndr * np.sin(ntheta)),
            axis=1,
        )

        return dthetaR_dr

    def get_d2thetaR_drdtheta(self, theta, dcndr, dsndr):
        """

        Parameters
        ----------
        theta : Array
            theta angles
        dcndr : Array
            Asymmetric coefficients in thetaR
        dsndr : Array
            Symmetric coefficients in thetaR

        Returns
        -------

        """

        if hasattr(theta, "magnitude"):
            theta = theta.m

        ntheta = np.outer(theta, self.n)

        d2thetaR_drdtheta = np.sum(
            (-self.n * dcndr * np.sin(ntheta) + self.n * dsndr * np.cos(ntheta)),
            axis=1,
        )

        return d2thetaR_drdtheta

    def get_RZ_derivatives(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        """
        Calculates the derivatives of :math:`R(r, \theta)` and :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`, used in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use params = [shift, s_kappa, dZ0dr, cn[nmoments], sn[nmoments] ] when calculating
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
            shift = self.shift
            s_kappa = self.s_kappa
            dZ0dr = self.dZ0dr
            dcndr = self.dcndr
            dsndr = self.dsndr
        else:

            if isinstance(self.rho, PyroQuantity):
                length_units = self.rho.units
            else:
                length_units = 1.0

            shift = params[0] * units.dimensionless
            s_kappa = params[1] * units.dimensionless
            dZ0dr = params[2] * units.dimensionless
            dcndr = params[3 : self.n_moments + 3] / length_units
            dsndr = params[self.n_moments + 3 :] / length_units

        thetaR = self.get_thetaR(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, dcndr, dsndr)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)

        dZdtheta = self.get_dZdtheta(theta)

        dZdr = self.get_dZdr(theta, dZ0dr, s_kappa)

        dRdtheta = self.get_dRdtheta(thetaR, dthetaR_dtheta)

        dRdr = self.get_dRdr(shift, thetaR, dthetaR_dr)

        return dRdtheta, dRdr, dZdtheta, dZdr

    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
    ) -> np.ndarray:
        """
        Calculates the second derivatives of :math:`R(r, \theta)` and :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`, used in geometry terms

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

        thetaR = self.get_thetaR(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, self.dcndr, self.dsndr)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)
        d2thetaR_drdtheta = self.get_d2thetaR_drdtheta(theta, self.dcndr, self.dsndr)
        d2thetaR_dtheta2 = self.get_d2thetaR_dtheta2(theta)

        d2Zdtheta2 = self.get_d2Zdtheta2(theta)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.s_kappa)
        d2Rdtheta2 = self.get_d2Rdtheta2(thetaR, dthetaR_dtheta, d2thetaR_dtheta2)
        d2Rdrdtheta = self.get_d2Rdrdtheta(
            thetaR, dthetaR_dr, dthetaR_dtheta, d2thetaR_drdtheta
        )

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

        return self.kappa * self.rho * np.cos(theta)

    def get_d2Zdtheta2(self, theta):
        """
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

        return -self.kappa * self.rho * np.sin(theta)

    def get_dZdr(self, theta, dZ0dr, s_kappa):
        r"""
        Calculates the derivatives of :math:`Z(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Derivative in midplane elevation
        s_kappa : Float
            Shear in Elongation :math:`r/\kappa \partial \kappa/\partial r`

        Returns
        -------
        dZdr : Array
            Derivative of :math:`Z` w.r.t :math:`r`
        """
        return dZ0dr + self.kappa * np.sin(theta) * (1 + s_kappa)

    def get_d2Zdrdtheta(self, theta, s_kappa):
        r"""
        Calculates the second derivative of :math:`Z(r, \theta)` w.r.t :math:`r` and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        s_kappa : Float
            Shear in Elongation :math:`r/\kappa \partial \kappa/\partial r`

        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of :math:`Z` w.r.t :math:`r` and :math:`\theta`
        """
        return self.kappa * np.cos(theta) * (1 + s_kappa)

    def get_dRdtheta(self, thetaR, dthetaR_dtheta):
        """
        Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        thetaR: ArrayLike
            Array of thetaR points to evaluate dRdtheta on
        dthetaR_dtheta : ArrayLike
            Theta derivative of thetaR
        -------
        dRdtheta : Array
            Derivative of :math:`R` w.r.t :math:`\theta`
        """

        return -self.rho * np.sin(thetaR) * dthetaR_dtheta

    def get_d2Rdtheta2(self, thetaR, dthetaR_dtheta, d2thetaR_dtheta2):
        """
        Calculates the second derivative of :math:`R(r, \theta)` w.r.t :math:`\theta`

        Parameters
        ----------
        thetaR: ArrayLike
            Array of thetaR points to evaluate dRdtheta on
        dthetaR_dtheta : ArrayLike
            Theta derivative of thetaR
        d2thetaR_dtheta2 : ArrayLike
            Second theta derivative of thetaR
        -------
        d2Rdtheta2 : Array
            Second derivative of :math:`R` w.r.t :math:`\theta`
        """

        return -self.rho * np.sin(thetaR) * d2thetaR_dtheta2 - self.rho * (
            dthetaR_dtheta**2
        ) * np.cos(thetaR)

    def get_dRdr(self, shift, thetaR, dthetaR_dr):
        r"""
        Calculates the derivatives of :math:`R(r, \theta)` w.r.t :math:`r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdr on
        shift : Float
            Shafranov shift
        thetaR: ArrayLike
            Array of thetaR points to evaluate dRdtheta on
        dthetaR_dr : ArrayLike
            Radial derivative of thetaR

        Returns
        -------
        dRdr : Array
            Derivative of :math:`R` w.r.t :math:`r`
        """
        return shift + np.cos(thetaR) - self.rho * np.sin(thetaR) * dthetaR_dr

    def get_d2Rdrdtheta(self, thetaR, dthetaR_dr, dthetaR_dtheta, d2thetaR_drdtheta):
        """
        Calculate the second derivative of :math:`R(r, \theta)` w.r.t :math:`r` and :math:`\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dRdr on
        thetaR: ArrayLike
            Array of thetaR points to evaluate dRdtheta on
        dthetaR_dr : ArrayLike
            Radial derivative of thetaR
        dthetaR_dtheta : ArrayLike
            Theta derivative of thetaR
        d2thetaR_drdtheta : ArrayLike
            Second derivative of thetaR w.r.t :math:`r` and :math:`\theta`

        Returns
        -------
        d2Rdrdtheta : Array
            Second derivative of R w.r.t :math:`r` and :math:`\theta`
        """
        return -dthetaR_dtheta * np.sin(thetaR) - self.rho * (
            np.sin(thetaR) * d2thetaR_drdtheta
            + dthetaR_dr * dthetaR_dtheta * np.cos(thetaR)
        )

    def get_flux_surface(
        self,
        theta: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of MXH fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface

        Returns
        -------
        R : Array
            R values for this flux surface (if not normalised then in [m])
        Z : Array
            Z Values for this flux surface (if not normalised then in [m])
        """

        thetaR = self.get_thetaR(theta)

        R = self.Rmaj + self.rho * np.cos(thetaR)
        Z = self.Z0 + self.kappa * self.rho * np.sin(theta)

        return R, Z

    def _generate_shape_coefficients_units(self, norms):
        """
        Need to change dcndr and dsndr to pyro norms
        """
        return {
            "kappa": units.dimensionless,
            "s_kappa": units.dimensionless,
            "cn": units.dimensionless,
            "sn": units.dimensionless,
            "shift": units.dimensionless,
            "dZ0dr": units.dimensionless,
            "dcndr": norms.lref**-1,
            "dsndr": norms.lref**-1,
            "dthetaR_dr": norms.lref**-1,
        }

    @staticmethod
    def _shape_coefficient_names():
        """
        List of shape coefficient names used for printing
        """
        return [
            "kappa",
            "s_kappa",
            "cn",
            "sn",
            "shift",
            "dZ0dr",
            "dcndr",
            "dsndr",
            "dthetaR_dr",
        ]

    @classmethod
    def from_local_geometry(
        cls,
        local_geometry: Self,
        verbose: bool = False,
        show_fit: bool = False,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
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
                B0=local_geometry.B0,
                Z0=local_geometry.Z0,
                q=local_geometry.q,
                shat=local_geometry.shat,
                beta_prime=local_geometry.beta_prime,
                dpsidr=local_geometry.dpsidr,
                ip_ccw=local_geometry.ip_ccw,
                bt_ccw=local_geometry.bt_ccw,
                kappa=local_geometry.kappa,
                s_kappa=local_geometry.s_kappa,
                delta=local_geometry.delta,
                s_delta=local_geometry.s_delta,
                shift=local_geometry.shift,
                dZ0dr=local_geometry.dZ0dr,
            )

            result.R = local_geometry.R
            result.Z = local_geometry.Z
            result.theta = local_geometry.theta

            result.R_eq = local_geometry.R_eq
            result.Z_eq = local_geometry.Z_eq
            result.theta_eq = local_geometry.theta
            result.b_poloidal_eq = local_geometry.b_poloidal_eq

            result.dRdtheta = local_geometry.dRdtheta
            result.dRdr = local_geometry.dRdr
            result.dZdtheta = local_geometry.dZdtheta
            result.dZdr = local_geometry.dZdr

            result.dthetaR_dr = result.get_dthetaR_dr(
                result.theta, result.dcndr, result.dsndr
            )

            # Bunit for GACODE codes
            result.bunit_over_b0 = local_geometry.get_bunit_over_b0()

            if show_fit or axes is not None:
                result.plot_equilibrium_to_local_geometry_fit(
                    show_fit=show_fit, axes=axes
                )
            return result

        return super().from_local_geometry(
            local_geometry, verbose=verbose, show_fit=show_fit, axes=axes
        )
