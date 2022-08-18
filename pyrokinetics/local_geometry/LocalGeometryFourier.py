import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import quad, simpson
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
import matplotlib.pyplot as plt


def default_fourier_inputs():
    # Return default args to build a LocalGeometryfourier
    # Uses a function call to avoid the user modifying these values
    return {
        "rho": 0.9,
        "Rmaj": 3.0,
        "Z0": 0.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "q": 2.0,
        "shat": 1.0,
        "shift": 0.0,
        "btccw": -1,
        "ipccw": -1,
        "beta_prime": 0.0,
        "local_geometry": "fourier",
    }


def grad_r(
    kappa: Scalar,
    R: ArrayLike,
    Z: ArrayLike,
    r: Scalar,
    shift: Scalar,
    dZ0dr: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    dthetaR_dtheta: ArrayLike,
    dthetaR_dr: ArrayLike,
) -> np.ndarray:
    """
    fourier definition of grad r from
    fourier, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
    Physics of Plasmas 5.4 (1998): 973-978.

    Parameters
    ----------
    kappa: Scalar
        fourier elongation
    shift: Scalar
        Shafranov shift
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    grad_r : Array
        grad_r(theta)
    """

    dZdtheta = kappa * r * np.cos(theta)

    # Assumes dZ0/dr = 0
    dZdr = dZ0dr + kappa * np.sin(theta)

    dRdtheta = -r * np.sin(thetaR) * dthetaR_dtheta

    dRdr = shift + np.cos(thetaR) - r * np.sin(thetaR) * dthetaR_dr

    # Plot dR_dtheta
    # plt.plot(theta, dRdtheta, label='Fit')
    # plt.plot(theta, np.gradient(R, theta), '--', label='Data')
    # plt.xlabel('theta')
    # plt.ylabel('dR / dtheta')
    # plt.legend()
    # plt.show()

    # Plot dZR_dtheta
    # plt.plot(theta, dZdtheta, label='Fit')
    # plt.plot(theta, np.gradient(Z, theta), '--', label='Data')
    # plt.xlabel('theta')
    # plt.ylabel('dZ / dtheta')
    # plt.legend()
    # plt.show()

    g_tt = dRdtheta**2 + dZdtheta**2

    grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    return grad_r


def flux_surface(
    kappa: Scalar,
    Rcen: Scalar,
    rmin: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    Zmid: Scalar,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (R,Z) of a flux surface given a set of fourier fits

    Parameters
    ----------
    kappa : Float
        Elongation
    delta : Float
        Triangularity
    Rcen : Float
        Major radius of flux surface [m]
    rmin : Float
        Minor radius of flux surface [m]
    Zmid : Float
        Vertical midpoint of flux surface [m]
    theta : Array
        Values of theta to evaluate flux surface
    thetaR : Array
        Values of thetaR to evaluate flux surface

    Returns
    -------
    R : Array
        R values for this flux surface [m]
    Z : Array
        Z Values for this flux surface [m]
    """
    R = Rcen + rmin * np.cos(thetaR)
    Z = Zmid + kappa * rmin * np.sin(theta)

    return R, Z


def get_b_poloidal(
    kappa: Scalar,
    R: ArrayLike,
    Z: ArrayLike,
    r: Scalar,
    shift: Scalar,
    dZ0dr: Scalar,
    dpsidr: Scalar,
    theta: ArrayLike,
    thetaR: ArrayLike,
    dthetaR_dtheta: ArrayLike,
    dthetaR_dr: ArrayLike,
) -> np.ndarray:
    r"""
    Returns fourier prediction for b_poloidal given flux surface parameters

    Parameters
    ----------
    kappa: Scalar
        fourier elongation
    delta: Scalar
        fourier triangularity
    s_kappa: Scalar
        Radial derivative of fourier elongation
    s_delta: Scalar
        Radial derivative of fourier triangularity
    shift: Scalar
        Shafranov shift
    dpsidr: Scalar
        :math: `\partial \psi / \partial r`
    R: ArrayLike
        Major radius
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    fourier_b_poloidal : Array
        Array of b_poloidal from fourier fit
    """

    return (
        dpsidr
        / R
        * grad_r(
            kappa, R, Z, r, shift, dZ0dr, theta, thetaR, dthetaR_dtheta, dthetaR_dr
        )
    )


class LocalGeometryFourier(LocalGeometry):
    r"""
    Fourier Object representing local fourier fit parameters
    Uses method in Plasma Phys. Control. Fusion 63 (2021) 012001 (5pp)
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
        Toroidal field at major radius (f_psi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    kappa : Float
        Elongation
    delta : Float
        Triangularity
    s_kappa : Float
        Shear in Elongation
    s_delta : Float
        Shear in Triangularity
    shift : Float
        Shafranov shift
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear
    beta_prime : Float
        :math:`\beta' = \beta * a/L_p`

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if (
            args
            and not isinstance(args[0], LocalGeometryFourier)
            and isinstance(args[0], dict)
        ):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    @classmethod
    def from_gk_data(cls, params: Dict[str, Any]):
        """
        Initialise from data gathered from GKCode object, and additionally set
        bunit_over_b0
        """
        # TODO change __init__ to take necessary parameters by name. It shouldn't
        # be possible to have a fourier object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        fourier = cls(params)
        fourier.bunit_over_b0 = fourier.get_bunit_over_b0()
        return fourier

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        fourier = cls()
        fourier.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return fourier

    def load_from_eq(self, eq: Equilibrium, psi_n: float, verbose=False):
        r"""
        Loads fourier object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local fourier parameters
        verbose : Boolean
            Controls verbosity

        """

        R, Z = eq.get_flux_surface(psi_n=psi_n)

        b_poloidal = eq.get_b_poloidal(R, Z)

        R_major = eq.R_major(psi_n)

        rho = eq.rho(psi_n)

        r_minor = rho * eq.a_minor

        kappa = (max(Z) - min(Z)) / (2 * r_minor)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        fpsi = eq.f_psi(psi_n)
        B0 = fpsi / R_major

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor
        dZ0dr = eq.Z_mid.derivative()(psi_n) / drho_dpsi / eq.a_minor

        pressure = eq.pressure(psi_n)
        q = eq.q(psi_n)

        dp_dpsi = eq.q.derivative()(psi_n)

        shat = rho / q * dp_dpsi / drho_dpsi

        dpressure_drho = eq.p_prime(psi_n) / drho_dpsi

        beta_prime = 8 * pi * 1e-7 * dpressure_drho / B0**2

        normalised_height = (Z - Zmid) / (kappa * r_minor)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        normalised_radius = (R - R_major) / r_minor

        normalised_radius = np.where(
            np.isclose(normalised_radius, 1.0), 1.0, normalised_radius
        )
        normalised_radius = np.where(
            np.isclose(normalised_radius, -1.0), -1.0, normalised_radius
        )

        thetaR = np.arccos(normalised_radius)

        for i in range(len(theta)):
            if R[i] < R_upper:
                theta[i] = np.pi - theta[i]
            else:
                if Z[i] < 0:
                    theta[i] = 2 * np.pi + theta[i]

        for i in range(len(thetaR)):
            if Z[i] < 0:
                thetaR[i] = 2 * np.pi - thetaR[i]

        theta_diff = thetaR - theta

        self.psi_n = psi_n
        self.rho = float(rho)
        self.r_minor = float(r_minor)
        self.Rmaj = float(R_major / eq.a_minor)
        self.a_minor = float(eq.a_minor)
        self.f_psi = float(fpsi)
        self.B0 = float(B0)

        self.Z0 = float(Zmid / eq.a_minor)
        self.R = R
        self.Z = Z
        self.theta = theta
        self.b_poloidal = b_poloidal

        self.q = float(q)
        self.shat = shat
        self.beta_prime = beta_prime
        self.pressure = pressure
        self.dpressure_drho = dpressure_drho

        for n_moments in [4, 5, 6, 8, 10, 12, 14, 16]:
            # n_moments = 8

            asym_coeff = np.empty(n_moments)
            sym_coeff = np.empty(n_moments)

            for i in range(n_moments):
                c_int = theta_diff * np.cos(i * theta)
                asym_coeff[i] = simpson(c_int, theta)

                s_int = theta_diff * np.sin(i * theta)
                sym_coeff[i] = simpson(s_int, theta)

            asym_coeff *= 1 / np.pi
            sym_coeff *= 1 / np.pi

            self.kappa = kappa
            self.n_moments = n_moments
            self.sym_coeff = sym_coeff
            self.asym_coeff = asym_coeff

            self.thetaR = self.get_thetaR(theta)
            self.dthetaR_dtheta = self.get_dthetaR_dtheta(theta)

            """
            R_fit, Z_fit, = flux_surface(
                kappa=kappa,
                Rcen=R_major,
                rmin=r_minor,
                theta=theta,
                thetaR=thetaR,
                Zmid=Zmid,
            )
            plt.plot(R_fit, Z_fit, label="Fit")
            plt.plot(R, Z, "--", label="Data")
            plt.xlabel("R")
            plt.ylabel("Z")
            plt.legend()
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.show()
    
            # Plot dthetaR_dtheta
            plt.plot(theta, self.thetaR, label="Fit")
            plt.plot(theta, thetaR, "--", label="Data")
            plt.xlabel("theta")
            plt.ylabel("thetaR")
            plt.legend()
            plt.show()
    
            # Plot dthetaR_dtheta
            plt.plot(theta, self.dthetaR_dtheta, label="Fit")
            plt.plot(theta, np.gradient(thetaR, theta), "--", label="Data")
            plt.xlabel("theta")
            plt.ylabel("dthetaR / dtheta")
            plt.legend()
            plt.show()
            """
            params = [shift, dZ0dr, 1.0, *[0.0] * self.n_moments * 2]

            fits = least_squares(self.minimise_b_poloidal, params)

            # Check that least squares didn't fail
            if not fits.success:
                raise Exception(
                    f"Least squares fitting in Fourier::load_from_eq failed with message : {fits.message}"
                )

            if verbose:
                print(f"Fourier :: Fit to Bpoloidal obtained with residual {fits.cost}")

            if fits.cost > 1:
                import warnings

                warnings.warn(
                    f"Warning Fit to Fourier in Miller::load_from_eq is poor with residual of {fits.cost}"
                )

            self.shift = fits.x[0]
            self.dpsidr = fits.x[1]
            self.dZ0dr = fits.x[2]
            self.dasym_dr = fits.x[3 : self.n_moments + 3]
            self.dsym_dr = fits.x[self.n_moments + 3 :]

            self.dthetaR_dr = self.get_dthetaR_dr(
                self.theta, self.dasym_dr, self.dsym_dr
            )

            bpol_fit = get_b_poloidal(
                kappa=self.kappa,
                r=self.r_minor,
                shift=self.shift,
                dpsidr=self.dpsidr,
                dZ0dr=self.dZ0dr,
                R=self.R,
                Z=self.Z,
                theta=self.theta,
                thetaR=self.thetaR,
                dthetaR_dtheta=self.dthetaR_dtheta,
                dthetaR_dr=self.dthetaR_dr,
            )

            plt.plot(self.theta, bpol_fit, label=f"N moments={n_moments}")

        plt.plot(self.theta, self.b_poloidal, "--", label="Data", color="k")
        plt.legend()
        plt.xlabel("theta")
        plt.title("Fit to poloidal field with different number of moments")
        plt.ylabel("Bpol")
        plt.show()

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

    def get_thetaR(self, theta):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        thetaR = theta + np.sum(
            (
                self.asym_coeff[:, None] * np.cos(ntheta)
                + self.sym_coeff[:, None] * np.sin(ntheta)
            ),
            axis=0,
        )

        return thetaR

    def get_dthetaR_dtheta(self, theta):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        dthetaR_dtheta = 1.0 + np.sum(
            (
                -self.asym_coeff[:, None] * n[:, None] * np.sin(ntheta)
                + self.sym_coeff[:, None] * n[:, None] * np.cos(ntheta)
            ),
            axis=0,
        )

        return dthetaR_dtheta

    def get_dthetaR_dr(self, theta, dasym_dr, dsym_dr):

        n = np.linspace(0, self.n_moments - 1, self.n_moments)
        ntheta = n[:, None] * theta[None, :]
        dthetaR_dr = np.sum(
            (dasym_dr[:, None] * np.cos(ntheta) + dsym_dr[:, None] * np.sin(ntheta)),
            axis=0,
        )

        return dthetaR_dr

    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between fourier and equilibrium b_poloidal

        """

        shift = params[0]
        dpsidr = params[1]
        dZ0dr = params[2]
        dasym_dr = params[3 : self.n_moments + 3]
        dsym_dr = params[self.n_moments + 3 :]

        dthetaR_dr = self.get_dthetaR_dr(self.theta, dasym_dr, dsym_dr)

        return self.b_poloidal - get_b_poloidal(
            kappa=self.kappa,
            r=self.r_minor,
            shift=shift,
            dpsidr=dpsidr,
            dZ0dr=dZ0dr,
            R=self.R,
            Z=self.Z,
            theta=self.theta,
            thetaR=self.thetaR,
            dthetaR_dtheta=self.dthetaR_dtheta,
            dthetaR_dr=dthetaR_dr,
        )

    def test_safety_factor(self):
        r"""
        Calculate safety fractor from fourier Object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from fourier B_poloidal
        """

        R = self.R
        Z = self.Z

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR**2 + dZ**2)

        b_poloidal = self.get_b_poloidal

        f = self.f_psi

        integral = np.sum(f * dL / (R**2 * b_poloidal))

        q = integral / (2 * pi)

        return q

    def get_bunit_over_b0(self):
        r"""
        Get Bunit/B0 using q and loop integral of Bp

        :math:`\frac{B_{unit}}{B_0} = \frac{R_0}{2\pi r_{minor}} \oint \frac{a}{R} \frac{dl_N}{\nabla r}`

        where :math:`dl_N = \frac{dl}{a_{minor}}` coming from the normalising a_minor

        Returns
        -------
        bunit_over_b0 : Float
             :math:`\frac{B_{unit}}{B_0}`

        """

        theta = np.linspace(0, 2 * pi, 256)

        thetaR = self.get_thetaR(theta)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, self.dasym_dr, self.dsym_dr)

        R, Z = flux_surface(
            kappa=self.kappa,
            Rcen=self.Rmaj,
            rmin=self.rho,
            Zmid=self.Z0,
            theta=theta,
            thetaR=thetaR,
        )

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR**2 + dZ**2)

        R_grad_r = R * grad_r(
            self.kappa,
            self.R,
            self.Z,
            self.r_minor,
            self.shift,
            self.dZ0dr,
            theta,
            thetaR,
            dthetaR_dtheta,
            dthetaR_dr,
        )

        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryFourier, self).__init__(default_fourier_inputs())
