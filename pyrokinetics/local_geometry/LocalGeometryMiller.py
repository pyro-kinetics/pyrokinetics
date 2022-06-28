import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike


def default_miller_inputs():
    # Return default args to build a LocalGeometryMiller
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
        "local_geometry": "Miller",
    }


def grad_r(
    kappa: Scalar,
    delta: Scalar,
    s_kappa: Scalar,
    s_delta: Scalar,
    shift: Scalar,
    theta: ArrayLike,
) -> np.ndarray:
    """
    Miller definition of grad r from
    Miller, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
    Physics of Plasmas 5.4 (1998): 973-978.

    Parameters
    ----------
    kappa: Scalar
        Miller elongation
    delta: Scalar
        Miller triangularity
    s_kappa: Scalar
        Radial derivative of Miller elongation
    s_delta: Scalar
        Radial derivative of Miller triangularity
    shift: Scalar
        Shafranov shift
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    grad_r : Array
        grad_r(theta)
    """

    x = np.arcsin(delta)

    term1 = 1 / kappa

    term2 = np.sqrt(
        np.sin(theta + x * np.sin(theta)) ** 2 * (1 + x * np.cos(theta)) ** 2
        + (kappa * np.cos(theta)) ** 2
    )

    term3 = np.cos(x * np.sin(theta)) + shift * np.cos(theta)

    term4 = (
        (s_kappa - s_delta * np.cos(theta) + (1 + s_kappa) * x * np.cos(theta))
        * np.sin(theta)
        * np.sin(theta + x * np.sin(theta))
    )

    return term1 * term2 / (term3 + term4)


def flux_surface(
    kappa: Scalar,
    delta: Scalar,
    Rcen: Scalar,
    rmin: Scalar,
    theta: ArrayLike,
    Zmid: Scalar,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (R,Z) of a flux surface given a set of Miller fits

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

    Returns
    -------
    R : Array
        R values for this flux surface [m]
    Z : Array
        Z Values for this flux surface [m]
    """
    R = Rcen + rmin * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z = Zmid + kappa * rmin * np.sin(theta)

    return R, Z


def b_poloidal(
    kappa: Scalar,
    delta: Scalar,
    s_kappa: Scalar,
    s_delta: Scalar,
    shift: Scalar,
    dpsi_dr: Scalar,
    theta: ArrayLike,
    R: ArrayLike,
) -> np.ndarray:
    r"""
    Returns Miller prediction for b_poloidal given flux surface parameters

    Parameters
    ----------
    kappa: Scalar
        Miller elongation
    delta: Scalar
        Miller triangularity
    s_kappa: Scalar
        Radial derivative of Miller elongation
    s_delta: Scalar
        Radial derivative of Miller triangularity
    shift: Scalar
        Shafranov shift
    dpsi_dr: Scalar
        :math: `\partial \psi / \partial r`
    R: ArrayLike
        Major radius
    theta: ArrayLike
        Array of theta points to evaluate grad_r on

    Returns
    -------
    miller_b_poloidal : Array
        Array of b_poloidal from Miller fit
    """

    return dpsi_dr / R * grad_r(kappa, delta, s_kappa, s_delta, shift, theta)


class LocalGeometryMiller(LocalGeometry):
    r"""
    Miller Object representing local Miller fit parameters

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
            and not isinstance(args[0], LocalGeometryMiller)
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
        # be possible to have a miller object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        miller = cls(params)
        miller.bunit_over_b0 = miller.get_bunit_over_b0()
        return miller

    @classmethod
    def from_global_eq(cls, global_eq: Equilibrium, psi_n: float, verbose=False):
        # TODO this should replace load_from_eq.
        miller = cls()
        miller.load_from_eq(global_eq, psi_n=psi_n, verbose=verbose)
        return miller

    def load_from_eq(self, eq: Equilibrium, psi_n: float, verbose=False):
        r"""
        Loads Miller object from a GlobalEquilibrium Object

        Flux surface contours are fitted from 2D psi grid
        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        eq : GlobalEquilibrium
            GlobalEquilibrium object
        psi_n : Float
            Value of :math:`\psi_N` to generate local Miller parameters
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

        delta = (R_major - R_upper) / r_minor

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

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

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        R_miller, Z_miller = flux_surface(kappa, delta, R_major, r_minor, theta, Zmid)

        s_kappa_fit = 0.0
        s_delta_fit = 0.0
        shift_fit = shift
        dpsi_dr_fit = 1.0

        params = [s_kappa_fit, s_delta_fit, shift_fit, dpsi_dr_fit]

        self.psi_n = psi_n
        self.rho = float(rho)
        self.r_minor = float(r_minor)
        self.Rmaj = float(R_major / eq.a_minor)
        self.a_minor = float(eq.a_minor)
        self.f_psi = float(fpsi)
        self.B0 = float(B0)

        self.kappa = kappa
        self.delta = delta
        self.Z0 = float(Zmid / eq.a_minor)
        self.R = R
        self.Z = Z
        self.theta = theta
        self.b_poloidal = b_poloidal

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in Miller::load_from_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"Miller :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in Miller::load_from_eq is poor with residual of {fits.cost}"
            )

        self.s_kappa = fits.x[0]
        self.s_delta = fits.x[1]
        self.shift = fits.x[2]
        self.dpsidr = fits.x[3]

        self.q = float(q)
        self.shat = shat
        self.beta_prime = beta_prime
        self.pressure = pressure
        self.dpressure_drho = dpressure_drho

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between miller and equilibrium b_poloidal

        """

        return self.b_poloidal - b_poloidal(
            kappa=self.kappa,
            delta=self.delta,
            s_kappa=params[0],
            s_delta=params[1],
            shift=params[2],
            dpsi_dr=params[3],
            R=self.R,
            theta=self.theta,
        )

    def test_safety_factor(self):
        r"""
        Calculate safety fractor from Miller Object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from Miller B_poloidal
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

        R, Z = flux_surface(self.kappa, self.delta, self.Rmaj, self.rho, theta, self.Z0)

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR**2 + dZ**2)

        R_grad_r = R * grad_r(
            self.kappa, self.delta, self.s_kappa, self.s_delta, self.shift, theta
        )

        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMiller, self).__init__(default_miller_inputs())
