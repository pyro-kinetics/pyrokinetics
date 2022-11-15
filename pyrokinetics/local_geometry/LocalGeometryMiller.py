import numpy as np
from typing import Tuple, Dict, Any
from scipy.optimize import least_squares  # type: ignore
from ..constants import pi
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import Scalar, ArrayLike
from .LocalGeometry import default_inputs


def default_miller_inputs():
    # Return default args to build a LocalGeometryMiller
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    miller_defaults = {
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shift": 0.0,
        "local_geometry": "Miller",
    }

    return {**base_defaults, **miller_defaults}


def grad_r(
    kappa: Scalar,
    delta: Scalar,
    zeta: Scalar,
    s_kappa: Scalar,
    s_delta: Scalar,
    s_zeta: Scalar,
    shift: Scalar,
    dZ0dr: Scalar,
    theta: ArrayLike,
    rmin: Scalar,
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

    dZdtheta = (
        kappa
        * rmin
        * (1 + 2 * zeta * np.cos(2 * theta))
        * np.cos(theta + zeta * np.sin(2 * theta))
    )

    dZdr = (
        dZ0dr
        + kappa * np.sin(theta + zeta * np.sin(2 * theta))
        + s_kappa * kappa * np.sin(theta + zeta * np.sin(2 * theta))
        + kappa
        * rmin
        * s_zeta
        * np.sin(2 * theta)
        * np.cos(theta + zeta * np.sin(2 * theta))
    )

    dRdtheta = -rmin * np.sin(theta + x * np.sin(theta)) * (1 + x * np.cos(theta))

    dRdr = (
        shift
        + np.cos(theta + x * np.sin(theta))
        - np.sin(theta + x * np.sin(theta)) * np.sin(theta) * s_delta
    )

    g_tt = dRdtheta ** 2 + dZdtheta ** 2

    grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    return grad_r


def flux_surface(
    kappa: Scalar,
    delta: Scalar,
    zeta: Scalar,
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
    Z = Zmid + kappa * rmin * np.sin(theta + zeta * np.sin(2 * theta))

    return R, Z


def get_b_poloidal(
    kappa: Scalar,
    delta: Scalar,
    zeta: Scalar,
    s_kappa: Scalar,
    s_delta: Scalar,
    s_zeta: Scalar,
    shift: Scalar,
    dZ0dr: Scalar,
    dpsi_dr: Scalar,
    theta: ArrayLike,
    R: ArrayLike,
    rmin: Scalar,
) -> np.ndarray:
    r"""
    Returns Miller prediction for get_b_poloidal given flux surface parameters

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
        Array of get_b_poloidal from Miller fit
    """

    return (
        dpsi_dr
        / R
        * grad_r(
            kappa, delta, zeta, s_kappa, s_delta, s_zeta, shift, dZ0dr, theta, rmin
        )
    )


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

    def load_from_eq(self, eq: Equilibrium, psi_n: float, verbose=False, **kwargs):
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

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().load_from_eq(eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, **kwargs)

    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False, shift=0.0):

        kappa = (max(Z) - min(Z)) / (2 * self.r_minor)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        delta = self.Rmaj / self.rho - R_upper / self.r_minor

        normalised_height = (Z - Zmid) / (kappa * self.r_minor)

        self.kappa = kappa
        self.delta = delta
        self.Zmid = Zmid

        self.Z0 = float(Zmid / self.a_minor)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        R_pi4 = (
            self.Rmaj + self.rho * np.cos(pi / 4 + np.arcsin(delta) * np.sin(pi / 4))
        ) * self.a_minor

        R_gt_0 = np.where(Z>0, R, 0.0)
        Z_pi4 = Z[np.argmin(np.abs(R_gt_0 - R_pi4))]

        zeta = np.arcsin((Z_pi4 - Zmid) / (kappa * self.r_minor)) - pi / 4

        self.zeta = zeta

        theta_guess = np.arcsin(normalised_height)
        theta = self._get_theta_from_squareness(theta_guess)

        for i in range(len(theta)):
            if R[i] < R_upper:
                if Z[i] >= 0:
                    theta[i] = np.pi - theta[i]
                elif Z[i] < 0:
                    theta[i] = -np.pi - theta[i]

        self.kappa = kappa
        self.delta = delta
        self.Z0 = float(Zmid / self.a_minor)
        self.theta = theta

        s_kappa_fit = 0.0
        s_delta_fit = 0.0
        s_zeta_fit = 0.0
        shift_fit = shift
        dZ0dr_fit = 0.0
        dpsi_dr_fit = 1.0

        params = [
            s_kappa_fit,
            s_delta_fit,
            s_zeta_fit,
            shift_fit,
            dZ0dr_fit,
            dpsi_dr_fit,
        ]

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
        self.s_zeta = fits.x[2]
        self.shift = fits.x[3]
        self.dZ0dr = fits.x[4]
        self.dpsidr = fits.x[5]

    def minimise_b_poloidal(self, params):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between miller and equilibrium get_b_poloidal

        """

        return self.b_poloidal - get_b_poloidal(
            kappa=self.kappa,
            delta=self.delta,
            zeta=self.zeta,
            s_kappa=params[0],
            s_delta=params[1],
            s_zeta=params[2],
            shift=params[3],
            dZ0dr=params[4],
            dpsi_dr=params[5],
            R=self.R,
            theta=self.theta,
            rmin=self.r_minor,
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

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        b_poloidal = self.b_poloidal

        f = self.f_psi

        integral = np.sum(f * dL / (R ** 2 * b_poloidal))

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

        R, Z = flux_surface(
            self.kappa, self.delta, self.zeta, self.Rmaj, self.rho, theta, self.Z0
        )

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR ** 2 + dZ ** 2)

        R_grad_r = R * grad_r(
            self.kappa,
            self.delta,
            self.zeta,
            self.s_kappa,
            self.s_delta,
            self.s_zeta,
            self.shift,
            self.dZ0dr,
            theta,
            self.rho,
        )
        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def plot_fits(self):
        import matplotlib.pyplot as plt

        R_fit, Z_fit = flux_surface(
            self.kappa,
            self.delta,
            self.zeta,
            self.Rmaj * self.a_minor,
            self.r_minor,
            self.theta,
            self.Z0 * self.a_minor,
        )

        plt.plot(self.R, self.Z, label="Data")
        plt.plot(R_fit, Z_fit, "--", label="Fit")
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.title("Fit to flux surface for Miller")
        plt.legend()
        plt.grid()
        plt.show()

        bpol_fit = get_b_poloidal(
            kappa=self.kappa,
            delta=self.delta,
            zeta=self.zeta,
            s_kappa=self.s_kappa,
            s_delta=self.s_delta,
            s_zeta=self.s_zeta,
            shift=self.shift,
            dZ0dr=self.dZ0dr,
            dpsi_dr=self.dpsidr,
            R=self.R,
            theta=self.theta,
            rmin=self.r_minor,
        )

        plt.plot(self.theta, self.b_poloidal, label="Data")
        plt.plot(self.theta, bpol_fit, "--", label="Fit")
        plt.legend()
        plt.xlabel("theta")
        plt.title("Fit to poloidal field for Miller")
        plt.ylabel("Bpol")
        plt.grid()
        plt.show()

    def set_R_Z_b_poloidal(self, theta):

        self.theta = theta
        self.R, self.Z = flux_surface(
            theta=theta,
            kappa=self.kappa,
            delta=self.delta,
            zeta=self.zeta,
            Rcen=self.Rmaj,
            rmin=self.r_minor,
            Zmid=self.Z0,
        )

        self.b_poloidal = get_b_poloidal(
            kappa=self.kappa,
            delta=self.delta,
            zeta=self.zeta,
            s_kappa=self.s_kappa,
            s_delta=self.s_delta,
            s_zeta=self.zeta,
            dpsi_dr=self.dpsidr,
            shift=self.shift,
            dZ0dr=self.dZ0dr,
            theta=self.theta,
            R=self.R,
        )

    def _get_theta_from_squareness(self, theta):

        fits = least_squares(self._minimise_theta_from_squareness, theta)

        return fits.x

    def _minimise_theta_from_squareness(self, params):

        theta = params
        theta_func = np.arcsin((self.Z - self.Zmid) / (self.kappa * self.r_minor))

        sum_diff = np.sum(np.abs(theta_func - theta - self.zeta * np.sin(2 * theta)))
        return sum_diff

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMiller, self).__init__(default_miller_inputs())
