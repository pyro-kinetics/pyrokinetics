from ..decorators import not_implemented
from ..factory import Factory
from ..constants import pi
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..typing import ArrayLike
from ..equilibrium import Equilibrium
import matplotlib.pyplot as plt


def default_inputs():
    # Return default args to build a LocalGeometry
    # Uses a function call to avoid the user modifying these values
    return {
        "psi_n": 0.5,
        "rho": 0.5,
        "r_minor": 0.5,
        "Rmaj": 3.0,
        "Z0": 0.0,
        "a_minor": 1.0,
        "f_psi": 0.0,
        "B0": None,
        "q": 2.0,
        "shat": 1.0,
        "beta_prime": 0.0,
        "pressure:": 1.0,
        "dpressure_drho": 0.0,
        "dpsidr": 1.0,
        "btccw": -1,
        "ipccw": -1,
    }


class LocalGeometry:
    r"""
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary
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
        Toroidal field at major radius (f_psi / Rmajor) [T]
    bunit_over_b0 : Float
        Ratio of GACODE normalising field = :math: `q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear `r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta' = `2 \mu_0 \partial p \partial \rho 1/B0^2`

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
        Derivative of fitted `R` w.r.t `\theta`
    dRdr : Array
        Derivative of fitted `R` w.r.t `r`
    dZdtheta : Array
        Derivative of fitted `Z` w.r.t `\theta`
    dZdr : Array
        Derivative of fitted `Z` w.r.t `r`
    """

    def __init__(self, *args, **kwargs):
        s_args = list(args)
        if args and isinstance(s_args[0], dict):
            for key, value in s_args[0].items():
                setattr(self, key, value)

        elif len(args) == 0:
            self.local_geometry = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return self.__dict__.keys()

    def from_global_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, show_fit=False, **kwargs
    ):
        """
        Loads LocalGeometry object from an Equilibrium Object
        """
        # TODO Currently stripping units from Equilibrium/FluxSurface. These should be
        # added in a later update.
        # TODO FluxSurface is COCOS 11, this uses something else. Here we switch from
        # a clockwise theta grid to a counter-clockwise one, and divide any psi
        # quantities by 2 pi
        fs = eq.flux_surface(psi_n=psi_n)
        # Convert to counter-clockwise, discard repeated endpoint
        R = fs["R"].data.magnitude[:0:-1]
        Z = fs["Z"].data.magnitude[:0:-1]
        b_poloidal = fs["B_poloidal"].data.magnitude[:0:-1]

        R_major = fs.R_major.magnitude
        r_minor = fs.r_minor.magnitude
        rho = fs.rho.magnitude
        Zmid = fs.Z_mid.magnitude

        fpsi = fs.F.magnitude
        B0 = fpsi / R_major

        dpsidr = fs.psi_gradient.magnitude / (2 * np.pi)
        pressure = fs.p.magnitude
        q = fs.q.magnitude
        shat = fs.magnetic_shear.magnitude
        dpressure_drho = fs.pressure_gradient.magnitude * fs.a_minor.magnitude
        shift = fs.shafranov_shift.magnitude

        beta_prime = 8 * pi * 1e-7 * dpressure_drho / B0**2

        # Store Equilibrium values
        self.psi_n = psi_n
        self.rho = float(rho)
        self.r_minor = float(r_minor)
        self.Rmaj = float(R_major / fs.a_minor.magnitude)
        self.Z0 = float(Zmid / fs.a_minor.magnitude)
        self.a_minor = float(fs.a_minor.magnitude)
        self.f_psi = float(fpsi)
        self.B0 = float(B0)
        self.q = float(q)
        self.shat = shat
        self.beta_prime = beta_prime
        self.pressure = pressure
        self.dpressure_drho = dpressure_drho
        self.dpsidr = dpsidr
        self.shift = shift

        self.R_eq = R
        self.Z_eq = Z
        self.b_poloidal_eq = b_poloidal

        # Calculate shaping coefficients
        self._set_shape_coefficients(self.R_eq, self.Z_eq, self.b_poloidal_eq, **kwargs)

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta,
        )
        self.dRdtheta, self.dRdr, self.dZdtheta, self.dZdr = self.get_RZ_derivatives(
            self.theta
        )
        self.jacob = self.R * (self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta)

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_equilibrium_to_local_geometry_fit(show_fit=True)

    def from_local_geometry(self, local_geometry, verbose=False, show_fit=False):
        r"""
        Loads FourierCGYRO object from a LocalGeometry Object

        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        local_geometry : LocalGeometry
            LocalGeometry object
        verbose : Boolean
            Controls verbosity

        """

        if not isinstance(local_geometry, LocalGeometry):
            raise ValueError(
                "Input to from_local_geometry must be of type LocalGeometry"
            )

        # Load in parameters that
        self.psi_n = local_geometry.psi_n
        self.rho = local_geometry.rho
        self.r_minor = local_geometry.r_minor
        self.Rmaj = local_geometry.Rmaj
        self.a_minor = local_geometry.a_minor
        self.f_psi = local_geometry.f_psi
        self.B0 = local_geometry.B0
        self.Z0 = local_geometry.Z0
        self.q = local_geometry.q
        self.shat = local_geometry.shat
        self.beta_prime = local_geometry.beta_prime
        self.pressure = local_geometry.pressure
        self.dpressure_drho = local_geometry.dpressure_drho

        self.R_eq = local_geometry.R_eq
        self.Z_eq = local_geometry.Z_eq
        self.theta_eq = local_geometry.theta
        self.b_poloidal_eq = local_geometry.b_poloidal_eq
        self.dpsidr = local_geometry.dpsidr

        self._set_shape_coefficients(self.R_eq, self.Z_eq, self.b_poloidal_eq, verbose)

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta,
        )
        self.dRdtheta, self.dRdr, self.dZdtheta, self.dZdr = self.get_RZ_derivatives(
            self.theta
        )
        self.jacob = self.R * (self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta)

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_equilibrium_to_local_geometry_fit(show_fit=True)

    @classmethod
    def from_gk_data(cls, params: Dict[str, Any]):
        """
        Initialise from data gathered from GKCode object, and additionally set
        bunit_over_b0
        """
        # TODO change __init__ to take necessary parameters by name. It shouldn't
        # be possible to have a local_geometry object that does not contain all attributes.
        # bunit_over_b0 should be an optional argument, and the following should
        # be performed within __init__ if it is None
        local_geometry = cls(params)

        local_geometry.bunit_over_b0 = local_geometry.get_bunit_over_b0()

        local_geometry.r_minor = local_geometry.rho * local_geometry.a_minor

        # Get dpsidr from Bunit/B0
        local_geometry.dpsidr = (
            local_geometry.bunit_over_b0 / local_geometry.q * local_geometry.rho
        )

        # This is arbitrary, maybe should be a user input
        local_geometry.theta = np.linspace(0, 2 * pi, 256)

        local_geometry.R, local_geometry.Z = local_geometry.get_flux_surface(
            local_geometry.theta, normalised=True
        )
        local_geometry.b_poloidal = local_geometry.get_b_poloidal(
            theta=local_geometry.theta,
        )

        #  Fitting R_eq, Z_eq, and b_poloidal_eq need to be defined from local parameters
        local_geometry.R_eq = local_geometry.R
        local_geometry.Z_eq = local_geometry.Z
        local_geometry.b_poloidal_eq = local_geometry.b_poloidal

        (
            local_geometry.dRdtheta,
            local_geometry.dRdr,
            local_geometry.dZdtheta,
            local_geometry.dZdr,
        ) = local_geometry.get_RZ_derivatives(local_geometry.theta)
        local_geometry.jacob = local_geometry.R * (
            local_geometry.dRdr * local_geometry.dZdtheta
            - local_geometry.dZdr * local_geometry.dRdtheta
        )

        return local_geometry

    @not_implemented
    def _set_shape_coefficients(self, R, Z, b_poloidal, verbose=False):
        r"""
        Calculates LocalGeometry shaping coefficients from R, Z and b_poloidal

        Parameters
        ----------
        R : Array
            R for the given flux surface
        Z : Array
            Z for the given flux surface
        b_poloidal : Array
            `b_\theta` for the given flux surface
        verbose : Boolean
            Controls verbosity
        """

        pass

    @not_implemented
    def get_RZ_derivatives(self, params=None, normalised=False):
        pass

    def get_grad_r(
        self,
        theta: ArrayLike,
        params=None,
        normalised=False,
    ) -> np.ndarray:
        """
        MXH definition of grad r from
        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Parameters
        ----------
        kappa: Scalar
            elongation
        shift: Scalar
            Shafranov shift
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        grad_r : Array
            grad_r(theta)
        """

        dRdtheta, dRdr, dZdtheta, dZdr = self.get_RZ_derivatives(
            theta, params, normalised
        )

        g_tt = dRdtheta**2 + dZdtheta**2

        grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

        return grad_r

    def minimise_b_poloidal(self, params, even_space_theta=False):
        """
        Function for least squares minimisation of poloidal field

        Parameters
        ----------
        params : List
            List of the form [s_kappa, s_delta, shift, dpsidr]

        Returns
        -------
        Difference between local geometry and equilibrium get_b_poloidal

        """

        if even_space_theta:
            b_poloidal_eq = self.b_poloidal_even_space
        else:
            b_poloidal_eq = self.b_poloidal_eq
        return b_poloidal_eq - self.get_b_poloidal(theta=self.theta, params=params)

    def get_b_poloidal(self, theta: ArrayLike, params=None) -> np.ndarray:
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
        dpsidr: Scalar
            :math: `\partial \psi / \partial r`
        R: ArrayLike
            Major radius
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        local_geometry_b_poloidal : Array
            Array of get_b_poloidal from Miller fit
        """

        R = self.R * self.a_minor

        return np.abs(self.dpsidr) / R * self.get_grad_r(theta, params)

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

        R, Z = self.get_flux_surface(theta=theta, normalised=True)

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR**2 + dZ**2)

        R_grad_r = R * self.get_grad_r(theta, normalised=True)
        integral = np.sum(dL / R_grad_r)

        return integral * self.Rmaj / (2 * pi * self.rho)

    def get_f_psi(self):
        r"""
        Calculate safety fractor from b poloidal field, R, Z and q
        :math:`f = \frac{2\pi q}{\oint \frac{dl}{R^2 B_{\theta}}}`

        Returns
        -------
        f : Float
            Prediction for :math:`f_\psi` from B_poloidal
        """

        R = self.R
        Z = self.Z
        b_poloidal = self.b_poloidal
        q = self.q

        dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
        dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0

        dL = np.sqrt(dR**2 + dZ**2)

        integral = np.sum(dL / (R**2 * b_poloidal))

        return 2 * pi * q / integral

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

        b_poloidal = self.b_poloidal

        f = self.f_psi

        integral = np.sum(f * dL / (R**2 * b_poloidal))

        q = integral / (2 * pi)

        return q

    def plot_equilibrium_to_local_geometry_fit(
        self, axes: Optional[Tuple[plt.Axes, plt.Axes]] = None, show_fit=False
    ):
        # Get flux surface and b_poloidal
        R_fit, Z_fit = self.get_flux_surface(theta=self.theta, normalised=False)

        bpol_fit = self.get_b_poloidal(
            theta=self.theta,
        )

        # Set up plot if one doesn't exist already
        if axes is None:
            fig, axes = plt.subplots(1, 2)
        else:
            fig = axes[0].get_figure()

        # Plot R, Z
        axes[0].plot(self.R_eq, self.Z_eq, label="Data")
        axes[0].plot(R_fit, Z_fit, "--", label="Fit")
        axes[0].set_xlabel("R")
        axes[0].set_ylabel("Z")
        axes[0].set_aspect("equal")
        axes[0].set_title(f"Fit to flux surface for {self.local_geometry}")
        axes[0].legend()
        axes[0].grid()

        # Plot Bpoloidal
        axes[1].plot(self.theta_eq, self.b_poloidal_eq, label="Data")
        axes[1].plot(self.theta, bpol_fit, "--", label="Fit")
        axes[1].legend()
        axes[1].set_xlabel("theta")
        axes[1].set_title(f"Fit to poloidal field for {self.local_geometry}")
        axes[1].set_ylabel("Bpol")
        axes[1].grid()

        if show_fit:
            plt.show()
        else:
            return fig, axes


# Create global factory for LocalGeometry objects
local_geometries = Factory(LocalGeometry)
