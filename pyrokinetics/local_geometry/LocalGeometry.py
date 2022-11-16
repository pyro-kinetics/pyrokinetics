from cleverdict import CleverDict
from copy import deepcopy
from ..decorators import not_implemented
from ..factory import Factory
from ..constants import pi
import numpy as np
from typing import Tuple, Dict, Any
from ..typing import Scalar, ArrayLike


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
        "btccw": -1,
        "ipccw": -1,
    }


class LocalGeometry(CleverDict):
    """
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary

    """

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if args and not isinstance(args[0], CleverDict) and isinstance(args[0], dict):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            _data_dict = {"local_geometry": None}
            super(LocalGeometry, self).__init__(_data_dict)

    # TODO replace this with an abstract classmethod
    def load_from_eq(self, eq, psi_n, show_fit=False, **kwargs):
        """ "
        Loads LocalGeometry object from an Equilibrium Object

        """
        R, Z = eq.get_flux_surface(psi_n=psi_n)

        # Start at outboard midplance
        Z = np.roll(Z, -np.argmax(R))
        R = np.roll(R, -np.argmax(R))

        Z = Z[np.where(np.diff(R) != 0.0)]
        R = R[np.where(np.diff(R) != 0.0)]

        R_major = (max(R) + min(R)) / 2

        r_minor = (max(R) - min(R)) / 2

        rho = r_minor / eq.a_minor

        Zmid = (max(Z) + min(Z)) / 2

        fpsi = eq.f_psi(psi_n)
        B0 = fpsi / R_major

        drho_dpsi = eq.rho.derivative()(psi_n)
        dpsidr = 1 / drho_dpsi / eq.a_minor * (eq.psi_bdry - eq.psi_axis)

        pressure = eq.pressure(psi_n)
        q = eq.q(psi_n)

        dp_dpsi = eq.q.derivative()(psi_n)

        shat = rho / q * dp_dpsi / drho_dpsi

        dpressure_drho = eq.p_prime(psi_n) / drho_dpsi

        beta_prime = 8 * pi * 1e-7 * dpressure_drho / B0**2

        b_poloidal = eq.get_b_poloidal(R, Z)

        # Store Equilibrium values
        self.psi_n = psi_n
        self.rho = float(rho)
        self.r_minor = float(r_minor)
        self.Rmaj = float(R_major / eq.a_minor)
        self.Z0 = float(Zmid / eq.a_minor)
        self.a_minor = float(eq.a_minor)
        self.f_psi = float(fpsi)
        self.B0 = float(B0)
        self.q = float(q)
        self.shat = shat
        self.beta_prime = beta_prime
        self.pressure = pressure
        self.dpressure_drho = dpressure_drho
        self.dpsidr = dpsidr

        self.R_eq = R
        self.Z_eq = Z
        self.b_poloidal_eq = b_poloidal

        # Calculate shaping coefficients
        self.get_shape_coefficients(self.R_eq, self.Z_eq, self.b_poloidal_eq, **kwargs)

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta,
        )
        self.dRdtheta, self.dRdr, self.dZdtheta, self.dZdr = self.get_RZ_derivatives(self.theta)

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_fits()

    def load_from_local_geometry(self, local_geometry, verbose=False, show_fit=False):
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
                "Input to load_from_local_geometry must be of type LocalGeometry"
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

        self.get_shape_coefficients(self.R_eq, self.Z_eq, self.b_poloidal_eq, verbose)

        self.b_poloidal = self.get_b_poloidal(
            theta=self.theta,
        )
        self.dRdtheta, self.dRdr, self.dZdtheta, self.dZdr = self.get_RZ_derivatives(self.theta)

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_fits()

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

        # Get dpsidr from Bunit/B0
        local_geometry.dpsidr = local_geometry.bunit_over_b0 / local_geometry.q * local_geometry.rho

        local_geometry.theta = np.linspace(0, 2 * pi, 256)

        local_geometry.R, local_geometry.Z = local_geometry.get_flux_surface(local_geometry.theta, normalised=True)
        local_geometry.b_poloidal = local_geometry.get_b_poloidal(
            theta=local_geometry.theta,
        )

        #  Fitting R_eq, Z_eq, and b_poloidal_eq need to be defined from local parameters
        local_geometry.R_eq = local_geometry.R
        local_geometry.Z_eq = local_geometry.Z
        local_geometry.b_poloidal_eq = local_geometry.b_poloidal

        local_geometry.dRdtheta, local_geometry.dRdr, dZdtheta, dZdr = local_geometry.get_RZ_derivatives(local_geometry.theta)



        return local_geometry
    @not_implemented
    def get_shape_coefficients(self, R, Z, b_poloidal, verbose=False):
        r"""

        Parameters
        ----------
        Z
        b_poloidal
        verbose

        Returns
        -------

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

        dRdtheta, dRdr, dZdtheta, dZdr = self.get_RZ_derivatives(theta, params, normalised)

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

        return self.dpsidr / R * self.get_grad_r(theta, params)

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

    def plot_fits(self):
        import matplotlib.pyplot as plt

        R_fit, Z_fit = self.get_flux_surface(theta=self.theta, normalised=False)

        plt.plot(self.R_eq, self.Z_eq, label="Data")
        plt.plot(R_fit, Z_fit, "--", label="Fit")
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.title(f"Fit to flux surface for {self.local_geometry}")
        plt.legend()
        plt.grid()
        plt.show()

        bpol_fit = self.get_b_poloidal(
            theta=self.theta,
        )

        plt.plot(self.theta_eq, self.b_poloidal_eq, label="Data")
        plt.plot(self.theta, bpol_fit, "--", label="Fit")
        plt.legend()
        plt.xlabel("theta")
        plt.title(f"Fit to poloidal field for {self.local_geometry}")
        plt.ylabel("Bpol")
        plt.grid()
        plt.show()

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a LocalGeometry object

        Returns
        -------
        Copy of LocalGeometry object
        """
        # Create new empty object. Works for derived classes too.
        new_localgeometry = self.__class__()
        for key, value in self.items():
            new_localgeometry[key] = deepcopy(value, memodict)
        return new_localgeometry


# Create global factory for LocalGeometry objects
local_geometries = Factory(LocalGeometry)
