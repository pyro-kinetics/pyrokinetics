from cleverdict import CleverDict
from copy import deepcopy
from ..decorators import not_implemented
from ..factory import Factory
from ..constants import pi
import numpy as np


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

        R_major = eq.R_major(psi_n)

        rho = eq.rho(psi_n)

        r_minor = rho * eq.a_minor

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

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_fits()

    def load_from_lg(self, lg, verbose=False, show_fit=False):
        r"""
        Loads FourierCGYRO object from a LocalGeometry Object

        Gradients in shaping parameters are fitted from poloidal field

        Parameters
        ----------
        lg : LocalGeometry
            LocalGeometry object
        verbose : Boolean
            Controls verbosity

        """

        if not isinstance(lg, LocalGeometry):
            raise ValueError("Input to load_from_lg must be of type LocalGeometry")

        # Load in parameters that
        self.psi_n = lg.psi_n
        self.rho = lg.rho
        self.r_minor = lg.r_minor
        self.Rmaj = lg.Rmaj
        self.a_minor = lg.a_minor
        self.f_psi = lg.f_psi
        self.B0 = lg.B0
        self.Z0 = lg.Z0
        self.q = lg.q
        self.shat = lg.shat
        self.beta_prime = lg.beta_prime
        self.pressure = lg.pressure
        self.dpressure_drho = lg.dpressure_drho

        self.R_eq = lg.R_eq
        self.Z_eq = lg.Z_eq
        self.theta_eq = lg.theta
        self.b_poloidal_eq = lg.b_poloidal_eq
        self.dpsidr = lg.dpsidr

        self.get_shape_coefficients(self.R_eq, self.Z_eq, self.b_poloidal_eq, verbose)

        # Bunit for GACODE codes
        self.bunit_over_b0 = self.get_bunit_over_b0()

        if show_fit:
            self.plot_fits()

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
    def plot_fits(self):
        pass

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
