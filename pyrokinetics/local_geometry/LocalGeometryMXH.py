import numpy as np
from typing import Tuple
from scipy.optimize import least_squares  # type: ignore
from scipy.integrate import simpson
from .LocalGeometry import LocalGeometry
from ..equilibrium import Equilibrium
from ..typing import ArrayLike
from .LocalGeometry import default_inputs


def default_mxh_inputs(n_moments=4):
    # Return default args to build a LocalGeometryMXH
    # Uses a function call to avoid the user modifying these values

    base_defaults = default_inputs()
    mxh_defaults = {
        "cn": np.zeros(n_moments),
        "dcndr": np.zeros(n_moments),
        "sn": np.zeros(n_moments),
        "dsndr": np.zeros(n_moments),
        "local_geometry": "MXH",
    }

    return {**base_defaults, **mxh_defaults}


class LocalGeometryMXH(LocalGeometry):
    r"""
    Local equilibrium representation defined as in:
    Plasma Phys. Control. Fusion 63 (2021) 012001 (5pp) https://doi.org/10.1088/1361-6587/abc63b
    Miller eXtended Harmonic (MXH)

    R(r, theta) = Rmajor(r) + r * cos(thetaR)
    Z(r, theta) = Z0(r) + r * kappa(r) * sin(theta)

    thetaR = theta + c0(r) + sum_n=1^N [cn(r) * cos(n*theta) + sn(r) * sin(n*theta)]

    r = (max(R) - min(R)) / 2

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
        Ratio of GACODE normalising field = :math:`q/r \partial \psi/\partial r` [T] to B0
    dpsidr : Float
        :math: `\partial \psi / \partial r`
    q : Float
        Safety factor
    shat : Float
        Magnetic shear `r/q \partial q/ \partial r`
    beta_prime : Float
        :math:`\beta' = `2 \mu_0 \partial p \partial \rho 1/B0^2`

    kappa : Float
        Elongation
    dkapdr : Float
        Derivative of kappa w.r.t r
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
        Derivative of cosine moments w.r.t r
    dsndr : ArrayLike
        Derivative of sine moments w.r.t r

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

        if (
            args
            and not isinstance(args[0], LocalGeometryMXH)
            and isinstance(args[0], dict)
        ):
            s_args[0] = sorted(args[0].items())

            super(LocalGeometry, self).__init__(*s_args, **kwargs)

        elif len(args) == 0:
            self.default()

    def from_global_eq(
        self, eq: Equilibrium, psi_n: float, verbose=False, n_moments=4, show_fit=False
    ):
        r"""
        Loads MXH object from a GlobalEquilibrium Object

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
        n_moments : Int
            Sets number of moments to be used in fit
        show_fit : Boolean
            Controls whether fit vs equilibrium is plotted
        """

        self.n_moments = n_moments

        drho_dpsi = eq.rho.derivative()(psi_n)
        shift = eq.R_major.derivative()(psi_n) / drho_dpsi / eq.a_minor

        super().from_global_eq(
            eq=eq, psi_n=psi_n, verbose=verbose, shift=shift, show_fit=show_fit
        )

    def from_local_geometry(
        self, local_geometry: LocalGeometry, verbose=False, n_moments=4, show_fit=False
    ):
        r"""
        Loads MXH object from an existing LocalGeometry Object

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
        n_moments : Int
            Sets number of moments to be used in fit
        show_fit : Boolean
            Controls whether fit vs equilibrium is plotted
        """

        self.n_moments = n_moments

        super().from_local_geometry(
            local_geometry=local_geometry, verbose=verbose, show_fit=show_fit
        )

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
            `b_\theta` for the given flux surface
        verbose : Boolean
            Controls verbosity
        shift : Float
            Initial guess for shafranov shift
        """

        kappa = (max(Z) - min(Z)) / (2 * self.r_minor)

        Zmid = (max(Z) + min(Z)) / 2

        Zind = np.argmax(abs(Z))

        R_upper = R[Zind]

        normalised_height = (Z - Zmid) / (kappa * self.r_minor)

        # Floating point error can lead to >|1.0|
        normalised_height = np.where(
            np.isclose(normalised_height, 1.0), 1.0, normalised_height
        )
        normalised_height = np.where(
            np.isclose(normalised_height, -1.0), -1.0, normalised_height
        )

        theta = np.arcsin(normalised_height)

        normalised_radius = (R - self.Rmaj * self.a_minor) / self.r_minor

        normalised_radius = np.where(
            np.isclose(normalised_radius, 1.0), 1.0, normalised_radius
        )
        normalised_radius = np.where(
            np.isclose(normalised_radius, -1.0), -1.0, normalised_radius
        )

        thetaR = np.arccos(normalised_radius)

        theta = np.where(R < R_upper, np.pi - theta, theta)
        theta = np.where((R >= R_upper) & (Z < 0), 2 * np.pi + theta, theta)
        thetaR = np.where(Z < 0, 2 * np.pi - thetaR, thetaR)

        self.theta_eq = theta

        # Ensure theta start from zero
        theta = np.roll(theta, -np.argmin(theta))
        thetaR = np.roll(thetaR, -np.argmin(thetaR))

        theta_diff = thetaR - theta

        ntheta = np.outer(self.n, theta)

        cn = simpson(theta_diff * np.cos(ntheta), theta, axis=1) / np.pi
        sn = simpson(theta_diff * np.sin(ntheta), theta, axis=1) / np.pi

        self.kappa = kappa
        self.sn = sn
        self.cn = cn

        self.theta = theta
        self.thetaR = self.get_thetaR(self.theta)
        self.dthetaR_dtheta = self.get_dthetaR_dtheta(self.theta)

        self.R, self.Z = self.get_flux_surface(self.theta)

        dkap_dr_init = 0.0
        params = [shift, dkap_dr_init, 0.0, *[0.0] * self.n_moments * 2]

        fits = least_squares(self.minimise_b_poloidal, params)

        # Check that least squares didn't fail
        if not fits.success:
            raise Exception(
                f"Least squares fitting in MXH::from_global_eq failed with message : {fits.message}"
            )

        if verbose:
            print(f"MXH :: Fit to Bpoloidal obtained with residual {fits.cost}")

        if fits.cost > 0.1:
            import warnings

            warnings.warn(
                f"Warning Fit to Bpoloidal in MXH::from_global_eq is poor with residual of {fits.cost}"
            )

        self.shift = fits.x[0]
        self.dkapdr = fits.x[1]
        self.s_kappa = self.r_minor / self.kappa * self.dkapdr
        self.dZ0dr = fits.x[2]
        self.dcndr = fits.x[3 : self.n_moments + 3]
        self.dsndr = fits.x[self.n_moments + 3 :]

        self.dthetaR_dr = self.get_dthetaR_dr(self.theta, self.dcndr, self.dsndr)

    @property
    def n(self):
        return np.linspace(0, self.n_moments - 1, self.n_moments)

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

        ntheta = np.outer(theta, self.n)

        d2thetaR_dtheta2 = - np.sum(
            ((self.n ** 2) * (self.cn * np.cos(ntheta) + self.sn * np.sin(ntheta))),
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
        normalised=False,
    ) -> np.ndarray:
        """
        Calculates the derivatives of `R(r, \theta)` and `Z(r, \theta)` w.r.t `r` and `\theta`, used in B_poloidal calc

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        params : Array [Optional]
            If given then will use params = [shift, dkap_dr, dZ0dr, cn[nmoments], sn[nmoments] ] when calculating
            derivatives, otherwise will use object attributes
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        dRdtheta : Array
            Derivative of `R` w.r.t `\theta`
        dRdr : Array
            Derivative of `R` w.r.t `r`
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """

        if params is None:
            shift = self.shift
            dkapdr = self.dkapdr
            dZ0dr = self.dZ0dr
            dcndr = self.dcndr
            dsndr = self.dsndr
        else:
            shift = params[0]
            dkapdr = params[1]
            dZ0dr = params[2]
            dcndr = params[3 : self.n_moments + 3]
            dsndr = params[self.n_moments + 3 :]

        thetaR = self.get_thetaR(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, dcndr, dsndr)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)

        dZdtheta = self.get_dZdtheta(theta)

        dZdr = self.get_dZdr(theta, dZ0dr, dkapdr)

        dRdtheta = self.get_dRdtheta(thetaR, dthetaR_dtheta)

        dRdr = self.get_dRdr(shift, thetaR, dthetaR_dr)

        return dRdtheta, dRdr, dZdtheta, dZdr
    
    def get_RZ_second_derivatives(
        self,
        theta: ArrayLike,
        normalised=False,
        ) -> np.ndarray:
        """
        Calculates the second derivatives of `R(r, \theta)` and `Z(r, \theta)` w.r.t `r` and `\theta`, used in geometry terms

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate grad_r on
        normalised : Boolean
            Control whether or not to return normalised values
        Returns
        -------
        d2Rdtheta2 : Array
                        Second derivative of `R` w.r.t `\theta`
        d2Rdrdtheta : Array
                        Second derivative of `R` w.r.t `r` and `\theta`
        d2Zdtheta2 : Array
                        Second derivative of `Z` w.r.t `\theta`
        d2Zdrdtheta : Array
                        Second derivative of `Z` w.r.t `r` and `\theta`
        """
        
        thetaR = self.get_thetaR(theta)
        dthetaR_dr = self.get_dthetaR_dr(theta, self.dcndr, self.dsndr)
        dthetaR_dtheta = self.get_dthetaR_dtheta(theta)
        d2thetaR_drdtheta = self.get_d2thetaR_drdtheta(theta, self.dcndr, self.dsndr)
        d2thetaR_dtheta2 = self.get_d2thetaR_dtheta2(theta)

        d2Zdtheta2 = self.get_d2Zdtheta2(theta)
        d2Zdrdtheta = self.get_d2Zdrdtheta(theta, self.dkapdr)
        d2Rdtheta2 = self.get_d2Rdtheta2(thetaR, dthetaR_dtheta, d2thetaR_dtheta2)
        d2Rdrdtheta = self.get_d2Rdrdtheta(thetaR, dthetaR_dr, dthetaR_dtheta, d2thetaR_drdtheta)

        return d2Rdtheta2, d2Rdrdtheta, d2Zdtheta2, d2Zdrdtheta

    def get_dZdtheta(self, theta):
        """
        Calculates the derivatives of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        Returns
        -------
        dZdtheta : Array
            Derivative of `Z` w.r.t `\theta`
        """

        return self.kappa * self.rho * np.cos(theta)
        
    def get_d2Zdtheta2(self, theta):
        """
        Calculates the second derivative of `Z(r, theta)` w.r.t `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdtheta on
        Returns
        -------
        d2Zdtheta2 : Array
            Second derivative of `Z` w.r.t `\theta`
        """

        return - self.kappa * self.rho * np.sin(theta)

    def get_dZdr(self, theta, dZ0dr, dkapdr):
        """
        Calculates the derivatives of `Z(r, \theta)` w.r.t `r`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dZ0dr : Float
            Derivative in midplane elevation
        dkapdr : Float
            Derivative of kappa w.r.t r
        Returns
        -------
        dZdr : Array
            Derivative of `Z` w.r.t `r`
        """
        return dZ0dr + self.kappa * np.sin(theta) + dkapdr * self.rho * np.sin(theta)
    
    def get_d2Zdrdtheta(self, theta, dkapdr):
        """
        Calculates the second derivative of `Z(r, \theta)` w.r.t `r` and `\theta`

        Parameters
        ----------
        theta: ArrayLike
            Array of theta points to evaluate dZdr on
        dkapdr : Float
            Derivative of kappa w.r.t r
        Returns
        -------
        d2Zdrdtheta : Array
            Second derivative of `Z` w.r.t `r` and `\theta`
        """
        return self.kappa * np.cos(theta) + dkapdr * self.rho * np.cos(theta)

    def get_dRdtheta(self, thetaR, dthetaR_dtheta):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `\theta`

        Parameters
        ----------
        thetaR: ArrayLike
            Array of thetaR points to evaluate dRdtheta on
        dthetaR_dtheta : ArrayLike
            Theta derivative of thetaR
        -------
        dRdtheta : Array
            Derivative of `R` w.r.t `\theta`
        """
        return -self.rho * np.sin(thetaR) * dthetaR_dtheta
    
    def get_d2Rdtheta2(self, thetaR, dthetaR_dtheta, d2thetaR_dtheta2):
        """
        Calculates the second derivative of `R(r, \theta)` w.r.t `\theta`

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
            Second derivative of `R` w.r.t `\theta`
        """
        return -self.rho * np.sin(thetaR) * d2thetaR_dtheta2 - self.rho * (dthetaR_dtheta ** 2) * np.cos(thetaR)

    def get_dRdr(self, shift, thetaR, dthetaR_dr):
        """
        Calculates the derivatives of `R(r, \theta)` w.r.t `r`

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
            Derivative of `R` w.r.t `r`
        """
        return shift + np.cos(thetaR) - self.rho * np.sin(thetaR) * dthetaR_dr
             
    def get_d2Rdrdtheta(self, thetaR, dthetaR_dr, dthetaR_dtheta, d2thetaR_drdtheta):
        """
        Calculate the second derivative of `R(r, \theta)` w.r.t `r` and `\theta`

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
            Second derivative of thetaR w.r.t `r` and `\theta`
        Returns
        -------
        d2Rdrdtheta : Array
            Second derivative of R w.r.t `r` and `\theta`
        """
        return - dthetaR_dtheta *  np.sin(thetaR) - self.rho * np.sin(thetaR) * d2thetaR_drdtheta - self.rho * dthetaR_dr * dthetaR_dtheta * np.cos(thetaR)

    def get_flux_surface(
        self,
        theta: ArrayLike,
        normalised=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates (R,Z) of a flux surface given a set of MXH fits

        Parameters
        ----------
        theta : Array
            Values of theta to evaluate flux surface
        normalised : Boolean
            Control whether or not to return normalised flux surface
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

        if not normalised:
            R *= self.a_minor
            Z *= self.a_minor

        return R, Z

    def default(self):
        """
        Default parameters for geometry
        Same as GA-STD case
        """
        super(LocalGeometryMXH, self).__init__(default_mxh_inputs())
