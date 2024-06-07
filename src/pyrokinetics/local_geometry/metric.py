import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from . import LocalGeometry
from ..units import ureg


class MetricTerms:  # CleverDict
    r"""
    General geometry Object representing local LocalGeometry fit parameters

    Data stored in a ordered dictionary

    The following methods are used to access the metric tensor terms via the following

    - `toroidal_covariant_metric`
    - `toroidal_contravariant_metric`
    - `field_aligned_covariant_metric`
    - `field_aligned_contravariant_metric`

    Examples
    --------

    >>> g_r_r = MetricTerms.toroidal_covariant_metric("r", "r")

    Attributes
    ----------
    regulartheta : Array
        Evenly spaced theta grid
    R : Array
        Flux surface R (normalised to a_minor)
    Z : Array
        Flux surface Z (normalised to a_minor)

    dRdtheta : Array
        Derivative of :math:`R` w.r.t :math:`\theta`
    dRdr : Array
        Derivative of :math:`R` w.r.t :math:`r`
    dZdtheta : Array
        Derivative of :math:`Z` w.r.t :math:`\theta`
    dZdr : Array
        Derivative of :math:`Z` w.r.t :math:`r`


    d2Rdtheta2 : Array
        Second derivative of :math:`R` w.r.t :math:`\theta`
    d2Rdrdtheta : Array
        Second derivative of :math:`R` w.r.t :math:`r` and :math:`\theta`
    d2Zdtheta2 : Array
        Second derivative of :math:`Z` w.r.t :math:`\theta`
    d2Zdrdtheta : Array
        Second derivative of :math:`Z` w.r.t :math:`r` and :math:`theta`

    Jacobian : Array
        Jacobian of flux surface

    q : Float
        Safety factor
    dqdr : Float
        Derivative of :math:`q` w.r.t :math:`r`
    dpsidr : Float
        :math:`\partial \psi / \partial r`
    d2psidr2 : Float
        Second derivative of :math:`psi` w.r.t :math:`r` - arbitrary to set to 0
    mu0dPdr : Float
        :math:`mu_0 \partial p \partial r`

    sigma_alpha : Integer
        Sign to select if (r, alpha, theta) or (r, theta, alpha) is a RHS system

    field_coords : List
        List of field-aligned co-ordinates
    toroidal_coords : List
        List of toroidal co-ordinates

    dg_r_theta_dtheta : Array
        Derivative of toroidal covariant metric term g_r_theta w.r.t :math:`theta`

    dg_theta_theta_dr : Array
        Derivative of toroidal covariant metric term g_theta_theta w.r.t :math:`r`

    dg_zeta_zeta_dr : Array
        Derivative of toroidal covariant metric term g_zeta_zeta w.r.t :math:`r`

    dg_zeta_zeta_dtheta : Array
        Derivative of toroidal covariant metric term g_zeta_zeta w.r.t :math:`theta`

    """

    def __init__(self, local_geometry: LocalGeometry, ntheta=None, theta=None):
        if theta is not None and ntheta is not None:
            raise ValueError("Can't set both theta and ntheta, please select one")

        if theta is not None:
            if not np.all(np.isclose(np.diff(np.diff(theta)), 0)):
                raise ValueError("Specified theta is not evenly spaced")
            self.regulartheta = theta
        else:
            if ntheta is None:
                ntheta = 1024
                print(f"ntheta not specified, defaulting to {ntheta} points")
            self.regulartheta = np.linspace(-np.pi, np.pi, ntheta)  # theta grid

        if not isinstance(local_geometry, LocalGeometry):
            raise TypeError("local_geometry input must be of type LocalGeometry")

        # Range of theta
        self.theta_range = np.max(self.regulartheta) - np.min(self.regulartheta)

        # R and Z of flux surface (normalised to a_minor)
        self.R, self.Z = local_geometry.get_flux_surface(self.regulartheta)

        # 1st derivatives of R and Z
        (
            self.dRdtheta,
            self.dRdr,
            self.dZdtheta,
            self.dZdr,
        ) = local_geometry.get_RZ_derivatives(self.regulartheta)

        # 2nd derivatives of R and Z
        (
            self.d2Rdtheta2,
            self.d2Rdrdtheta,
            self.d2Zdtheta2,
            self.d2Zdrdtheta,
        ) = local_geometry.get_RZ_second_derivatives(self.regulartheta)

        # Jacobian, equation 9
        # NOTE: The Jacobians of the toroidal system and the field-aligned system are the same
        self.Jacobian = self.R * (self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta)

        # safety factor
        self.q = local_geometry.q

        # poloidal average of Jacobian * g^zetazeta: <Jacobian * g^zetazeta>,
        # e.g. the denominator of equation 16
        self.Y = integrate.trapezoid(
            (self.Jacobian / self.R**2).m, self.regulartheta
        ) / (self.theta_range)

        # This defines the reference magnetic field as B0:
        # dpsidr / (B0 * a) = <Jacobian * g^zetazeta> * (R0 / a) / q
        self.dpsidr = self.Y * local_geometry.Rmaj / self.q

        # rho is defined as r / a
        self.rho = local_geometry.rho

        # safety factor derivative
        self.dqdr = self.q * local_geometry.shat / self.rho

        # Second derivative of poloidal flux divided by 2 pi. Arbitrary
        # for local equilibria, take to be 0
        # d2psidr2_N = d2psidr2 / B0
        self.d2psidr2 = 0.0

        # mu0_N = mu0 * n_ref * T_ref / B0^2 = beta / 2 (normalised mu0)
        # dPdr_N = (a / (n_ref * T_ref)) * dPdr (normalised pressure gradient)
        # mu0dPdr_N = (a / B0^2) * mu0 * dPdr = beta_prime / 2 (normalised product)
        # Technically beta_prime should have units of a
        self.mu0dPdr = local_geometry.beta_prime.m / 2.0 / local_geometry.Rmaj.units

        # either 1 or -1, affects handedness of field-aligned system
        # If 1, (r, alpha, theta) forms RHS
        # If -1, (r, theta, alpha) forms RHS
        # see equations 23, 24
        self.sigma_alpha = 1

        # Specify allowed coordinates
        self.field_coords = np.array(["r", "alpha", "theta"])
        self.toroidal_coords = np.array(["r", "theta", "zeta"])

        # Initialise different metric tensors
        self._toroidal_covariant_metric = np.empty((3, 3), dtype=object)
        self._toroidal_contravariant_metric = np.empty((3, 3), dtype=object)
        self._field_aligned_covariant_metric = np.empty((3, 3), dtype=object)
        self._field_aligned_contravariant_metric = np.empty((3, 3), dtype=object)

        # Set up toroidal metric
        self.set_toroidal_covariant_metric()
        self.set_toroidal_covariant_metric_derivatives()
        self.set_toroidal_contravariant_metric()

        # Set up field aligned metric
        self.set_field_aligned_covariant_metric()
        self.set_field_aligned_contravariant_metric()

    def toroidal_covariant_metric(self, coord1: str, coord2: str):
        """Toroidal contravariant metric tensor at requested co-ordinate

        .. todo::
           Make type hints literals

        Parameters
        ----------
        coord1:
            Co-ordinate of first index in metric tensor.
            Can be r, theta, zeta

        coord2:
            Co-ordinate of second index in metric tensor.
            Can be r, theta, zeta
        """

        if coord1 not in self.toroidal_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.toroidal_coords}"
            )
        else:
            index1 = np.argwhere(self.toroidal_coords == coord1)

        if coord2 not in self.toroidal_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.toroidal_coords}"
            )
        else:
            index2 = np.argwhere(self.toroidal_coords == coord2)

        return self._toroidal_covariant_metric[index1, index2][0][0]

    def toroidal_contravariant_metric(self, coord1: str, coord2: str):
        """Toroidal contravariant metric tensor at requested co-ordinate

        Parameters
        ----------
        coord1:
            Co-ordinate of first index in metric tensor.
            Can be r, theta, zeta

        coord2:
            Co-ordinate of second index in metric tensor.
            Can be r, theta, zeta
        """

        if coord1 not in self.toroidal_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.toroidal_coords}"
            )
        else:
            index1 = np.argwhere(self.toroidal_coords == coord1)

        if coord2 not in self.toroidal_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.toroidal_coords}"
            )
        else:
            index2 = np.argwhere(self.toroidal_coords == coord2)

        return self._toroidal_contravariant_metric[index1, index2][0][0]

    def field_aligned_covariant_metric(self, coord1: str, coord2: str):
        """Field aligned covariant metric tensor at requested co-ordinate

        Parameters
        ----------
        coord1:
            Co-ordinate of first index in metric tensor.
            Can be r, alpha, theta

        coord2:
            Co-ordinate of second index in metric tensor.
            Can be r, alpha, theta
        """

        if coord1 not in self.field_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.field_coords}"
            )
        else:
            index1 = np.argwhere(self.field_coords == coord1)

        if coord2 not in self.field_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.field_coords}"
            )
        else:
            index2 = np.argwhere(self.field_coords == coord2)

        return self._field_aligned_covariant_metric[index1, index2][0][0]

    def field_aligned_contravariant_metric(self, coord1: str, coord2: str):
        """Field aligned contravariant metric tensor at requested co-ordinate

        Parameters
        ----------
        coord1:
            Co-ordinate of first index in metric tensor.
            Can be r, alpha, theta

        coord2:
            Co-ordinate of second index in metric tensor.
            Can be r, alpha, theta
        """

        if coord1 not in self.field_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.field_coords}"
            )
        else:
            index1 = np.argwhere(self.field_coords == coord1)

        if coord2 not in self.field_coords:
            raise ValueError(
                f"{coord1} is not an acceptable co-ordinates. Should be either {self.field_coords}"
            )
        else:
            index2 = np.argwhere(self.field_coords == coord2)

        return self._field_aligned_contravariant_metric[index1, index2][0][0]

    @property
    def B_zeta(self):
        """
        Returns
        -------
        B_zeta : Array
            B_zeta which is the current function (I or f) (equation 16)
        """

        return self.q * self.dpsidr / self.Y

    @property
    def dB_zeta_dr(self):
        r"""
        Returns
        -------
        dB_zeta_dr : Array
            Radial derivative of :math:`B_\zeta` w.r.t :math:`r` (equation 19)
        """
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_r_theta = self.toroidal_covariant_metric("r", "theta")

        # eq 20
        # units:
        # - Jacobian: lref**2
        # - gcont_zeta_zeta: lref**-2
        # - g_theta_theta: lref**2
        # Overall dimensionless
        H = self.Y + ((self.q / self.Y) ** 2) * (
            integrate.trapezoid(
                ((self.Jacobian**3) * (gcont_zeta_zeta**2) / g_theta_theta).m,
                self.regulartheta,
            )
            / self.theta_range
        )

        # Uses B_zeta / dpsidr = q / Y
        term1 = self.Y * self.dqdr / self.q

        # uses dg^zetazeta/dr = - (2 / R^3) * dRdr
        term2_units = self.Jacobian.units * self.dRdr.units / (self.R.units**3)

        @ureg.wraps(term2_units, (term2_units, None))
        def trapezoid_term2(y, x):
            return integrate.trapezoid(y, x)

        term2 = (
            -trapezoid_term2(
                -2.0 * self.Jacobian * self.dRdr / (self.R**3), self.regulartheta
            )
            / self.theta_range
        )

        term3_units = (
            (self.Jacobian.units**3) * gcont_zeta_zeta.units / g_theta_theta.units
        )

        @ureg.wraps(term3_units, (term3_units, None))
        def trapezoid_term3(y, x):
            return integrate.trapezoid(y, x)

        term3 = -(self.mu0dPdr / (self.dpsidr**2)) * (
            trapezoid_term3(
                (self.Jacobian**3) * gcont_zeta_zeta / g_theta_theta,
                self.regulartheta,
            )
            / self.theta_range
        )

        # integrand of fourth term
        to_integrate = (self.Jacobian * gcont_zeta_zeta / g_theta_theta) * (
            self.dg_r_theta_dtheta
            - self.dg_theta_theta_dr
            - (g_r_theta * self.dJacobian_dtheta / self.Jacobian)
        )

        @ureg.wraps(to_integrate.units, (to_integrate.units, None))
        def trapezoid_term4(y, x):
            return integrate.trapezoid(y, x)

        term4 = trapezoid_term4(to_integrate, self.regulartheta) / self.theta_range

        # eq 19
        return (self.B_zeta / H) * (term1 + term2 + term3 + term4)

    @property
    def B_magnitude(self):
        """
        Returns
        -------
        B_magnitude : Array
            Magntiude of total field
        """

        g_theta_theta = self.field_aligned_covariant_metric("theta", "theta")

        return self.dpsidr * np.sqrt(g_theta_theta) / self.Jacobian

    @property
    def dB_magnitude_dr(self):
        """
        Returns
        -------
        dB_magnitude_dr : Array
            Derivative of magntiude of total field w.r.t r
        """

        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")

        return (1.0 / (2 * self.B_magnitude)) * (
            self.dg_theta_theta_dr * self.B_theta**2
            + 2 * g_theta_theta * self.B_theta * self.dB_theta_drho
            + self.dg_zeta_zeta_dr * self.B_zeta**2
            + 2 / self.R**2 * self.B_zeta * self.dB_zeta_dr
        )

    @property
    def dB_magnitude_dtheta(self):
        """
        Returns
        -------
        dB_magnitude_dr : Array
            Derivative of magntiude of total field w.r.t theta
        """

        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")

        return (
            1.0
            / (2 * self.B_magnitude)
            * (
                self.dg_theta_theta_dtheta * self.B_theta**2
                + 2 * g_theta_theta * self.B_theta * self.dB_theta_dtheta
                + self.dg_zeta_zeta_dtheta * self.B_zeta**2
            )
        )

    @property
    def B_theta(self):
        """
        Returns
        -------
        B_theta : Array
            Poloidal field in toroidal coordinates
        """

        return self.dpsidr / self.Jacobian

    @property
    def dB_theta_drho(self):
        """
        Returns
        -------
        dB_theta_drho : Array
            Derivative of B_theta w.r.t :math:`r`
        """

        return -self.dpsidr / self.Jacobian**2 * self.dJacobian_dr

    @property
    def dB_theta_dtheta(self):
        """
        Returns
        -------
        dB_theta_drho : Array
            Derivative of B_theta w.r.t :math:`r`
        """

        return -self.dpsidr / self.Jacobian**2 * self.dJacobian_dtheta

    @property
    def dJacobian_dtheta(self):
        """
        Differentiate eq 9 w.r.t theta

        Returns
        -------
        dJacobian_dtheta : Array
            Derivative of Jacobian w.r.t :math:`\theta`
        """
        return (self.dRdtheta * self.Jacobian / self.R) + self.R * (
            self.d2Rdrdtheta * self.dZdtheta
            + self.dRdr * self.d2Zdtheta2
            - self.d2Rdtheta2 * self.dZdr
            - self.dRdtheta * self.d2Zdrdtheta
        )

    @property
    def dJacobian_dr(self):
        """
        Returns
        -------
        dJacobian_dr : Array
            Derivative of Jacobian w.r.t :math:`r` (equation 21)
        """
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_r_theta = self.toroidal_covariant_metric("r", "theta")

        term1 = self.Jacobian * self.d2psidr2 / self.dpsidr
        term2 = -(self.Jacobian / g_theta_theta) * (
            self.dg_r_theta_dtheta
            - self.dg_theta_theta_dr
            - (g_r_theta * self.dJacobian_dtheta / self.Jacobian)
        )
        term3 = (self.mu0dPdr / (self.dpsidr**2)) * (self.Jacobian**3) / g_theta_theta
        term4 = (
            (self.B_zeta * self.dB_zeta_dr / (self.dpsidr**2))
            * (self.Jacobian**3)
            * gcont_zeta_zeta
            / g_theta_theta
        )

        # eq 21
        return term1 + term2 + term3 + term4

    @property
    def dalpha_dtheta(self):
        r"""
        Returns
        -------
        dalpha_dtheta : Array
            Derivative of alpha w.r.t :math:`\theta` (equation 37)
        """
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")

        return self.sigma_alpha * (self.q * self.Jacobian * gcont_zeta_zeta / self.Y)

    @property
    def d2alpha_drdtheta(self):
        r"""
        Returns
        -------
        d2alpha_drdtheta : Array
            Second derivative of alpha w.r.t :math:`\theta` and :math:`r` (equation 38)
        """
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")

        term1 = self.dB_zeta_dr * self.Jacobian * gcont_zeta_zeta / self.dpsidr
        term2 = (
            -self.d2psidr2
            * self.Jacobian
            * gcont_zeta_zeta
            * self.B_zeta
            / (self.dpsidr**2)
        )
        term3 = self.B_zeta * self.dJacobian_dr * gcont_zeta_zeta / self.dpsidr
        term4 = -(2.0 * self.dRdr / (self.R**3)) * (
            self.B_zeta * self.Jacobian / self.dpsidr
        )
        return self.sigma_alpha * (term1 + term2 + term3 + term4)

    @property
    def dalpha_dr(self):
        r"""
        Equation 39
        inherits correct :math:`\sigma_\alpha` from `d2alpha_drdtheta`
        integrate over theta

        Returns
        -------
        dalpha_dr : Array
            Derivative of alpha w.r.t :math:`r`
        """
        initial_units = self.d2alpha_drdtheta.units

        # cumulative_trapezoid strips units, integration adds no unit
        dalpha_dr = integrate.cumulative_trapezoid(
            self.d2alpha_drdtheta.m, self.regulartheta, initial=0.0
        )

        f = interp1d(self.regulartheta, dalpha_dr)

        # set dalpha/dr(r,theta=0.0)=0.0, assumed by codes, add unit back
        return (dalpha_dr - f(0.0)) * initial_units

    @property
    def alpha(self):
        r"""
        Equation 37
        inherits correct :math:`\sigma_\alpha` from `dalpha_dtheta`
        integrate over theta

        Returns
        -------
        dalpha_dr : Array
            Derivative of alpha w.r.t :math:`r`
        """
        initial_units = self.dalpha_dtheta.units

        # cumulative_trapezoid strips units, integration adds no unit
        dalpha_dtheta = integrate.cumulative_trapezoid(
            self.dalpha_dtheta, self.regulartheta, initial=0.0
        )
        f = interp1d(self.regulartheta, dalpha_dtheta)

        # set dalpha/dr(r,theta=0.0)=0.0, assumed by codes, add unit back
        return (dalpha_dtheta - f(0.0)) * initial_units

    def set_toroidal_covariant_metric(self):
        """
        Sets up toroidal covariant metric tensor

        """

        # g_r_r: eq 4
        self._toroidal_covariant_metric[0, 0] = self.dRdr**2 + self.dZdr**2

        # g_r_theta: eq 5
        self._toroidal_covariant_metric[1, 0] = (
            self.dRdr * self.dRdtheta + self.dZdr * self.dZdtheta
        )
        self._toroidal_covariant_metric[0, 1] = self._toroidal_covariant_metric[1, 0]

        # g_theta_theta: eq 6
        self._toroidal_covariant_metric[1, 1] = self.dRdtheta**2 + self.dZdtheta**2

        # g_zeta_zeta: eq 8
        self._toroidal_covariant_metric[2, 2] = self.R**2

    def set_toroidal_covariant_metric_derivatives(self):
        """
        Sets up required terms of derivative of toroidal covariant metric tensor

        """

        # differentiate eq 5 w.r.t theta
        self.dg_r_theta_dtheta = (
            self.d2Rdrdtheta * self.dRdtheta
            + self.d2Rdtheta2 * self.dRdr
            + self.d2Zdrdtheta * self.dZdtheta
            + self.d2Zdtheta2 * self.dZdr
        )

        # differentiate eq 6 w.r.t r
        self.dg_theta_theta_dr = 2 * (
            self.dRdtheta * self.d2Rdrdtheta + self.dZdtheta * self.d2Zdrdtheta
        )

        # differentiate eq 6 w.r.t theta
        self.dg_theta_theta_dtheta = (
            2 * self.dRdtheta * self.d2Rdtheta2 + 2 * self.dZdtheta * self.d2Zdtheta2
        )

        # differentiate eq 7 w.r.t r
        self.dg_zeta_zeta_dr = -2 / self.R**3 * self.dRdr

        # differentiate eq 7 w.r.t theta
        self.dg_zeta_zeta_dtheta = -2 / self.R**3 * self.dRdtheta

    def set_toroidal_contravariant_metric(self):
        """
        Sets contravariant metric components of toroidal system using covariant components
        """
        g_r_r = self.toroidal_covariant_metric("r", "r")
        g_r_theta = self.toroidal_covariant_metric("r", "theta")
        g_zeta_zeta = self.toroidal_covariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")

        # g^r^r: eq 10
        self._toroidal_contravariant_metric[0, 0] = (
            g_theta_theta * g_zeta_zeta / self.Jacobian**2
        )

        # g^r^theta: eq 11
        self._toroidal_contravariant_metric[0, 1] = -(
            g_r_theta * g_zeta_zeta / self.Jacobian**2
        )

        # g^theta^r
        self._toroidal_contravariant_metric[1, 0] = self._toroidal_contravariant_metric[
            0, 1
        ]

        # g^theta^theta: eq 12
        self._toroidal_contravariant_metric[1, 1] = (
            g_r_r * g_zeta_zeta / self.Jacobian**2
        )

        # g^zeta^zeta: eq 14
        self._toroidal_contravariant_metric[2, 2] = 1 / self.R**2

    def set_field_aligned_covariant_metric(self):
        """
        Sets up field-aligned covariant metric tensor

        """
        # covariant toroidal metric terms
        g_r_r = self.toroidal_covariant_metric("r", "r")
        g_zeta_zeta = self.toroidal_covariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_r_theta = self.toroidal_covariant_metric("r", "theta")

        # tilde{g}_r_r: eq 25
        self._field_aligned_covariant_metric[0, 0] = (
            g_r_r + (self.dalpha_dr**2) * g_zeta_zeta
        )

        # tilde{g}_r_alpha : eq 26
        self._field_aligned_covariant_metric[0, 1] = -self.dalpha_dr * g_zeta_zeta

        # tilde{g}_alpha_r
        self._field_aligned_covariant_metric[1, 0] = (
            self._field_aligned_covariant_metric[0, 1]
        )

        # tilde{g}_r_theta: eq 27
        self._field_aligned_covariant_metric[0, 2] = (
            g_r_theta + self.dalpha_dr * self.dalpha_dtheta * g_zeta_zeta
        )

        # tilde{g}_theta_r
        self._field_aligned_covariant_metric[2, 0] = (
            self._field_aligned_covariant_metric[0, 2]
        )

        # tilde{g}_alpha_alpha: eq 28
        self._field_aligned_covariant_metric[1, 1] = g_zeta_zeta

        # tilde{g}_alpha_theta: eq 29
        self._field_aligned_covariant_metric[1, 2] = -self.dalpha_dtheta * g_zeta_zeta

        # tilde{g}_theta_alpha
        self._field_aligned_covariant_metric[2, 1] = (
            self._field_aligned_covariant_metric[1, 2]
        )

        # tilde{g}_theta_theta: eq 30
        self._field_aligned_covariant_metric[2, 2] = (
            g_theta_theta + (self.dalpha_dtheta**2) * g_zeta_zeta
        )

    def set_field_aligned_contravariant_metric(self):
        """
        Sets up field-aligned contravariant metric tensor

        """
        # contravariant toroidal metric terms
        gcont_r_r = self.toroidal_contravariant_metric("r", "r")
        gcont_r_theta = self.toroidal_contravariant_metric("r", "theta")
        gcont_theta_theta = self.toroidal_contravariant_metric("theta", "theta")
        # covariant field-aligned metric terms
        gf_r_r = self.field_aligned_covariant_metric("r", "r")
        gf_r_theta = self.field_aligned_covariant_metric("r", "theta")
        gf_theta_theta = self.field_aligned_covariant_metric("theta", "theta")

        # tilde{g}^r^r: eq 31
        self._field_aligned_contravariant_metric[0, 0] = gcont_r_r

        # tilde{g}^r^theta: eq 34
        self._field_aligned_contravariant_metric[0, 2] = gcont_r_theta
        # tilde{g}^theta^r
        self._field_aligned_contravariant_metric[2, 0] = (
            self._field_aligned_contravariant_metric[0, 2]
        )

        # tilde{g}^theta^theta: eq 35
        self._field_aligned_contravariant_metric[2, 2] = gcont_theta_theta

        # tilde{g}^r^alpha: eq 32
        self._field_aligned_contravariant_metric[0, 1] = (
            self.dalpha_dr * gcont_r_r + self.dalpha_dtheta * gcont_r_theta
        )
        # tilde{g}^alpha^r
        self._field_aligned_contravariant_metric[1, 0] = (
            self._field_aligned_contravariant_metric[1, 2]
        )

        # tilde{g}^theta^alpha: eq 36
        self._field_aligned_contravariant_metric[2, 1] = (
            self.dalpha_dr * gcont_r_theta + self.dalpha_dtheta * gcont_theta_theta
        )
        # tilde{g}^alpha^theta
        self._field_aligned_contravariant_metric[1, 2] = (
            self._field_aligned_contravariant_metric[2, 1]
        )

        # tilde{g}^alpha^alpha: eq 33
        self._field_aligned_contravariant_metric[1, 1] = (
            gf_r_r * gf_theta_theta - (gf_r_theta**2)
        ) / (self.Jacobian**2)

    def k_perp(self, ky: float, theta0: float, nperiod: int):
        r"""
        Equation 155

        Returns
        -------
        k_perp : Array
            Perpendicular wavevector along the field line
        """

        Cy = self.rho / self.q

        shat = Cy * self.dqdr

        # The total number of poloidal turns is 2*nperiod-1
        m = np.linspace(-(nperiod - 1), nperiod - 1, 2 * nperiod - 1)

        # Exclude last point to avoid duplicates
        theta = np.tile(self.regulartheta[:-1], 2 * nperiod - 1)

        ntheta = len(self.regulartheta) - 1

        m = np.repeat(m, ntheta)

        theta = theta + 2.0 * np.pi * m

        g_rr = self.field_aligned_contravariant_metric("r", "r")[:-1]
        g_ra = self.field_aligned_contravariant_metric("r", "alpha")[:-1]
        g_aa = self.field_aligned_contravariant_metric("alpha", "alpha")[:-1]

        g_xx = np.tile(g_rr, 2 * nperiod - 1)
        g_xy = np.tile(g_ra, 2 * nperiod - 1) * Cy
        g_yy = np.tile(g_aa, 2 * nperiod - 1) * Cy**2

        # Actually kx / ky
        kx = shat * (theta0 + m * 2.0 * np.pi)

        k_perp2 = g_xx * kx**2 + 2.0 * g_xy * kx + g_yy

        # Need to normalise to ky
        k_perp = np.sqrt(k_perp2) * ky

        return theta, k_perp
