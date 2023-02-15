import numpy as np
from . import LocalGeometry
import scipy.integrate as integrate
from scipy.interpolate import interp1d


class MetricTerms:  # CleverDict
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
        :math:`\beta' = 2 \mu_0 \partial p \partial \rho 1/B0^2`

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


    This class uses two (somewhat similar) coordinate systems. These are the 'toroidal system' and the 'field-aligned system'.
    The toroidal system is defined on page 73, with symbols {\varrho, \vartheta, \zeta}. For the purpose of variable names,
    we shall use {tr, tt, zeta}. (tr = 'toroidal system radial', tt = 'toroidal system theta').
    The field-aligned system is defined on page 82, with symbols {r, \alpha, \theta}. For the purpose of variable names,
    we shall use {fr, alpha, ft}. (fr = 'field-aligned radial', ft = 'field-aligned theta').

    Even though the transformations are defined such that tr = fr and tt = ft (see equations 3.149-3.151), we change
    the variable names due to the metric terms, which become ambiguous if the same symbol is used.
    For example, looking at equation D.87, one finds g_ftft = g_tttt + (q + dG_0dtheta)^2 * g_zetazeta. This would be
    unclear if both g_ftft and g_tttt were labelled 'g_thetatheta'.

    Note that for axisymmetric quantities (which is almost all geometrical quantities in tokamaks) partial derivatives
    with respect to the radial direction and the theta direction in the two systems are identical (see equation D.91).
    Where this is the case, 'r' and 'theta' will be used in the variable names, to keep more in line with Pyrokinetics convention.

    Almost all calculations are done using normalised quantities, for which the variable names have a subscript '_N'. The
    relevant normalising quantities are the minor radius 'a' for the length, and the magnetic field B0 = q dpsidr / (R0 * < Jacobian g^zetazeta >)
    Both partial derivatives and derivatives of univariate functions will be written using 'd'. (Little information is lost here).

    The \varrho (= 'tr') used in the thesis has dimensions of length, and is NOT the normalised rho = r / a defined by Pyrokinetics.
    In fact, rho = tr / a. Such is the nature of bringing tosether two works after they have been mainly written.
    """

    def __init__(self, local_geometry: LocalGeometry):
        self.regulartheta = np.linspace(-np.pi, np.pi, 219)  # theta grid

        self.R, self.Z = local_geometry.get_flux_surface(
            self.regulartheta, normalised=True
        )  # R/a, Z/a
        (
            self.dRdtheta,  # (dR/dtheta)/a
            self.dRdr,  # dR/dr, already normalised
            self.dZdtheta,  # (dZ/dtheta)/a
            self.dZdr,  # dZ/dr, already normalised
        ) = local_geometry.get_RZ_derivatives(self.regulartheta, normalised=True)
        (
            self.d2Rdtheta2,  # (d2R/dtheta2)/a
            self.d2Rdrdtheta,  # d2R/drdtheta, already normalised
            self.d2Zdtheta2,  # (d2Z/dtheta2)/a
            self.d2Zdrdtheta,  # d2Z/drdtheta, already normalised
        ) = local_geometry.get_RZ_second_derivatives(self.regulartheta, normalised=True)

        self.Jacobian = self.R * (
            self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta
        )  # Jacobian / a^2, equation D.35
        # NOTE: The Jacobians of the toroidal system and the
        # field-aligned system are the same

        self.q = local_geometry.q  # safety factor
        self.Y = integrate.trapezoid(self.Jacobian / self.R**2, self.regulartheta) / (
            2.0 * np.pi
        )  # frequently occuring quantity, already normalised.

        # poloidal average of Jacobian * g^zetazeta: <Jacobian * g^zetazeta>_P (see equation 3.133 for average definition)
        self.dpsidr = (
            self.Y * local_geometry.Rmaj / self.q
        )  # This defines the reference magnetic field as B0:
        # dpsidr_N = dpsidr / (B0 * a) = <Jacobian * g^zetazeta>_P * (R0 / a) / q
        self.dqdr = (
            self.q * local_geometry.shat / local_geometry.rho
        )  # safety factor derivative, dqdr_N = a * dqdr = q * shat / (r / a)
        self.d2psidr2 = 0.0  # Take d2psidr2 = 0. This quantity doesn't appear in any
        # physical quantities, and so is essentially arbitrary. d2psidr2_N = d2psidr2 / B0

        self.mu0dPdr = (
            local_geometry.beta_prime / 2.0
        )  # mu0_N = mu0 * n_ref * T_ref / B0^2 = beta / 2 (normalised mu0)
        # dPdr_N = (a / (n_ref * T_ref)) * dPdr (normalised pressure gradient)
        # mu0dPdr_N = (a / B0^2) * mu0 * dPdr = beta_prime / 2 (normalised product)
        self.sigma_alpha = 1  # either 1 or -1, affects field-aligned metric components (included after completion of thesis).
        # Defined via alpha = sigma_alpha * (q \vartheta - \zeta + G_0) (equation 3.150)
        # If 1, then {r, alpha, theta} forms a right-handed system (as in thesis, more 'x,y,z' style)
        # If -1, then {r, theta, alpha} forms a right-handed system (as in CGYRO)
        # In both cases, theta increases in the anti-clockwise direction

        # Specify allowed coordinates
        self.field_coords = np.array(["rho", "alpha", "theta"])
        self.toroidal_coords = np.array(["rho", "theta", "zeta"])

        # Initialise different metric tensors
        self._toroidal_covariant_metric = np.empty((3, 3), dtype=object)
        self._toroidal_contravariant_metric = np.empty((3, 3), dtype=object)
        self._field_aligned_covariant_metric = np.empty((3, 3), dtype=object)
        self._field_aligned_contravariant_metric = np.empty((3, 3), dtype=object)

        # Set up toroidal metric
        self.set_toroidal_covariant_metric()
        self.set_toroidal_covariant_metric_derivatives()
        self.set_toroidal_contravariant_metric()

        # Set up B_zeta derivatives plus Jacobian derivative
        self.set_B_zeta()
        self.set_dJacobian_dtheta()
        self.set_dB_zeta_dr()
        self.set_dJacobian_dr()

        # Set up alpha derivatives
        self.set_dalpha_dtheta()
        self.set_d2alpha_drdtheta()
        self.set_dalpha_dr()

        # Set up field aligned metric
        self.set_field_aligned_covariant_metric()
        self.set_field_aligned_contravariant_metric_components()

    def toroidal_covariant_metric(self, coord1, coord2):
        """

        Parameters
        ----------
        coord1 str
        Co-ordinate of first index in metric tensor
        Can be r, theta, zeta

        coord2 str
        Co-ordinate of second index in metric tensor
        Can be r, theta, zeta

        Returns
        -------
        Toroidal contravariant metric tensor at requested co-ordinate

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

    def toroidal_contravariant_metric(self, coord1, coord2):
        """

        Parameters
        ----------
        coord1 str
        Co-ordinate of first index in metric tensor
        Can be r, theta, zeta

        coord2 str
        Co-ordinate of second index in metric tensor
        Can be r, theta, zeta

        Returns
        -------
        Toroidal contravariant metric tensor at requested co-ordinate

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

    def field_aligned_covariant_metric(self, coord1, coord2):
        """

        Parameters
        ----------
        coord1 str
        Co-ordinate of first index in metric tensor
        Can be r, theta, zeta

        coord2 str
        Co-ordinate of second index in metric tensor
        Can be r, theta, zeta

        Returns
        -------
        Field aligned covariant metric tensor at requested co-ordinate

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

        return self._field_aligned_covariant_metric[index1, index2]

    def field_aligned_contravariant_metric(self, coord1, coord2):
        """

        Parameters
        ----------
        coord1 str
        Co-ordinate of first index in metric tensor
        Can be r, theta, zeta

        coord2 str
        Co-ordinate of second index in metric tensor
        Can be r, theta, zeta

        Returns
        -------
        Field aligned contravariant metric tensor at requested co-ordinate

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

        return self._field_aligned_contravariant_metric[index1, index2]

    def set_toroidal_covariant_metric(self):
        # g_rho rho
        self._toroidal_covariant_metric[0, 0] = (
            self.dRdr**2 + self.dZdr**2
        )  # eq D.30, already normalised

        # g_rho theta
        self._toroidal_covariant_metric[1, 0] = (
            self.dRdr * self.dRdtheta + self.dZdr * self.dZdtheta
        )  # eq D.31, g_trtt / a
        self._toroidal_covariant_metric[0, 1] = self._toroidal_covariant_metric[1, 0]

        # g_theta theta
        self._toroidal_covariant_metric[1, 1] = (
            self.dRdtheta**2 + self.dZdtheta**2
        )  # eq D.32, g_tttt / a^2

        # g_theta theta
        self._toroidal_covariant_metric[2, 2] = self.R ** 2  # eq D.33, g_zetazeta / a^2

    def set_toroidal_covariant_metric_derivatives(self):
        self.dg_rho_theta_dtheta = (
            self.d2Rdrdtheta * self.dRdtheta
            + self.d2Rdtheta2 * self.dRdr
            + self.d2Zdrdtheta * self.dZdtheta
            + self.d2Zdtheta2 * self.dZdr
        )  # differentiate eq D.31 w.r.t theta, dg_rho_theta_dtheta / a

        self.dg_theta_theta_drho = 2 * (
            self.dRdtheta * self.d2Rdrdtheta + self.dZdtheta * self.d2Zdrdtheta
        )  # differentiate eq D.32 w.r.t r, dg_theta_theta_drho / a

    # calculate contravariant metric components of toroidal system using covariant components and equation C.50
    def set_toroidal_contravariant_metric(self):
        g_rho_rho = self.toroidal_covariant_metric("rho", "rho")
        g_rho_theta = self.toroidal_covariant_metric("rho", "rho")
        g_zeta_zeta = self.toroidal_covariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")

        self._toroidal_contravariant_metric[0, 0] = (
            g_theta_theta * g_zeta_zeta / self.Jacobian**2
        )

        self._toroidal_contravariant_metric[0, 1] = -(
            g_rho_theta * g_zeta_zeta / self.Jacobian**2
        )
        self._toroidal_contravariant_metric[1, 0] = self._toroidal_contravariant_metric[
            0, 1
        ]

        self._toroidal_contravariant_metric[1, 1] = (
            g_rho_rho * g_zeta_zeta / self.Jacobian**2
        )

        self._toroidal_contravariant_metric[2, 2] = (
            1 / self.R**2
        )  # a^2 * g^zetazeta = 1 / (R / a)^2, equation D.34

    def set_B_zeta(self):
        self.B_zeta = (
            self.q * self.dpsidr / self.Y
        )  # equation 3.132, B_zeta / (B0 * a). Note B_zeta is a covariant component
        # and does NOT have dimensions of magnetic field, but of (magnetic field * length).
        # This is also known as the 'current function', I, and is a flux function.
        # Note B0 is defined as B0 = B_zeta / R0, and thus because of our normalisations,
        # self.B_zeta should equal R0 / a.

    def set_dB_zeta_dr(self):  # eq 3.139, dB_zeta_dr / B0
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_rho_theta = self.toroidal_covariant_metric("rho", "theta")

        H = self.Y + ((self.q / self.Y) ** 2) * (
            integrate.trapezoid(
                (self.Jacobian**3) * (gcont_zeta_zeta**2) / g_theta_theta,
                self.regulartheta,
            )
            / (2.0 * np.pi)
        )  # eq 3.140, already normalised.
        # Uses B_zeta / dpsidr = q / Y
        term1 = self.Y * self.dqdr / self.q
        term2 = -(
            integrate.trapezoid(
                -2.0 * self.Jacobian * self.dRdr / (self.R**3), self.regulartheta
            )
            / (2.0 * np.pi)
        )  # uses dg^zetazeta/dr = - (2 / R^3) * dRdr
        term3 = -(self.mu0dPdr / (self.dpsidr**2)) * (
            integrate.trapezoid(
                (self.Jacobian**3) * gcont_zeta_zeta / g_theta_theta,
                self.regulartheta,
            )
            / (2.0 * np.pi)
        )
        to_integrate = (self.Jacobian * gcont_zeta_zeta / g_theta_theta) * (
            self.dg_rho_theta_dtheta
            - self.dg_theta_theta_drho
            - (g_rho_theta * self.dJacobian_dtheta / self.Jacobian)
        )  # integrand of fourth term
        term4 = integrate.trapezoid(to_integrate, self.regulartheta) / (2.0 * np.pi)
        self.dB_zeta_dr = (self.B_zeta / H) * (
            term1 + term2 + term3 + term4
        )  # eq 3.139, dB_zeta_dr / B0

    def set_dJacobian_dtheta(self):  # eq 3.137, uses 3.139. (dJac/dr) / a
        self.dJacobian_dtheta = (self.dRdtheta * self.Jacobian / self.R) + self.R * (
            self.d2Rdrdtheta * self.dZdtheta
            + self.dRdr * self.d2Zdtheta2
            - self.d2Rdtheta2 * self.dZdr
            - self.dRdtheta * self.d2Zdrdtheta
        )  # differentiate eq D.35 w.r.t theta, dJacobiandtheta / a^2

    def set_dJacobian_dr(self):  # eq 3.137, uses 3.139. (dJac/dr) / a
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_rho_theta = self.toroidal_covariant_metric("rho", "theta")

        term1 = self.Jacobian * self.d2psidr2 / self.dpsidr
        term2 = -(self.Jacobian / g_theta_theta) * (
            self.dg_rho_theta_dtheta
            - self.dg_theta_theta_drho
            - (g_rho_theta * self.dJacobian_dtheta / self.Jacobian)
        )
        term3 = (
            (self.mu0dPdr / (self.dpsidr**2)) * (self.Jacobian**3) / g_theta_theta
        )
        term4 = (
            (self.B_zeta * self.dB_zeta_dr / (self.dpsidr**2))
            * (self.Jacobian**3)
            * gcont_zeta_zeta
            / g_theta_theta
        )
        self.dJacobian_dr = term1 + term2 + term3 + term4  # eq 3.137, (dJac/dr) / a

    def set_dalpha_dtheta(self):
        gcont_zeta_zeta = self.toroidal_contravariant_metric("zeta", "zeta")

        self.dalpha_dtheta = self.sigma_alpha * (
            self.q * self.Jacobian * gcont_zeta_zeta / self.Y
        )  # eq D.92, already normalised

    def set_d2alpha_drdtheta(
        self,
    ):  # eq D.93, sometimes known as 'local shear', a * d2alpha/drdtheta
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
        self.d2alpha_drdtheta = self.sigma_alpha * (term1 + term2 + term3 + term4)

    def set_dalpha_dr(
        self,
    ):  # eq D.94, obtained by integrating D.93 over theta. Calculation in document
        # is bigger as the form of dJac/dr has been written explicitly.
        # a * dalpha/dr
        # inherets correct sigma_alpha from self.set_d2alpha_drdtheta
        # integrate over theta

        dalpha_dr = integrate.cumulative_trapezoid(
            self.d2alpha_drdtheta, self.regulartheta
        )
        dalpha_dr = list(dalpha_dr)
        dalpha_dr.insert(0, 0.0)
        dalpha_dr = np.array(dalpha_dr)
        f = interp1d(self.regulartheta, dalpha_dr)
        self.dalpha_dr = dalpha_dr - f(
            0.0
        )  # set dalpha/dr(r,theta=0.0)=0.0, assumed by codes

    def set_field_aligned_covariant_metric(self):
        g_rho_rho = self.toroidal_covariant_metric("rho", "rho")
        g_zeta_zeta = self.toroidal_covariant_metric("zeta", "zeta")
        g_theta_theta = self.toroidal_covariant_metric("theta", "theta")
        g_rho_theta = self.toroidal_covariant_metric("rho", "theta")

        self._field_aligned_covariant_metric[0, 0] = (
            g_rho_rho + (self.dalpha_dr**2) * g_zeta_zeta
        )  # eq D.82, already normalised

        self._field_aligned_covariant_metric[0, 1] = (
            -self.dalpha_dr * g_zeta_zeta
        )  # eq D.83, inherets correct sigma_alpha from previous calculation g_fralpha / a
        self._field_aligned_covariant_metric[
            1, 0
        ] = self._field_aligned_covariant_metric[0, 1]

        self._field_aligned_covariant_metric[0, 2] = (
            g_rho_theta + self.dalpha_dr * self.dalpha_dtheta * g_zeta_zeta
        )  # eq D.84, g_frft / a
        self._field_aligned_covariant_metric[
            2, 0
        ] = self._field_aligned_covariant_metric[0, 2]

        self._field_aligned_covariant_metric[
            1, 1
        ] = g_zeta_zeta  # eq D.85, g_alphaalpha / a^2

        self._field_aligned_covariant_metric[1, 2] = (
            -self.dalpha_dtheta * g_zeta_zeta
        )  # eq D.86, inherets correct sigma_alpha from previous calculation, g_alphaft / a^2
        self._field_aligned_covariant_metric[
            2, 1
        ] = self._field_aligned_covariant_metric[1, 2]

        self._field_aligned_covariant_metric[2, 2] = (
            g_theta_theta + (self.dalpha_dtheta**2) * g_zeta_zeta
        )  # eq D.87, g_ftft / a^2

    def set_field_aligned_contravariant_metric_components(
        self,
    ):  # use covariant components and equation C.50 to set contravariant components g^{ij}, defined on page 196.
        # Some are simpler to obtain by dotting LHS's of equations D.79-D.81.

        gcont_rho_rho = self.toroidal_contravariant_metric("rho", "rho")
        gcont_rho_theta = self.toroidal_contravariant_metric("rho", "rho")
        gcont_theta_theta = self.toroidal_contravariant_metric("theta", "theta")

        gf_rho_rho = self.field_aligned_covariant_metric("rho", "rho")
        gf_rho_theta = self.field_aligned_covariant_metric("rho", "theta")
        gf_theta_theta = self.field_aligned_covariant_metric("theta", "theta")

        self._field_aligned_contravariant_metric[
            0, 0
        ] = self._toroidal_contravariant_metric[
            0, 0
        ]  # g^frfr, already normalised

        self._field_aligned_contravariant_metric[
            0, 2
        ] = self._toroidal_contravariant_metric[
            0, 1
        ]  # g^frft * a
        self._field_aligned_contravariant_metric[
            2, 0
        ] = self._field_aligned_contravariant_metric[0, 2]

        self._field_aligned_contravariant_metric[
            2, 2
        ] = self._toroidal_contravariant_metric[
            1, 1
        ]  # g^ftft * a^2

        self._field_aligned_contravariant_metric[0, 2] = (
            self.dalpha_dr * gcont_rho_rho + self.dalpha_dtheta * gcont_rho_theta
        )  # g^fralpha * a
        self._field_aligned_contravariant_metric[
            2, 0
        ] = self._field_aligned_contravariant_metric[1, 2]

        self._field_aligned_contravariant_metric[2, 1] = (
            self.dalpha_dr * gcont_rho_theta + self.dalpha_dtheta * gcont_theta_theta
        )  # g^ftalpha * a^2
        self._field_aligned_contravariant_metric[
            1, 2
        ] = self._field_aligned_contravariant_metric[2, 1]

        self._field_aligned_contravariant_metric[1, 1] = (
            gf_rho_rho * gf_theta_theta - (gf_rho_theta**2)
        ) / (
            self.Jacobian**2
        )  # g^alphaalpha * a^2
