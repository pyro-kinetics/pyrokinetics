import numpy as np
from pint.errors import DimensionalityError
from scipy.integrate import cumulative_trapezoid, simpson

from ..decorators import not_implemented
from ..pyro import Pyro
from ..units import PyroContextError, ureg


class BootstrapModel:

    def __init__(
        self,
        pyro: Pyro,
        ntheta=None,
        radial_coordinate="r_minor",
        ion_collisionality="neo",
    ):

        pyro.load_metric_terms(ntheta)

        if ion_collisionality not in ("neo", "legacy"):
            raise ValueError(
                "ion_collisionality must be 'neo' or 'legacy', got "
                f"{ion_collisionality!r}"
            )

        self.pyro = pyro
        self.radial_coordinate = radial_coordinate
        self.ion_collisionality = ion_collisionality
        self.Zeff = self.pyro.local_species.zeff.m

        # Catch floating point errors
        if np.isclose(self.Zeff, 1.0):
            self.Zeff = 1.0

        self.ip_ccw = self.pyro.local_geometry.ip_ccw
        self.bt_ccw = self.pyro.local_geometry.bt_ccw
        self.Ipsi = self.pyro.local_geometry.Fpsi

        # Get trapped fraction
        self.get_trapped_fraction()

        # Get collisionalities
        self.get_collisionalities()

        # Get kinetics species and gradients
        self.get_kinetic_species_data()

        # Get coefficients
        self.get_bs_coeffs()

        # Get bootstrap current
        self.get_bs_current()

        # Get total current
        self.get_total_current()

        # Resolve the radial coordinate Jphi_eff is defined on
        self._get_radial_label()

        # Get toroidal current
        self.get_toroidal_current()

    def _get_grad_shafranov_terms(self):
        r"""
        Common Grad-Shafranov terms shared by the current calculations

        Returns
        -------
        dpsidr : Float
            :math:`\partial \psi / \partial r`, signed by ``ip_ccw``
        F : Float
            Current function :math:`F = R B_\zeta`, signed by ``bt_ccw``
        Fprime : Float
            :math:`\partial F / \partial \psi`
        mu0_dpdpsi : Float
            :math:`\mu_0 \partial p / \partial \psi`
        mu0 : Float
            :math:`\mu_0` in the normalised units of the current context
        """

        metric = self.pyro.metric_terms
        dpsidr = metric.dpsidr * -self.ip_ccw
        mu0_dpdr = metric.mu0dPdr
        mu0_dpdpsi = mu0_dpdr / dpsidr

        F = metric.B_zeta * self.bt_ccw
        Fprime = metric.dB_zeta_dr / dpsidr * self.bt_ccw

        try:
            beta = self.pyro.numerics.beta.m
        except AttributeError:
            beta = self.pyro.norms.beta.m

        B0 = 1 * self.B2_fsa.units**0.5
        mu0 = B0**2 * beta / (2 * self.pe)

        return dpsidr, F, Fprime, mu0_dpdpsi, mu0

    def _get_radial_label(self):
        r"""
        Radial coordinate that the effective toroidal current density is built on

        ``Jphi_eff`` is :math:`(1 / 2 \pi x) \partial I_p / \partial x`, which depends
        on the radial label :math:`x`. Two labels are supported, selected by the
        ``radial_coordinate`` argument:

        - ``"r_minor"`` (default): :math:`x = r`, the minor radius
          (``metric_terms.rho``). This is the historical behaviour.
        - ``"rho_tor"``: :math:`x = \rho = \sqrt{\Psi_{\rm tor} / (\pi B_{\rm geo})}`,
          the toroidal flux label used by JETTO (Tholerus eq 1) and by transport
          codes generally.

        Since both are :math:`(1 / 2 \pi x) \partial I_p / \partial x`, the chain rule
        relates them by a single dimensionless factor

        .. math::
            J_\phi^{\rm eff}\big|_\rho = J_\phi^{\rm eff}\big|_r
                \frac{\partial (r^2)}{\partial (\rho^2)}, \qquad
            \frac{\partial (r^2)}{\partial (\rho^2)}
                = \frac{2 \pi r B_{\rm geo}}{q} \frac{\partial r}{\partial \psi}

        :math:`\Psi_{\rm tor}` cancels, so the factor needs only :math:`r`,
        :math:`\partial r / \partial \psi`, :math:`q` and :math:`B_{\rm geo}`. All are
        available at a single flux surface from the ``Equilibrium`` splines, so no
        radial scan is required.

        Notes
        -----
        :math:`J_\phi^{\rm eff}` scales as :math:`1/c^2` under
        :math:`\rho \rightarrow c\rho`, so this is a change of *convention* and not a
        relabelling, and it is pinned by :math:`B_{\rm geo}`. That value is taken from
        ``Equilibrium.B_0``. It cannot be recovered from a local GK input, which is why
        ``"rho_tor"`` requires a global equilibrium: pyrokinetics' own
        :math:`B_{\rm ref} = F / R_{\rm maj}` is defined per flux surface and is not
        :math:`B_{\rm geo}`.

        Sets ``rho_label``, the radial coordinate itself, and ``label_factor``, which
        is exactly 1 for ``"r_minor"``.
        """

        metric = self.pyro.metric_terms

        if self.radial_coordinate == "r_minor":
            self.rho_label = metric.rho
            self.label_factor = 1.0 * ureg.dimensionless
            return

        if self.radial_coordinate != "rho_tor":
            raise ValueError(
                "radial_coordinate must be 'r_minor' or 'rho_tor', got "
                f"{self.radial_coordinate!r}"
            )

        eq = self.pyro.eq
        if eq is None:
            raise ValueError(
                "radial_coordinate='rho_tor' requires a global Equilibrium, but "
                "pyro.eq is None. The conversion is set by B_geo, the vacuum field at "
                "the geometric axis, which a local GK input does not carry. Load an "
                "equilibrium, or use radial_coordinate='r_minor'. Note that "
                "pyrokinetics' own Bref = F / Rmaj is a per-flux-surface quantity and "
                "is not B_geo, so it cannot be substituted for it."
            )

        psi_n = self.pyro.local_geometry.psi_n
        B_geo = np.abs(eq.B_0)

        # rho = sqrt(Psi_tor / (pi B_geo)) = a_tor rho_tor, using pyrokinetics'
        # rho_tor, which is normalised to 1 at the LCFS
        a_tor = np.sqrt(eq.psi_tor(1.0) / (np.pi * B_geo))
        self.rho_label = (a_tor * eq.rho_tor(psi_n)).to(self.pyro.norms.lref)

        # d(r^2)/d(rho^2). Positive by construction, so the signs of B_0, q and
        # dr/dpsi (which depend on the COCOS convention) are discarded.
        self.label_factor = np.abs(
            2
            * np.pi
            * eq.r_minor(psi_n)
            * B_geo
            * eq.r_minor_prime(psi_n)
            / eq.q(psi_n)
        ).to(ureg.dimensionless)

    def get_total_current(self):
        r"""
        Total parallel current moment :math:`\langle J \cdot B \rangle` from the
        Grad-Shafranov equation, and the external (non-bootstrap) remainder

        .. math::
            \langle J \cdot B \rangle =
                \frac{F' \langle B^2 \rangle}{\mu_0} + F \frac{\partial p}{\partial \psi}
        """

        _, F, Fprime, mu0_dpdpsi, mu0 = self._get_grad_shafranov_terms()

        self.JdotB = (Fprime * self.B2_fsa + F * mu0_dpdpsi) / mu0
        self.JextdotB = self.JdotB - self.JbsdotB

    def get_Fprime_from_total_current(self, JdotB=None):
        r"""
        Invert :func:`get_total_current` to obtain :math:`F'` from a given
        :math:`\langle J \cdot B \rangle`
        """

        _, F, _, mu0_dpdpsi, mu0 = self._get_grad_shafranov_terms()

        return (JdotB * mu0 - F * mu0_dpdpsi) / self.B2_fsa

    def get_toroidal_current(self):
        r"""
        Toroidal current density, decomposed into bootstrap, external and combined
        Pfirsch-Schlüter + diamagnetic contributions.

        The local toroidal current density follows from the Grad-Shafranov equation:

        .. math::
            J_\phi(\theta) = R \frac{\partial p}{\partial \psi}
                           + \frac{F F'}{\mu_0 R}

        Two flux-surface reductions of :math:`J_\phi` are provided, since they are
        **not** the same quantity (they differ by roughly the elongation):

        - ``Jphi_fsa``: the plain flux-surface average :math:`\langle J_\phi \rangle`.
          This is the quantity transport codes such as SCENE, JETTO and TRANSP
          report as their toroidal current density.
        - ``Jphi_eff``: the effective toroidal current density
          :math:`J_\phi^{\rm eff} = \frac{1}{2 \pi \rho} \frac{\partial I_p}{\partial \rho}
          = \frac{V'}{4 \pi^2 \rho} \langle J_\phi / R \rangle`, i.e. the current density
          of an equivalent circular cross-section of radius :math:`\rho`. Which
          :math:`\rho` this is depends on the ``radial_coordinate`` argument, see
          :func:`_get_radial_label`; the ``Jphi_fsa`` family does not depend on it.

        Each is split into three parts. Parallel driven currents (bootstrap and
        external) contribute

        .. math::
            J_\phi^{d} = \frac{F \langle J^d_\parallel \cdot B \rangle}{\langle B^2 \rangle}
                         \langle 1/R \rangle

        for ``Jphi_fsa``, and the analogous form with
        :math:`\frac{V'}{4 \pi^2 \rho} \langle R^{-2} \rangle` for ``Jphi_eff``. The
        remaining Pfirsch-Schlüter and diamagnetic currents are grouped together.

        Notes
        -----
        The external contribution is **auxiliary plus ohmic combined**. A local
        calculation cannot separate them, since ``JextdotB`` is obtained by
        subtracting the bootstrap current from the total.

        ``get_bs_current`` takes the absolute value of ``JbsdotB``, so ``JextdotB``
        and the external toroidal contributions derived from it are only correct
        when :math:`\langle J \cdot B \rangle` is positive in the sign convention
        of the equilibrium.

        Sets ``Jphi``, ``Jphi_fsa``, ``Jphi_bs_fsa``, ``Jphi_ext_fsa``,
        ``Jphi_psdia_fsa``, ``Jphi_eff``, ``Jphi_bs_eff``, ``Jphi_ext_eff``,
        ``Jphi_psdia_eff`` and ``Ip``.
        """

        metric = self.pyro.metric_terms

        dpsidr, F, Fprime, mu0_dpdpsi, mu0 = self._get_grad_shafranov_terms()

        # dp/dpsi
        dpdpsi = mu0_dpdpsi / mu0

        # Flux surface averages of the geometry
        R_fsa = metric.flux_surface_average(metric.R)
        R_inv_fsa = metric.flux_surface_average(1.0 / metric.R)
        R_inv2_fsa = metric.flux_surface_average(1.0 / metric.R**2)

        Vprime = metric.dVdr
        rho = metric.rho

        # Local toroidal current density from the Grad-Shafranov equation
        self.Jphi = metric.R * dpdpsi + F * Fprime / (mu0 * metric.R)

        # --- Flux-surface averaged toroidal current density, <J_phi> ---
        self.Jphi_fsa = R_fsa * dpdpsi + F * Fprime * R_inv_fsa / mu0

        # Parallel driven currents, <J_d . B> F <1/R> / <B^2>
        parallel_factor = F * R_inv_fsa / self.B2_fsa
        self.Jphi_bs_fsa = parallel_factor * self.JbsdotB
        self.Jphi_ext_fsa = parallel_factor * self.JextdotB

        # Combined Pfirsch-Schlüter and diamagnetic contribution
        self.Jphi_psdia_fsa = dpdpsi * (R_fsa - F**2 * R_inv_fsa / self.B2_fsa)

        # --- Effective toroidal current density, (1 / 2 pi rho) dIp/drho ---
        # label_factor = d(r^2)/d(rho_label^2) moves V'/(4 pi^2 r) onto the
        # requested radial label, and is 1 for "r_minor". See _get_radial_label.
        eff_factor = self.label_factor * Vprime / (4 * np.pi**2 * rho)
        self.Jphi_eff = eff_factor * (dpdpsi + F * Fprime * R_inv2_fsa / mu0)

        parallel_factor_eff = eff_factor * F * R_inv2_fsa / self.B2_fsa
        self.Jphi_bs_eff = parallel_factor_eff * self.JbsdotB
        self.Jphi_ext_eff = parallel_factor_eff * self.JextdotB

        self.Jphi_psdia_eff = (
            eff_factor * dpdpsi * (1.0 - F**2 * R_inv2_fsa / self.B2_fsa)
        )

        # --- Radial derivative of the enclosed current, dIp/drho = 2 pi rho J_eff ---
        # Integrating these over rho gives the total current carried by each
        # component, see ``integrate_toroidal_current``
        self.dIp_drho = 2 * np.pi * self.rho_label * self.Jphi_eff
        self.dIp_bs_drho = 2 * np.pi * self.rho_label * self.Jphi_bs_eff
        self.dIp_ext_drho = 2 * np.pi * self.rho_label * self.Jphi_ext_eff
        self.dIp_psdia_drho = 2 * np.pi * self.rho_label * self.Jphi_psdia_eff

        # --- Total toroidal current enclosed by this flux surface ---
        # mu0 Ip = V' psi' <|grad rho|^2 / R^2> / (4 pi^2), using V' = 2 pi int(J dtheta)
        # and psi' = 2 pi dpsidr. The leading minus sign puts Ip in the same sign
        # convention as Jphi above, so that Ip = int(Jphi_eff 2 pi rho drho).
        grad_r2 = metric.toroidal_contravariant_metric("r", "r")
        self.Ip = (
            -dpsidr
            * Vprime
            * metric.flux_surface_average(grad_r2 / metric.R**2)
            / (2 * np.pi * mu0)
        )

    def get_trapped_fraction(self):

        metric = self.pyro.metric_terms
        Jacobian = metric.Jacobian
        theta = metric.regulartheta
        ntheta = len(theta)
        B_mod = abs(metric.B_magnitude)
        B_max = np.max(B_mod)

        B2_fsa_units = metric.flux_surface_average(B_mod**2)
        B2_fsa = B2_fsa_units.m

        lambd_grid = np.linspace(0, 1 / B_max, 100)
        lambd = np.tile(lambd_grid, (ntheta, 1))

        lambd_fsa = simpson(
            np.sqrt(1.0 - lambd.m * B_mod.m[:, np.newaxis]) * Jacobian.m[:, np.newaxis],
            x=theta,
            axis=0,
        ) / simpson(Jacobian.m, x=theta)

        lambda_integral = simpson(lambd_grid.m / lambd_fsa, x=lambd_grid.m)
        ftrap = 1.0 - (3.0 / 4.0 * B2_fsa * lambda_integral)

        self.B2_fsa = B2_fsa_units
        self.trapped_fraction = ftrap

    @staticmethod
    def _is_fast(name, species):
        """
        Whether a species is fast/non-thermal, and so takes no part in thermal
        ion-ion collisions. Same heuristic as ``get_kinetic_species_data``.
        """
        return "fast" in name or species.temp.m > 10

    def _get_collisionality_ion(self):
        r"""
        Ion species and density rescaling used by :math:`\nu^*_i`

        Sauter eq (18c) is written for a plasma with a single ion species, so both
        :math:`n_i` and :math:`Z` there refer to that one species. Neither Sauter
        (1999) nor Redl (2021) states how to generalise it to several ion species --
        Sauter's conclusion explicitly leaves "the correct form of the value of Z"
        undetermined. NEO, whose results the Redl coefficients were fitted to,
        resolves it in ``neo_theory.f90`` by taking the majority ion by density and
        rescaling its collisionality by the summed ion density,

        .. code-block:: fortran

            nui_star_S = nui_star_HH / dens(is_ion,ir) * dens_sum

        so the charge and the Coulomb logarithm stay those of the majority ion and
        only the density becomes a sum. OMFIT's ``sauter_bootstrap`` does the same.
        That is the ``"neo"`` behaviour here, and it is the default.

        Fast species are left out of both the majority pick and the sum. NEO does not
        exclude them explicitly, but a NEO species list holds thermal species by
        construction, whereas a ``LocalSpecies`` built from a transport code carries
        fast ions too.

        ``"legacy"`` restores the previous behaviour -- the first non-electron species
        in ``LocalSpecies`` order, using its own density alone -- for reproducing
        results from before this was changed.

        Returns
        -------
        ion : Species
            Species supplying :math:`Z`, :math:`T_i` and the Coulomb logarithm
        dens_ratio : float
            Factor rescaling :math:`\nu^*_i` from that species' density to the
            summed thermal ion density. 1.0 in ``"legacy"`` mode.
        """
        ls = self.pyro.local_species
        ion_names = [name for name in ls.names if name != "electron"]

        if self.ion_collisionality == "legacy":
            return ls[ion_names[0]], 1.0

        thermal = [name for name in ion_names if not self._is_fast(name, ls[name])]
        if not thermal:
            thermal = ion_names

        # Densities all share units within a LocalSpecies, so compare magnitudes
        ion = ls[max(thermal, key=lambda name: ls[name].dens.m)]
        dens_sum = sum(ls[name].dens.m for name in thermal)

        return ion, dens_sum / ion.dens.m

    def get_collisionalities(self):

        lg = self.pyro.local_geometry
        ls = self.pyro.local_species

        eps = lg.rho / lg.Rmaj
        electron = ls.electron
        ion, ion_dens_ratio = self._get_collisionality_ion()

        lref = lg.Rmaj.units
        vref = ls.electron.nu.units * lref
        mref = ls.electron.mass.units
        tref = ls.electron.temp.units
        try:
            coolog_e = 31.3 - np.log(
                np.sqrt(electron.dens.to("meter**-3").m) / electron.temp.to("eV").m
            )

            self.nu_star_e = (
                6.921e-18
                * abs(lg.q)
                * lg.Rmaj.to("meter")
                * self.Zeff
                * electron.dens.to("meter**-3")
                * coolog_e
                / (electron.temp.to("eV") ** 2 * eps**1.5)
            ).m
        except (DimensionalityError, PyroContextError):
            # Account for different defn of coulomb logarithm (pretty accurate - within 0.1%)
            coulomb_factor = 1.027
            self.nu_star_e = (
                electron.nu
                * abs(lg.q)
                * lg.Rmaj
                / eps**1.5
                * 3.0
                / 4.0
                * self.Zeff
                / vref
                * np.sqrt(electron.mass / mref)
                * coulomb_factor
            )

        try:
            coolog_i = 30.0 - np.log(
                ion.z.m**3
                * np.sqrt(ion.dens.to("meter**-3").m)
                / ion.temp.to("eV").m ** 1.5
            )

            self.nu_star_i = (
                4.900e-18
                * abs(lg.q)
                * lg.Rmaj.to("meter")
                * ion.z**4
                * ion.dens.to("meter**-3")
                * coolog_i
                / (ion.temp.to("eV") ** 2 * eps**1.5)
            ).m
        except (DimensionalityError, PyroContextError):
            # Account for different defn of coulomb logarithm (not as accurate ~ 5%)
            coulomb_factor = 1.17
            self.nu_star_i = (
                3.0
                / 4.0
                / np.sqrt(2)
                * ion.nu
                * abs(lg.q)
                * lg.Rmaj
                / eps**1.5
                * ion.z.m**4
                / vref
                * np.sqrt(ion.mass / mref)
                * np.sqrt(tref / ion.temp)
                * coulomb_factor
            )

        # Rescale from the majority ion density to the summed thermal ion density,
        # keeping that ion's charge and Coulomb logarithm. See
        # ``_get_collisionality_ion``. No-op when ion_collisionality="legacy".
        self.nu_star_i = self.nu_star_i * ion_dens_ratio

    def get_bs_coeffs(self):

        self.L31 = self.get_L31()
        self.L32 = self.get_L32()
        self.L34 = self.get_L34()
        self.alpha = self.get_alpha()

    @not_implemented
    def get_kinetic_species_data(self):
        pass

    @not_implemented
    def get_L31(self):
        pass

    @not_implemented
    def get_f_eff_t31(self):
        pass

    @not_implemented
    def get_L32(self):
        pass

    @not_implemented
    def get_F32_ee(self):
        pass

    @not_implemented
    def get_f_eff_t32_ee(self):
        pass

    @not_implemented
    def get_F32_ei(self):
        pass

    @not_implemented
    def get_f_eff_t32_ei(self):
        pass

    @not_implemented
    def get_sigma_neo_over_sigma_spitzer(self):
        pass

    @not_implemented
    def get_f_eff_t33(self):
        pass

    @not_implemented
    def get_L34(self):
        pass

    @not_implemented
    def get_alpha0(self):
        pass

    @not_implemented
    def get_alpha(self):
        pass

    @not_implemented
    def get_bs_current(self):
        pass


class Redl2021(BootstrapModel):
    r"""
    Bootstrap model based off of A. Redl Phys. Plasmas 28, 022502 (2021)
    https://doi.org/10.1063/5.0012664
    """

    def get_kinetic_species_data(self):

        lg = self.pyro.local_geometry
        ls = self.pyro.local_species
        electron = ls.electron
        ion_names = [name for name in ls.names if ls[name].z.m > 0]
        main_ion = ls[ion_names[0]]

        # self.ptot = ls.pressure
        self.pe = electron.dens * electron.temp
        self.dlnTe_dpsi = electron.inverse_lt / lg.dpsidr
        self.dlnne_dpsi = electron.inverse_ln / lg.dpsidr

        # Arrays of pion and dlnTi/dpsi
        self.pion = np.zeros(len(ion_names)) * self.pe.units
        self.ptot = 0.0 * self.pe.units
        self.dlnTi_dpsi = np.zeros(len(ion_names)) * self.dlnTe_dpsi.units

        self.ptot += self.pe
        for i_s, ion_name in enumerate(ion_names):
            species = ls[ion_name]
            if "fast" in ion_name or species.temp.m > 10:
                self.pion[i_s] = species.dens * main_ion.temp
                self.dlnTi_dpsi[i_s] = main_ion.inverse_lt / lg.dpsidr
            else:
                self.pion[i_s] = species.dens * species.temp
                self.dlnTi_dpsi[i_s] = species.inverse_lt / lg.dpsidr

            self.ptot += self.pion[i_s]

    # Equation (10)
    def get_L31(self):
        X31 = self.get_f_eff_t31()
        return (
            (1.0 + 0.15 / (self.Zeff**1.2 - 0.71)) * X31
            - 0.22 / (self.Zeff**1.2 - 0.71) * X31**2
            + 0.01 / (self.Zeff**1.2 - 0.71) * X31**3
            + 0.06 / (self.Zeff**1.2 - 0.71) * X31**4
        )

    # Equation (11)
    def get_f_eff_t31(self):
        return self.trapped_fraction / (
            1.0
            + 0.67
            * (1.0 - 0.7 * self.trapped_fraction)
            * np.sqrt(self.nu_star_e)
            / (0.56 + 0.44 * self.Zeff)
            + (0.52 + 0.086 * np.sqrt(self.nu_star_e))
            * (1.0 + 0.87 * self.trapped_fraction)
            * self.nu_star_e
            / (1.0 + 1.13 * np.sqrt(self.Zeff - 1))
        )

    # Equation (12)
    def get_L32(self):
        return self.get_F32_ee() + self.get_F32_ei()

    # Equation (13)
    def get_F32_ee(self):
        X32e = self.get_f_eff_t32_ee()
        return (
            (0.1 + 0.6 * self.Zeff)
            / (self.Zeff * (0.77 + 0.63 * (1.0 + (self.Zeff - 1) ** 1.1)))
            * (X32e - X32e**4)
            + 0.7
            / (1.0 + 0.2 * self.Zeff)
            * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
            + 1.3 / (1.0 + 0.5 * self.Zeff) * X32e**4
        )

    # Equation (14)
    def get_f_eff_t32_ee(self):
        return self.trapped_fraction / (
            1.0
            + 0.23
            * (1.0 - 0.96 * self.trapped_fraction)
            * np.sqrt(self.nu_star_e)
            / (self.Zeff**0.5)
            + 0.13
            * (1.0 - 0.38 * self.trapped_fraction)
            * self.nu_star_e
            / (self.Zeff**2)
            * (
                np.sqrt(1.0 + 2 * np.sqrt(self.Zeff - 1))
                + self.trapped_fraction**2
                * np.sqrt((0.075 + 0.25 * (self.Zeff - 1) ** 2) * self.nu_star_e)
            )
        )

    # Equation (15)
    def get_F32_ei(self):
        X32i = self.get_f_eff_t32_ei()
        return (
            -(0.4 + 1.93 * self.Zeff)
            / (self.Zeff * (0.8 + 0.6 * self.Zeff))
            * (X32i - X32i**4)
            + 5.5
            / (1.5 + 2 * self.Zeff)
            * (X32i**2 - X32i**4 - 0.8 * (X32i**3 - X32i**4))
            - 1.3 / (1.0 + 0.5 * self.Zeff) * X32i**4
        )

    # Equation (16)
    def get_f_eff_t32_ei(self):
        return self.trapped_fraction / (
            1.0
            + 0.87
            * (1.0 + 0.39 * self.trapped_fraction)
            * np.sqrt(self.nu_star_e)
            / (1.0 + 2.95 * (self.Zeff - 1) ** 2)
            + 1.53
            * (1.0 - 0.37 * self.trapped_fraction)
            * self.nu_star_e
            * (2 + 0.375 * (self.Zeff - 1))
        )

    # Equation (17)
    def get_sigma_neo_over_sigma_spitzer(self):
        X33 = self.get_f_eff_t33()
        return (
            1.0
            - (1.0 + 0.21 / self.Zeff) * X33
            + 0.54 / self.Zeff * X33**2
            - 0.33 / self.Zeff * X33**3
        )

    # Equation (18)
    def get_f_eff_t33(self):
        return self.trapped_fraction / (
            1.0
            + 0.25
            * (1.0 - 0.7 * self.trapped_fraction)
            * np.sqrt(self.nu_star_e)
            * (1.0 + 0.45 * np.sqrt(self.Zeff - 1))
            + 0.61
            * (1.0 - 0.41 * self.trapped_fraction)
            * self.nu_star_e
            / np.sqrt(self.Zeff)
        )

    # Equation (19)
    def get_L34(self):
        return self.get_L31()

    # Equation (20)
    def get_alpha0(self):
        return (
            -(0.62 + 0.055 * (self.Zeff - 1))
            / (0.53 + 0.17 * (self.Zeff - 1))
            * (1.0 - self.trapped_fraction)
            / (
                1.0
                - (0.31 - 0.065 * (self.Zeff - 1)) * self.trapped_fraction
                - 0.25 * self.trapped_fraction**2
            )
        )

    # Equation (21)
    def get_alpha(self):
        a0 = self.get_alpha0()
        num = (
            a0
            + 0.7
            * self.Zeff
            * np.sqrt(self.trapped_fraction)
            * np.sqrt(self.nu_star_i)
            / (1.0 + 0.18 * np.sqrt(self.nu_star_i))
            - 0.002 * (self.nu_star_i**2) * self.trapped_fraction**6
        )
        den = 1.0 + 0.004 * (self.nu_star_i**2) * self.trapped_fraction**6
        return num / den

    # Equation (2)
    def get_bs_current(self):

        self.JbsdotB = np.abs(
            -self.Ipsi
            * (
                self.ptot * self.L31 * self.dlnne_dpsi
                + self.pe * (self.L31 + self.L32) * self.dlnTe_dpsi
                + np.sum(
                    self.pion * (self.L31 + self.alpha * self.L34) * self.dlnTi_dpsi,
                    axis=0,
                )
            )
        )

        self.Jbs = self.JbsdotB / np.sqrt(self.B2_fsa)


class Sauter1999(BootstrapModel):
    r"""
    Bootstrap model based off of O. Sauter et al Physics of Plasma 6, 2834 (1999).
    and O. Sauter et al Physics of Plasma 9, 5140 (2002)
    """

    def get_kinetic_species_data(self):

        lg = self.pyro.local_geometry
        ls = self.pyro.local_species
        electron = ls.electron
        ion_names = [name for name in ls.names if ls[name].z.m > 0.0]
        main_ion = ls[ion_names[0]]

        self.ptot = ls.pressure
        self.pe = electron.dens * electron.temp
        self.dlnTe_dpsi = electron.inverse_lt / lg.dpsidr
        self.dlnne_dpsi = electron.inverse_ln / lg.dpsidr

        # Arrays of pion and dlnTi/dpsi
        self.pion = np.zeros(len(ion_names)) * self.pe.units
        self.ptot = 0.0 * self.pe.units
        self.dlnTi_dpsi = np.zeros(len(ion_names)) * self.dlnTe_dpsi.units
        self.dlnp_dpsi = self.pe * ls.inverse_lp / lg.dpsidr * 0.0

        self.ptot += self.pe
        self.dlnp_dpsi += (
            self.pe * (electron.inverse_lt + electron.inverse_ln) / lg.dpsidr
        )
        for i_s, ion_name in enumerate(ion_names):
            species = ls[ion_name]
            if "fast" in ion_name or species.temp.m > 10:
                self.pion[i_s] = species.dens * main_ion.temp
                self.dlnTi_dpsi[i_s] = main_ion.inverse_lt / lg.dpsidr
                self.dlnp_dpsi += (
                    self.pion[i_s]
                    * (main_ion.inverse_lt + species.inverse_ln)
                    / lg.dpsidr
                )
            else:
                self.pion[i_s] = species.dens * species.temp
                self.dlnTi_dpsi[i_s] = species.inverse_lt / lg.dpsidr
                self.dlnp_dpsi += (
                    self.pion[i_s]
                    * (species.inverse_lt + species.inverse_ln)
                    / lg.dpsidr
                )
            self.ptot += self.pion[i_s]

        self.dlnp_dpsi *= 1.0 / self.ptot

        self.Rpe = self.pe / self.ptot

    # Equation (13a)
    def get_sigma_neo_over_sigma_spitzer(self):
        X = self.get_fteff_33()
        return (
            1.0
            - (1.0 + 0.36 / self.Zeff) * X
            + (0.59 / self.Zeff) * X**2
            - (0.23 / self.Zeff) * X**3
        )

    # Equation (13b)
    def get_fteff_33(self):
        return self.trapped_fraction / (
            1.0
            + (0.55 - 0.1 * self.trapped_fraction) * np.sqrt(self.nu_star_e)
            + 0.45 * (1.0 - self.trapped_fraction) * self.nu_star_e / (self.Zeff**1.5)
        )

    # Equation (14a)
    def get_L31(self):
        X = self.get_fteff_31()
        return (
            (1.0 + 1.4 / (self.Zeff + 1)) * X
            - (1.9 / (self.Zeff + 1)) * X**2
            + (0.3 / (self.Zeff + 1)) * X**3
            + (0.2 / (self.Zeff + 1)) * X**4
        )

    # Equation (14b)
    def get_fteff_31(self):
        return self.trapped_fraction / (
            1.0
            + (1.0 - 0.1 * self.trapped_fraction) * np.sqrt(self.nu_star_e)
            + 0.5 * (1.0 - self.trapped_fraction) * self.nu_star_e / self.Zeff
        )

    # Equation (15a)
    def get_L32(self):
        return self.get_F32_ee(self.get_fteff_32_ee()) + self.get_F32_ei(
            self.get_fteff_32_ei()
        )

    # Equation (15b)
    def get_F32_ee(self, X):
        return (
            ((0.05 + 0.62 * self.Zeff) / (self.Zeff * (1.0 + 0.44 * self.Zeff)))
            * (X - X**4)
            + (1.0 / (1.0 + 0.22 * self.Zeff)) * (X**2 - X**4 - 1.2 * (X**3 - X**4))
            + (1.2 / (1.0 + 0.5 * self.Zeff)) * X**4
        )

    # Equation (15c)
    def get_F32_ei(self, Y):
        return (
            -(0.56 + 1.93 * self.Zeff)
            / (self.Zeff * (1.0 + 0.44 * self.Zeff))
            * (Y - Y**4)
            + (4.95 / (1.0 + 2.48 * self.Zeff)) * (Y**2 - Y**4 - 0.55 * (Y**3 - Y**4))
            - (1.2 / (1.0 + 0.5 * self.Zeff)) * Y**4
        )

    # Equation (15d)
    def get_fteff_32_ee(self):
        return self.trapped_fraction / (
            1.0
            + 0.26 * (1.0 - self.trapped_fraction) * np.sqrt(self.nu_star_e)
            + 0.18
            * (1.0 - 0.37 * self.trapped_fraction)
            * self.nu_star_e
            / np.sqrt(self.Zeff)
        )

    # Equation (15e)
    def get_fteff_32_ei(self):
        return self.trapped_fraction / (
            1.0
            + (1.0 + 0.6 * self.trapped_fraction) * np.sqrt(self.nu_star_e)
            + 0.85
            * (1.0 - 0.37 * self.trapped_fraction)
            * self.nu_star_e
            * (1.0 + self.Zeff)
        )

    # Equation (16a)
    def get_L34(self):
        X = self.get_fteff_34()
        return (
            (1.0 + 1.4 / (self.Zeff + 1)) * X
            - (1.9 / (self.Zeff + 1)) * X**2
            + (0.3 / (self.Zeff + 1)) * X**3
            + (0.2 / (self.Zeff + 1)) * X**4
        )

    # Equation (16b)
    def get_fteff_34(self):
        return self.trapped_fraction / (
            1.0
            + (1.0 - 0.1 * self.trapped_fraction) * np.sqrt(self.nu_star_e)
            + 0.5 * (1.0 - 0.5 * self.trapped_fraction) * self.nu_star_e / self.Zeff
        )

    # Equation (17a)
    def get_alpha(self):
        alpha0 = self.get_alpha0()
        num = (
            alpha0
            + 0.25
            * (1.0 - self.trapped_fraction**2)
            * np.sqrt(self.nu_star_i)
            / (1.0 + 0.5 * np.sqrt(self.nu_star_i))
            + 0.315 * (self.nu_star_i**2) * self.trapped_fraction**6
        )
        den = 1.0 + 0.15 * (self.nu_star_i**2) * self.trapped_fraction**6
        return num / den

    # Equation (17b) and errata Equation 1
    def get_alpha0(self):
        return -(1.17 * (1.0 - self.trapped_fraction)) / (
            1.0 - 0.22 * self.trapped_fraction - 0.19 * self.trapped_fraction**2
        )

    # Equation 2 and Errata Equation 2
    def get_bs_current(self):
        self.JbsdotB = np.abs(
            -self.Ipsi
            * self.pe
            * (
                self.L31 / self.Rpe * self.dlnp_dpsi
                + self.L32 * self.dlnTe_dpsi
                + np.sum(
                    self.L34 * self.alpha * self.pion / self.pe * self.dlnTi_dpsi,
                    axis=0,
                )
            )
        )

        # Maybe L31 term should be partitioned?
        # self.JbsdotB = np.abs(
        #     -self.Ipsi
        #     * (
        #         self.ptot * self.L31 * self.dlnne_dpsi
        #         + self.pe * (self.L31 + self.L32) * self.dlnTe_dpsi
        #         + np.sum(
        #             self.pion * (self.L31 + self.alpha * self.L34) * self.dlnTi_dpsi,
        #             axis=0,
        #         )
        #     )
        # )

        self.Jbs = self.JbsdotB / np.sqrt(self.B2_fsa)


def integrate_toroidal_current(
    pyro,
    psi_n,
    model=Redl2021,
    ntheta=None,
    radial_coordinate="r_minor",
    ion_collisionality="neo",
    **kwargs,
):
    r"""
    Total toroidal current carried by each component, integrated over the poloidal
    cross-section of a set of flux surfaces.

    A ``BootstrapModel`` describes a single flux surface, so the current enclosed by
    each component cannot be obtained locally -- it requires a radial scan. This
    function loops over ``psi_n``, builds ``model`` on each surface, and cumulatively
    integrates

    .. math::
        I^d_p(\rho) = \int_0^{\rho} 2 \pi \rho' J^{d,\rm eff}_\phi \, d\rho'

    for each of the bootstrap, external and Pfirsch-Schlueter + diamagnetic parts.

    Notes
    -----
    There is only one set of total currents, and it follows from the ``Jphi_*_eff``
    densities. ``Jphi_eff`` is *defined* as
    :math:`\frac{1}{2 \pi \rho} \partial I_p / \partial \rho`, so its
    :math:`2 \pi \rho` weighted integral is the enclosed current by construction.
    ``Jphi_fsa`` is a reporting convention (it is what SCENE, JETTO and TRANSP call
    their toroidal current density) but it is volume weighted, so integrating it over
    the poloidal area does **not** give the plasma current -- for a strongly shaped
    equilibrium that route is wrong by tens of percent.

    The innermost surface contributes :math:`\pi \rho_0^2 J^{\rm eff}_\phi(\rho_0)`,
    i.e. the current density is taken as constant between the magnetic axis and the
    first surface. Start the scan close to the axis to keep this small.

    Parameters
    ----------
    pyro : Pyro
        Pyro object with a global equilibrium and kinetics loaded
    psi_n : ArrayLike
        Normalised poloidal flux values to scan over, in increasing order
    model : type, default ``Redl2021``
        ``BootstrapModel`` subclass to use
    ntheta : int, optional
        Number of theta points passed to the model
    radial_coordinate : str, default ``"r_minor"``
        Radial label the effective current densities are built on, passed to
        ``model``. See ``BootstrapModel._get_radial_label``. The integration is
        performed over whichever label is chosen, so ``Ip`` is unchanged by it.
    **kwargs
        Additional keyword arguments passed to ``pyro.load_local``, for example
        ``local_geometry="MXH"``

    Returns
    -------
    dict
        Dictionary of arrays over the surfaces that loaded successfully, with keys
        ``psi_n``, ``rho`` [metre] (the ``radial_coordinate`` label), ``Ip_bs``,
        ``Ip_ext``, ``Ip_psdia``, ``Ip`` (the
        sum of the three components) and ``Ip_ampere`` (the enclosed current obtained
        locally from Ampere's law on each surface, as an independent check), all in
        amperes.

        Results are returned in physical units rather than normalised ones, because
        the normalisation references (``nref``, ``tref``, ``bref``) differ from one
        flux surface to the next and so cannot label a radial profile.
    """

    psi_n = np.asarray(psi_n)

    psi_n_ok = []
    rho = []
    dIp_drho = {"bs": [], "ext": [], "psdia": []}
    Ip_ampere = []

    for psi in psi_n:
        try:
            pyro.load_local(psi_n=psi, **kwargs)
        except Exception:
            continue

        bootstrap = model(
            pyro,
            ntheta=ntheta,
            radial_coordinate=radial_coordinate,
            ion_collisionality=ion_collisionality,
        )

        # Convert to physical units here, inside the loop. The reference values
        # behind the normalised units (nref, tref, bref) are those of the flux
        # surface currently loaded, and are overwritten by the next load_local
        # call -- so a normalised quantity kept across surfaces would later be
        # converted using the wrong references.
        psi_n_ok.append(psi)
        rho.append(bootstrap.rho_label.to("meter").m)
        dIp_drho["bs"].append(bootstrap.dIp_bs_drho.to("ampere / meter").m)
        dIp_drho["ext"].append(bootstrap.dIp_ext_drho.to("ampere / meter").m)
        dIp_drho["psdia"].append(bootstrap.dIp_psdia_drho.to("ampere / meter").m)
        Ip_ampere.append(bootstrap.Ip.to("ampere").m)

    if len(rho) < 2:
        raise RuntimeError(
            "Need at least two flux surfaces to integrate the toroidal current, "
            f"but only {len(rho)} of {len(psi_n)} loaded successfully"
        )

    rho = np.array(rho) * ureg.meter

    result = {
        "psi_n": np.array(psi_n_ok),
        "rho": rho,
        "Ip_ampere": np.array(Ip_ampere) * ureg.ampere,
    }

    total = None
    for name, values in dIp_drho.items():
        derivative = np.array(values) * ureg.ampere / ureg.meter

        # Contribution from the axis up to the first surface, assuming a constant
        # current density there: int_0^rho0 2 pi rho J drho = rho0 / 2 * dIp/drho
        axis = 0.5 * rho[0] * derivative[0]

        integral = (
            cumulative_trapezoid(derivative.m, rho.m, initial=0.0) * ureg.ampere + axis
        )

        result[f"Ip_{name}"] = integral
        total = integral if total is None else total + integral

    result["Ip"] = total

    return result
