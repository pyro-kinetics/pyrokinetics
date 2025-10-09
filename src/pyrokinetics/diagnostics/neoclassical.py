import numpy as np
from pint.errors import DimensionalityError
from scipy.integrate import simpson

from ..decorators import not_implemented
from ..pyro import Pyro


class BootstrapModel:

    def __init__(self, pyro: Pyro, ion_type="all", ntheta=None):

        pyro.load_metric_terms(ntheta)

        self.pyro = pyro
        self.Zeff = self.pyro.local_species.zeff.m

        # Catch floating point errors
        if np.isclose(self.Zeff, 1.0):
            self.Zeff = 1.0

        self.ip_ccw = self.pyro.local_geometry.ip_ccw
        self.Ipsi = self.pyro.local_geometry.Fpsi

        # Get trapped fraction
        self.get_trapped_fraction()

        # Get collisionalities
        self.get_collisionalities()

        # Get kinetics species and gradients
        self.ion_type = ion_type
        self.get_kinetic_species_data()

        # Get coefficients
        self.get_bs_coeffs()

        # Get bootstrap current
        self.get_bs_current()

    def get_trapped_fraction(self):

        metric = self.pyro.metric_terms
        Jacobian = metric.Jacobian
        theta = metric.regulartheta
        ntheta = len(theta)
        B_mod = abs(metric.B_magnitude)
        B_max = np.max(B_mod)

        B2_fsa = simpson(B_mod.m**2 * Jacobian.m, x=theta) / simpson(
            Jacobian.m, x=theta
        )
        B2_fsa_units = B2_fsa * B_mod.units**2

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

    def get_collisionalities(self):

        lg = self.pyro.local_geometry
        ls = self.pyro.local_species

        eps = lg.rho / lg.Rmaj
        electron = ls.electron
        ion_name = [name for name in ls.names if name != "electron"][0]
        ion = ls[ion_name]

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
        except DimensionalityError:
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
        except DimensionalityError:
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
            if self.ion_type == "thermal":
                if "fast" in ion_name:
                    continue
                if species.temp.m > 10:
                    continue
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

        self.JdotB = np.abs(
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

        self.Jbs = self.JdotB / np.sqrt(self.B2_fsa)


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
            if self.ion_type == "thermal":
                if "fast" in ion_name:
                    continue
                if species.temp.m > 10:
                    continue
            self.pion[i_s] = species.dens * species.temp
            self.dlnTi_dpsi[i_s] = species.inverse_lt / lg.dpsidr
            self.ptot += self.pion[i_s]
            self.dlnp_dpsi += (
                self.pion[i_s] * (species.inverse_lt + species.inverse_ln) / lg.dpsidr
            )

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
        self.JdotB = (
            np.abs(
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
            * self.ip_ccw
        )

        # Maybe L31 term should be partitioned?
        # self.JdotB = np.abs(
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

        self.Jbs = self.JdotB / np.sqrt(self.B2_fsa)
