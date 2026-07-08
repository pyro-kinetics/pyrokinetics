import warnings

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs
from scipy.optimize import newton
from scipy.special import logsumexp

from ..pyro import Pyro
from ..units import ureg as units


class Diagnostics:
    """
    Contains all the diagnistics that can be applied to simulation output data.

    Currently, this class contains only the function to generate a Poincare map,
    but new diagnostics will be available in future.

    Please call "load_gk_output" before attempting to use any diagnostic

    Parameters
    ----------
    pyro: Pyro object containing simulation output data (and geometry)

    """

    def __init__(self, pyro: Pyro):
        self.pyro = pyro

    def gs2_geometry_terms(self, ntheta_multiplier: int = 1):
        nperiod = self.pyro.numerics.nperiod
        ntheta = ((2 * nperiod) - 1) * self.pyro.numerics.ntheta * ntheta_multiplier

        theta_max = ((2 * nperiod) - 1) * np.pi
        theta_even = np.linspace(-theta_max, theta_max, ntheta + 1)

        self.pyro.load_metric_terms(theta=theta_even)

        metric = self.pyro.metric_terms
        metric.to(self.pyro.norms.gs2, self.pyro.norms.context)

        theta = metric.regulartheta * units.radians

        dpdrho = metric.mu0dPdr
        dpsidrho = metric.dpsidr

        # Metric tensor terms
        g_ra_contr = metric.field_aligned_contravariant_metric("r", "alpha")
        g_aa = metric.field_aligned_contravariant_metric("alpha", "alpha")

        g_at = metric.field_aligned_covariant_metric("alpha", "theta")

        grad_alpha = np.sqrt(g_aa)

        g_tt = metric.field_aligned_covariant_metric("theta", "theta")
        g_rt = metric.field_aligned_covariant_metric("r", "theta")
        g_rr = metric.field_aligned_contravariant_metric("r", "r")

        g_rt_contr = metric.toroidal_contravariant_metric("r", "theta")
        g_rr_contr = metric.toroidal_contravariant_metric("r", "r")

        # Jacobian and derivatives
        jacob = metric.Jacobian

        # R and derivatives
        R = metric.R
        dR_dr = metric.dRdr

        # Z and derivatives
        Z = metric.Z
        dZ_dr = metric.dZdr

        # B_mag and derivatives
        B_mag = metric.B_magnitude
        dB_dr = metric.dB_magnitude_dr
        dB_dtheta = metric.dB_magnitude_dtheta

        # Drifts
        gbdrift = -2 * (dB_dr - g_rt / g_tt * dB_dtheta) / B_mag
        gbdrift0 = 2 * metric.dqdr * g_at * dB_dtheta / (g_tt * B_mag)

        press = -2 * dpdrho * jacob / (np.sqrt(g_tt) * B_mag * dpsidrho)
        cvdrift = gbdrift + press
        cvdrift0 = gbdrift0

        # GDS values
        gds2 = (grad_alpha * dpsidrho) ** 2
        gds21 = -(dpsidrho**2) * g_ra_contr * metric.dqdr
        gds22 = (dpsidrho * metric.dqdr) ** 2 * g_rr

        # Parallel gradient
        gradpar = 1 / np.sqrt(g_tt)

        # B field
        bmag = B_mag
        bpol = dpsidrho * np.sqrt(g_rr) / R

        # Grad rho
        grho = np.sqrt(g_rr)

        # Eikonal
        S = -metric.alpha
        dS_dr = (
            -dpsidrho
            / np.sqrt(g_rr_contr)
            * (metric.dalpha_dr * g_rr_contr + metric.dalpha_dtheta * g_rt_contr)
        )

        return {
            "dpdrho": dpdrho,
            "theta": theta,
            "bmag": bmag,
            "bpol": bpol,
            "gradpar": gradpar,
            "cvdrift": cvdrift,
            "cvdrift0": cvdrift0,
            "gds2": gds2,
            "gbdrift": gbdrift,
            "gbdrift0": gbdrift0,
            "grho": grho,
            "gds21": gds21,
            "gds22": gds22,
            "jacob": jacob,
            "rplot": R,
            "zplot": Z,
            "rprime": dR_dr,
            "zprime": dZ_dr,
            "aplot": S,
            "aprime": dS_dr,
        }

    def calculate_sonic_quasineutrality(self, ntheta_multiplier: int = 4):
        """
        Calculates potential phi0 generated on a flux surface given
        a specific rotation set by local_geometry.electron.omega0 that
        satifies quasineutrality

        Also calculates dphi0/dr, dphi0/dtheta and the force balance term
        F_rho that can go into metric_terms

        Returns

           sonic_term: dict
               Dictionary of phi0, dphi_dr, dphi0_dtheta, F_rho

        """
        local_geometry = self.pyro.local_geometry
        local_species = self.pyro.local_species

        # Poloidal variation calculation on high resolution grid
        ntheta = self.pyro.numerics.ntheta * ntheta_multiplier + 1

        theta = np.linspace(-np.pi, np.pi, ntheta)

        theta_zero = np.argmin(abs(theta))

        R, Z = local_geometry.get_flux_surface(theta)

        dR_dtheta, dR_dr, dZ_dtheta, dZ_dr = local_geometry.get_RZ_derivatives(theta)

        electron = local_species.electron
        omega0 = electron.omega0
        domega_drho = electron.domega_drho

        R_theta_zero = R[theta_zero]
        dR_dr_theta_zero = dR_dr[theta_zero]

        # Is wrong but works because of minisation set up
        dR_dtheta_theta_zero = R[theta_zero]

        phi0_max = (electron.mass * omega0**2 * R[theta_zero] ** 2) / (electron.z * 2)

        dphi0max_dr = (
            electron.mass
            * omega0
            * R_theta_zero
            / electron.z
            * (omega0 * dR_dr_theta_zero + R_theta_zero * domega_drho)
        )

        dphi0max_dtheta = (
            electron.mass * omega0**2 * R_theta_zero / electron.z * dR_dtheta_theta_zero
        )

        inverse_lNs = {}
        for name in local_species.names:
            species = local_species[name]
            z = species.z
            mass = species.mass
            temp = species.temp
            dens = species.dens.m
            inverse_ln = species.inverse_ln
            inverse_lt = species.inverse_lt

            potential0 = z * phi0_max / temp - mass * omega0**2 * R_theta_zero**2 / (
                2 * temp
            )
            dpotential0_dr = (
                z / temp * dphi0max_dr
                + potential0 * inverse_lt
                - mass
                * omega0
                * R_theta_zero
                / temp
                * (omega0 * dR_dr_theta_zero + R_theta_zero * domega_drho)
            )
            inverse_lNs[name] = inverse_ln - dpotential0_dr

        # Determine phi0
        def quasineutrality_calc(phi0_shape):

            phi0 = phi0_shape * phi0_max
            log_ion_terms = []

            for name in local_species.names:
                species = local_species[name]
                z = species.z
                mass = species.mass
                temp = species.temp
                dens = species.dens.m
                potential0 = z * phi0_max / temp - mass * omega0**2 * R[
                    theta_zero
                ] ** 2 / (2 * temp)
                flux_dens = dens * np.exp(potential0)
                potential = z * phi0 / temp - mass * omega0**2 * R**2 / (2 * temp)

                log_n = np.log(flux_dens) - potential

                if species.name == "electron":
                    log_electron = log_n
                else:
                    log_ion_terms.append(np.log(z.m) + log_n.m)
            log_ions = logsumexp(log_ion_terms, axis=0)
            residual = (log_ions - log_electron).m
            residual[theta_zero] = phi0_shape[theta_zero] - 1.0

            return residual

        phi0_guess = np.zeros(ntheta)
        phi0_shape = newton(quasineutrality_calc, phi0_guess)

        test_qn = np.max(abs(quasineutrality_calc(phi0_shape)))
        if test_qn > 1e-10:
            print(f"Difference in quasineutrality in newton solve {test_qn}")

        phi0 = phi0_shape * phi0_max

        # R derivative of phi0
        def quasineutrality_deriv_r_calc(dphi0_dr_shape):

            residual = np.zeros(dphi0_dr_shape.shape)
            dphi0_dr = dphi0_dr_shape * dphi0max_dr
            for name in local_species.names:
                species = local_species[name]
                z = species.z
                mass = species.mass
                temp = species.temp
                dens = species.dens.m
                inverse_lt = species.inverse_lt

                potential0 = (
                    z * phi0_max / temp
                    - mass * omega0**2 * R_theta_zero**2 / (2 * temp)
                )
                flux_dens = dens * np.exp(potential0)

                # Actual 1D theta data
                potential = z * phi0 / temp - mass * omega0**2 * R**2 / (2 * temp)
                dpotential_dr = (
                    z / temp * dphi0_dr
                    + potential * inverse_lt
                    - mass * omega0 * R * (omega0 * dR_dr + R * domega_drho) / temp
                )
                dens_theta = flux_dens * np.exp(-potential)

                residual += (z * dens_theta * (-inverse_lNs[name] - dpotential_dr)).m

            residual[theta_zero] = dphi0_dr_shape[theta_zero] - 1.0
            return residual

        dphi0_dr_guess = np.zeros(len(theta))
        dphi0_dr_shape = newton(quasineutrality_deriv_r_calc, dphi0_dr_guess)

        test_qn = np.max(abs(quasineutrality_deriv_r_calc(dphi0_dr_shape)))

        if test_qn > 1e-10:
            print(
                f"Difference in quasineutrality radial derivative in newton solve {test_qn}"
            )

        dphi0_dr = dphi0_dr_shape * dphi0max_dr

        # theta derivative of phi0 - overkill but consistent
        # DOESNT work unless dR_dtheta is set to non-zero value
        def quasineutrality_deriv_t_calc(dphi0_dtheta_shape):

            residual = np.zeros(dphi0_dtheta_shape.shape)
            dphi0_dtheta = dphi0_dtheta_shape * dphi0max_dtheta
            for name in local_species.names:
                species = local_species[name]
                z = species.z
                mass = species.mass
                temp = species.temp
                dens = species.dens.m

                potential0 = (
                    z * phi0_max / temp
                    - mass * omega0**2 * R_theta_zero**2 / (2 * temp)
                )
                flux_dens = dens * np.exp(potential0)

                # Actual 1D theta data
                potential = z * phi0 / temp - mass * omega0**2 * R**2 / (2 * temp)
                dpotential_dtheta = (
                    z / temp * dphi0_dtheta - mass * omega0**2 * R * dR_dtheta / temp
                )
                dens_theta = flux_dens * np.exp(-potential)

                residual += (z * dens_theta * (-dpotential_dtheta)).m

            residual[theta_zero] = dphi0_dr_shape[theta_zero] - 1.0
            return residual

        dphi0_dtheta_guess = np.zeros(len(theta))
        dphi0_dtheta_shape = newton(quasineutrality_deriv_t_calc, dphi0_dtheta_guess)

        test_qn = np.max(abs(quasineutrality_deriv_t_calc(dphi0_dtheta_shape)))
        if test_qn > 1e-10:
            print(
                f"Difference in quasineutrality theta derivative in newton solve {test_qn}"
            )

        dphi0_dtheta = dphi0_dtheta_shape * dphi0max_dtheta

        F_rho = 0.0
        for isp, name in enumerate(local_species.names):
            species = local_species[name]
            z = species.z
            mass = species.mass
            temp = species.temp
            dens = species.dens
            inverse_lt = species.inverse_lt
            inverse_ln = species.inverse_ln
            potential0 = z * phi0_max / temp - mass * omega0**2 * R_theta_zero**2 / (
                2 * temp
            )
            dpotential0_dr = (
                z / temp * dphi0max_dr
                + potential0 * inverse_lt
                - mass
                * omega0
                * R_theta_zero
                / temp
                * (omega0 * dR_dr_theta_zero + R_theta_zero * domega_drho)
            )
            flux_dens = dens * np.exp(potential0)

            # Actual 1D theta data
            potential = z * phi0 / temp - mass * omega0**2 * R**2 / (2 * temp)
            dpotential_dr = (
                z / temp * dphi0_dr
                + potential * inverse_lt
                - mass * omega0 * R * (omega0 * dR_dr + R * domega_drho) / temp
            )
            dens_theta = flux_dens * np.exp(-potential)

            dn_dr = dens_theta * (-inverse_lNs[name] - dpotential_dr)
            dT_dr = -inverse_lt * temp

            F_rho += (
                dn_dr * temp
                + dens_theta * dT_dr
                - mass * dens_theta * omega0**2 * R * dR_dr
            )

        return {
            "theta_qn": theta,
            "phi0": phi0,
            "dphi0_dr": dphi0_dr,
            "dphi0_dtheta": dphi0_dtheta,
            "F_rho": F_rho,
        }

    def generate_eik_geometry_terms(self, ntheta_multiplier: int = 4):
        local_geometry = self.pyro.local_geometry
        local_species = self.pyro.local_species
        numerics = self.pyro.numerics
        convention = self.pyro.norms.pyrokinetics

        sonic_terms = self.calculate_sonic_quasineutrality(
            ntheta_multiplier=ntheta_multiplier
        )

        beta_prime_scale = -local_geometry.beta_prime.to(convention).m / (
            local_species.inverse_lp.to(convention).m
            * local_species.pressure.to(convention).m
            * numerics.beta.to(convention).m
        )

        theta_qn = sonic_terms["theta_qn"]
        F_rho = sonic_terms["F_rho"]
        phi0 = sonic_terms["phi0"]
        dphi0_dr = sonic_terms["dphi0_dr"]
        dphi0_dtheta = sonic_terms["dphi0_dtheta"]

        mu0F_rho = (
            F_rho * self.pyro.numerics.beta / 2 * beta_prime_scale
        ).m * local_geometry.beta_prime.units

        nperiod = self.pyro.numerics.nperiod
        ntheta = ((2 * nperiod) - 1) * self.pyro.numerics.ntheta * ntheta_multiplier

        theta_max = ((2 * nperiod) - 1) * np.pi
        theta_even = np.linspace(-theta_max, theta_max, ntheta + 1)

        # Interp onto broad grid
        phi0 = np.interp(theta_even, theta_qn, phi0, period=2 * np.pi)
        dphi0_dr = np.interp(theta_even, theta_qn, dphi0_dr, period=2 * np.pi)
        dphi0_dtheta = np.interp(theta_even, theta_qn, dphi0_dtheta, period=2 * np.pi)
        mu0F_rho = np.interp(theta_even, theta_qn, mu0F_rho, period=2 * np.pi)

        self.pyro.load_metric_terms(theta=theta_even, sonic=True, mu0F_rho=mu0F_rho)

        metric = self.pyro.metric_terms
        metric.to(self.pyro.norms.gs2, self.pyro.norms.context)

        theta = metric.regulartheta * units.radians

        Rmaj = local_geometry.Rmaj
        mach = local_species.electron.omega0 * Rmaj
        mu0F_rho = metric.mu0dPdr
        dpsidrho = metric.dpsidr

        # Metric tensor terms
        g_ra_contr = metric.field_aligned_contravariant_metric("r", "alpha")
        g_aa = metric.field_aligned_contravariant_metric("alpha", "alpha")

        g_at = metric.field_aligned_covariant_metric("alpha", "theta")

        grad_alpha = np.sqrt(g_aa)

        g_tt = metric.field_aligned_covariant_metric("theta", "theta")
        g_rt = metric.field_aligned_covariant_metric("r", "theta")
        g_rr = metric.field_aligned_contravariant_metric("r", "r")

        g_rt_contr = metric.toroidal_contravariant_metric("r", "theta")
        g_rr_contr = metric.toroidal_contravariant_metric("r", "r")

        # Jacobian and derivatives
        jacob = metric.Jacobian

        # R and derivatives
        R = metric.R
        dR_dr = metric.dRdr
        dR_dtheta = metric.dRdtheta

        # Z and derivatives
        Z = metric.Z
        dZ_dr = metric.dZdr

        # B_mag and derivatives
        B_mag = metric.B_magnitude
        dB_dr = metric.dB_magnitude_dr
        dB_dtheta = metric.dB_magnitude_dtheta

        # alpha derivatives
        dalpha_dr = metric.dalpha_dr
        dalpha_dtheta = metric.dalpha_dtheta

        # q derivative
        dqdr = metric.dqdr

        # Drifts

        # Grad-B
        gbdrift = -2 * (dB_dr - g_rt / g_tt * dB_dtheta) / B_mag
        gbdrift0 = 2 * dqdr * g_at * dB_dtheta / (g_tt * B_mag)

        press = -2 * mu0F_rho / B_mag**2

        # Curvature
        cvdrift = gbdrift + press
        cvdrift0 = gbdrift0

        # Coriolis
        crdrift = (
            mach
            / Rmaj
            * (
                4
                * dpsidrho
                * R
                / jacob
                * (dR_dtheta * dalpha_dr - dR_dr * dalpha_dtheta)
                / B_mag
            )
        )
        crdrift0 = mach / Rmaj * 4 * dpsidrho * dqdr * R / jacob * dR_dtheta

        # Centrifugal
        cfdrift = (mach / Rmaj) ** 2 * 2 * R * (dR_dr - g_rt / g_tt * dR_dtheta)
        cfdrift0 = (mach / Rmaj) ** 2 * 2 * R * dR_dtheta * g_at / g_tt * dqdr

        # Potential
        phdrift = -2 * (dphi0_dr - g_rt / g_tt * dphi0_dtheta)
        phdrift0 = 2 * dphi0_dtheta * g_at / g_tt * dqdr

        # GDS values
        gds2 = (grad_alpha * dpsidrho) ** 2
        gds21 = -(dpsidrho**2) * g_ra_contr * metric.dqdr
        gds22 = (dpsidrho * metric.dqdr) ** 2 * g_rr

        # Parallel gradient
        gradpar = 1 / np.sqrt(g_tt)

        # B field
        bmag = B_mag
        bpol = dpsidrho * np.sqrt(g_rr) / R

        # Grad rho
        grho = np.sqrt(g_rr)

        # Eikonal
        S = -metric.alpha
        dS_dr = (
            -dpsidrho
            / np.sqrt(g_rr_contr)
            * (metric.dalpha_dr * g_rr_contr + metric.dalpha_dtheta * g_rt_contr)
        )

        return {
            "theta": theta,
            "dpsidrho": dpsidrho,
            "mu0F_rho": mu0F_rho,
            "bmag": bmag,
            "bpol": bpol,
            "gradpar": gradpar,
            "cvdrift": cvdrift,
            "cvdrift0": cvdrift0,
            "gds2": gds2,
            "gbdrift": gbdrift,
            "gbdrift0": gbdrift0,
            "crdrift": crdrift,
            "crdrift0": crdrift0,
            "cfdrift": cfdrift,
            "cfdrift0": cfdrift0,
            "phdrift": phdrift,
            "phdrift0": phdrift0,
            "phi0": phi0,
            "dphi0_dr": dphi0_dr,
            "dphi0_dtheta": dphi0_dtheta,
            "grho": grho,
            "gds21": gds21,
            "gds22": gds22,
            "jacob": jacob,
            "rplot": R,
            "zplot": Z,
            "rprime": dR_dr,
            "zprime": dZ_dr,
            "aplot": S,
            "aprime": dS_dr,
        }

    def ideal_ballooning_solver(self, theta0: float = 0.0):
        r"""
        Adapted from ideal-ballooning-solver
        https://github.com/rahulgaur104/ideal-ballooning-solver/blob/master/ball_scan.py

        Parameters
        ----------
        theta0: float
            Ballooning angle

        Returns
        -------
        gamma: Float
            Ideal ballooning growth rate
        """

        geometry_terms = self.gs2_geometry_terms()

        dpdrho = geometry_terms["dpdrho"]
        theta = geometry_terms["theta"]
        bmag = geometry_terms["bmag"]
        gradpar = geometry_terms["gradpar"]
        cvdrift = geometry_terms["cvdrift"]
        cvdrift0 = geometry_terms["cvdrift0"]
        gds2 = geometry_terms["gds2"]
        gds21 = geometry_terms["gds21"]
        gds22 = geometry_terms["gds22"]

        # theta0 correction
        gds2 = gds2 + 2 * theta0 * gds21 + theta0**2 * gds22
        cvdrift = cvdrift + theta0 * cvdrift0

        gamma, X_arr, dX_arr, g_arr, c_arr, f_arr = gamma_ball_full(
            dpdrho, theta, bmag, gradpar, cvdrift, gds2
        )

        return gamma

    def bicoherence(self, fluctuation, wavenumber_tolerance=1e-5, stationary=True):
        r"""
        Perform bicoherence analysis for a given DataArray

        Bispectrum given by:

        .. math::
            Bi(k_{x,1}, k_{y,1}, k_{x,2}, k_{y,2}) = \langle f_1 * f_2 * f_3^\dagger\rangle_{t}

        The bicoherence is normalised to

        .. math::
            b^2 = \frac{|Bi|^2}{\langle|f_1 * f_2|^2\rangle_{t} \langle|f_3|^2\rangle_{t}}

        where

        .. math::
            f_1 = fluctuation(k_{x,1}, k_{y,1}, t)

        .. math::
            f_2 = fluctuation(k_{x,2}, k_{y,2}, t)

        .. math::
            f_3 = fluctuation(k_{x,3}, k_{y,3}, t)


        with :math:`\langle\rangle_{t}` being an average over time

        where
        :math:`k_{x,3} = k_{x,1}+k_{x,2}`
        :math:`k_{y,3} = k_{y,1}+k_{y,2}`

        Parameters
        ----------
        fluctuation: xr.DataArray
            Fluctuation dataset which must be a DataArray with
            dimensions (:math:`k_x`, :math:`k_y`, :math:`t`)

        wavenumber_tolerance: float
            Tolerance to find matching wavenumbers in :math:`k_{x,3}, k_{y,3}`

        Returns
        -------
        data: xr.Dataset
            Dataset with DataArray of bicoherence and phase with dimensions
            (:math:`k_{x,1}`, :math:`k_{y,1}`,:math:`k_{x,2}`, :math:`k_{y,2}`)

        """

        time = fluctuation["time"].data
        kx = fluctuation["kx"].data
        ky = fluctuation["ky"].data

        ntime = len(time)
        nkx = len(kx)
        nky = len(ky)

        kx_max = max(abs(kx))
        ky_max = max(abs(ky))

        # Initialise different Arrays
        bispectrum = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        bicoherence = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        phase = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        # Create mesh grids of the kx and ky values
        kx1, kx2 = np.meshgrid(kx, kx, indexing="ij")
        ky1, ky2 = np.meshgrid(ky, ky, indexing="ij")

        # Compute kx3 and ky3 with modular arithmetic
        kx3 = kx1 + kx2
        ky3 = ky1 + ky2

        kx3 = np.where(abs(kx3) <= max(abs(kx)), kx3, kx3 % kx_max * np.sign(kx3))
        ky3 = np.where(abs(ky3) <= max(abs(ky)), ky3, ky3 % ky_max * np.sign(ky3))

        # Extract the relevant data slices based on calculated indices
        # fft_data for (kx1, ky1), (kx2, ky2), (kx3, ky3) Currently in the form
        fluctuation_kx1_ky1 = fluctuation.sel(
            kx=kx1.ravel(), ky=ky1.ravel()
        ).values.reshape(nkx, nkx, nky, nky, ntime)
        fluctuation_kx2_ky2 = fluctuation.sel(
            kx=kx2.ravel(), ky=ky2.ravel()
        ).values.reshape(nkx, nkx, nky, nky, ntime)
        fluctuation_kx3_ky3 = fluctuation.sel(
            kx=kx3.ravel(),
            ky=ky3.ravel(),
            method="nearest",
            tolerance=wavenumber_tolerance,
        ).values.reshape(nkx, nkx, nky, nky, ntime)

        # Swap axes such that dimensions are (kx1, ky1, kx2, ky2)
        fluctuation_kx1_ky1 = np.swapaxes(fluctuation_kx1_ky1, 1, 2)
        fluctuation_kx2_ky2 = np.swapaxes(fluctuation_kx2_ky2, 1, 2)
        fluctuation_kx3_ky3 = np.swapaxes(fluctuation_kx3_ky3, 1, 2)

        if not stationary:
            fluctuation_kx1_ky1 *= 1.0 / np.abs(fluctuation_kx1_ky1)
            fluctuation_kx2_ky2 *= 1.0 / np.abs(fluctuation_kx2_ky2)
            fluctuation_kx3_ky3 *= 1.0 / np.abs(fluctuation_kx3_ky3)

        bispectrum_data = (
            fluctuation_kx1_ky1 * fluctuation_kx2_ky2 * np.conj(fluctuation_kx3_ky3)
        )
        phase.data = np.mean(
            np.arctan2(bispectrum_data.imag, bispectrum_data.real), axis=-1
        )

        power_total = np.mean(
            np.abs(fluctuation_kx1_ky1 * fluctuation_kx2_ky2) ** 2, axis=-1
        ) * np.mean(np.abs(fluctuation_kx3_ky3) ** 2, axis=-1)

        bispectrum.data = np.abs(np.mean(bispectrum_data, axis=-1))

        bicoherence.data = np.abs(bispectrum) ** 2 / (power_total)

        bicoherence = bicoherence.fillna(0.0)

        data = bicoherence.to_dataset(name="bicoherence")
        data["phase"] = phase

        return data

    def cross_bicoherence(
        self,
        fluctuation1,
        fluctuation2,
        fluctuation3,
        wavenumber_tolerance=1e-5,
        stationary=True,
    ):
        r"""
        Perform cross bicoherence analysis for a given DataArray

        Bispectrum given by:

        .. math::
            Bi(k_{x,1}, k_{y,1}, k_{x,2}, k_{y,2}) = \langle f_1 * f_2 * f_3^\dagger\rangle_{t}

        The bicoherence is normalised to

        .. math::
            b^2 = \frac{|Bi|^2}{\langle|f_1 * f_2|^2\rangle_{t} \langle|f_3|^2\rangle_{t}}

        where

        .. math::
            f_1 = fluctuation1(k_{x,1}, k_{y,1}, t)

        .. math::
            f_2 = fluctuation2(k_{x,2}, k_{y,2}, t)

        .. math::
            f_3 = fluctuation3(k_{x,3}, k_{y,3}, t)


        with :math:`\langle\rangle_{t}` being an average over time

        where
        :math:`k_{x,3} = k_{x,1}+k_{x,2}`
        :math:`k_{y,3} = k_{y,1}+k_{y,2}`

        Parameters
        ----------
        fluctuation1: xr.DataArray
            First fluctuation dataset which must be a DataArray with
            dimensions (:math:`k_x`, :math:`k_y`, :math:`t`)

        fluctuation2: xr.DataArray
            Second fluctuation dataset which must be a DataArray with
            dimensions (:math:`k_x`, :math:`k_y`, :math:`t`)

        fluctuation3: xr.DataArray
            Third fluctuation dataset which must be a DataArray with
            dimensions (:math:`k_x`, :math:`k_y`, :math:`t`)

        wavenumber_tolerance: float
            Tolerance to find matching wavenumbers in :math:`k_{x,3}, k_{y,3}`

        Returns
        -------
        data: xr.Dataset
            Dataset with DataArray of cross-bicoherence and phase with dimensions
            (:math:`k_{x,1}`, :math:`k_{y,1}`,:math:`k_{x,2}`, :math:`k_{y,2}`)

        """

        time = fluctuation1["time"].data
        kx = fluctuation1["kx"].data
        ky = fluctuation1["ky"].data

        ntime = len(time)
        nkx = len(kx)
        nky = len(ky)

        kx_max = max(abs(kx))
        ky_max = max(abs(ky))

        # Initialise different Arrays
        bispectrum = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        bicoherence = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        phase = xr.DataArray(
            np.zeros((nkx, nky, nkx, nky)),
            coords=[kx, ky, kx, ky],
            dims=["kx1", "ky1", "kx2", "ky2"],
        )

        # Create mesh grids of the kx and ky values
        kx1, kx2 = np.meshgrid(kx, kx, indexing="ij")
        ky1, ky2 = np.meshgrid(ky, ky, indexing="ij")

        # Compute kx3 and ky3 with modular arithmetic
        kx3 = kx1 + kx2
        ky3 = ky1 + ky2

        kx3 = np.where(abs(kx3) <= max(kx), kx3, kx3 % kx_max * np.sign(kx3))
        ky3 = np.where(abs(ky3) <= max(ky), ky3, ky3 % ky_max * np.sign(ky3))

        # Extract the relevant data slices based on calculated indices
        # fft_data for (kx1, ky1), (kx2, ky2), (kx3, ky3) Currently in the form
        fluctuation_kx1_ky1 = fluctuation1.sel(
            kx=kx1.ravel(), ky=ky1.ravel()
        ).values.reshape(nkx, nkx, nky, nky, ntime)
        fluctuation_kx2_ky2 = fluctuation2.sel(
            kx=kx2.ravel(), ky=ky2.ravel()
        ).values.reshape(nkx, nkx, nky, nky, ntime)
        fluctuation_kx3_ky3 = fluctuation3.sel(
            kx=kx3.ravel(),
            ky=ky3.ravel(),
            method="nearest",
            tolerance=wavenumber_tolerance,
        ).values.reshape(nkx, nkx, nky, nky, ntime)

        # Swap axes such that dimensions are (kx1, ky1, kx2, ky2)
        fluctuation_kx1_ky1 = np.swapaxes(fluctuation_kx1_ky1, 1, 2)
        fluctuation_kx2_ky2 = np.swapaxes(fluctuation_kx2_ky2, 1, 2)
        fluctuation_kx3_ky3 = np.swapaxes(fluctuation_kx3_ky3, 1, 2)

        if not stationary:
            fluctuation_kx1_ky1 *= 1.0 / np.abs(fluctuation_kx1_ky1)
            fluctuation_kx2_ky2 *= 1.0 / np.abs(fluctuation_kx2_ky2)
            fluctuation_kx3_ky3 *= 1.0 / np.abs(fluctuation_kx3_ky3)

        bispectrum_data = (
            fluctuation_kx1_ky1 * fluctuation_kx2_ky2 * np.conj(fluctuation_kx3_ky3)
        )
        phase.data = np.mean(
            np.arctan2(bispectrum_data.imag, bispectrum_data.real), axis=-1
        )

        power_total = np.mean(
            np.abs(fluctuation_kx1_ky1 * fluctuation_kx2_ky2) ** 2, axis=-1
        ) * np.mean(np.abs(fluctuation_kx3_ky3) ** 2, axis=-1)

        bispectrum.data = np.abs(np.mean(bispectrum_data, axis=-1))

        bicoherence.data = np.abs(bispectrum) ** 2 / (power_total)

        bicoherence = bicoherence.fillna(0.0)

        data = bicoherence.to_dataset(name="bicoherence")
        data["phase"] = phase

        return data


def gamma_ball_full(
    dPdrho, theta_PEST, B, gradpar, cvdrift, gds2, vguess=None, sigma0=0.42
):
    r"""
    Ideal-ballooning growth rate finder.
    This function uses a finite-difference method
    to calculate the maximum growth rate against the
    infinite-n ideal ballooning mode. The equation being solved is
    :math::`\frac{\partial}{\partial \theta} \bigg( g \frac{\partial X}{\partial \theta} \bigg) + c X - \lambda f X = 0, g, f > 0`

    where
    :math::`g = \nabla_{||} gds2 / |B|`
    :math::`c = \frac{1}{\nabla_{||}}  \frac{\partial p}{\partial \rho} cvdrift`
    :math::`f = \frac{gds2}{|B|^3} \frac{1}{\nabla_{||}}`

    are needed along a field line to solve the ballooning equation once.

    :math::`gds2 = \frac{\partial\psi}{\partial \rho}^2  |\nabla \alpha|^2`, :math::`\alpha = \phi - q (\theta - \theta_0)`
    which is the field line bending term

    :math::`\nabla_{||} = \hat{b} \cdot \vec{\nabla} \theta` which is the parallel gradient,
    :math::`cvdrift = \frac{\partial\psi}{\partial\rho}^2 (\hat{b} \times (\hat{b} \cdot \vec{\nabla} \hat{b})) \cdot \vec{\nabla} \alpha`
    which is the curvature drift

    Parameters
    ----------
    dPdrho: float
        Gradient of the total plasma pressure w.r.t the radial coordinate rho
    theta_PEST: ArrayLike
        Vector of grid points in a straight field line theta (PEST coordinates)
    B: ArrayLike
        The magnetic field strength
    gradpar: ArrayLike
        Parallel gradient
    cvdrift: ArrayLike
        Geometric factor in curvature drift
    gds2: ArrayLike
        Field line bending term
    vguess: ArrayLike
        starting guess for the eigenfunction (length in N_theta_PEST -2)
    sigma0: float
        starting guess for the eigenvalue (square of the growth rate). Must always be greater than the final value. Choose a large value

    Returns
    -------
    gam: Float
        Ideal ballooning growth rate
    """

    theta_ball = theta_PEST
    # ntheta = len(theta_ball)

    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    g = np.abs(gradpar) * gds2 / (B)
    c = -1 * dPdrho * cvdrift * 1 / (np.abs(gradpar) * B)
    f = gds2 / B**2 * 1 / (np.abs(gradpar) * B)

    len1 = len(g)

    # Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)

    # TODO pint=0.23 doesnt support np.diag
    g_u = np.interp(theta_ball_u, theta_ball, g).m
    c_u = np.interp(theta_ball_u, theta_ball, c).m
    f_u = np.interp(theta_ball_u, theta_ball, f).m

    # uniform theta_ball on half points with half the size, i.e., only from [0, (2*nperiod-1)*np.pi]
    theta_ball_u_half = (theta_ball_u[:-1] + theta_ball_u[1:]) / 2
    h = np.diff(theta_ball_u_half)[2].m
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g).m
    g_u1 = g_u[:]
    c_u1 = c_u[:]
    f_u1 = f_u[:]

    len2 = int(len1) - 2
    A = np.zeros((len2, len2))

    A = (
        np.diag(g_u_half[1:-1] / f_u1[2:-1] * 1 / h**2, -1)
        + np.diag(
            -(g_u_half[1:] + g_u_half[:-1]) / f_u1[1:-1] * 1 / h**2
            + c_u1[1:-1] / f_u1[1:-1],
            0,
        )
        + np.diag(g_u_half[1:-1] / f_u1[1:-2] * 1 / h**2, 1)
    )

    # Method without M is approx 3 X faster with Arnoldi iteration
    # Perhaps, we should try dstemr as suggested by Max Ruth. However, I doubt if
    # that will give us a significant speedup
    w, v = eigs(A, 1, sigma=sigma0, v0=vguess, tol=5.0e-7, OPpart="r")
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')

    # ## Richardson extrapolation
    X = np.zeros((len2 + 2,))
    dX = np.zeros((len2 + 2,))
    # X[1:-1]     = np.reshape(v[:, idx_max].real, (-1,))/np.max(np.abs(v[:, idx_max].real))
    X[1:-1] = np.reshape(v[:, 0].real, (-1,)) / np.max(np.abs(v[:, 0].real))

    X[0] = 0.0
    X[-1] = 0.0

    dX[0] = (-1.5 * X[0] + 2 * X[1] - 0.5 * X[2]) / h
    dX[1] = (X[2] - X[0]) / (2 * h)

    dX[-2] = (X[-1] - X[-3]) / (2 * h)
    dX[-1] = (0.5 * X[-3] - 2 * X[-2] + 1.5 * 0.0) / (h)

    dX[2:-2] = 2 / (3 * h) * (X[3:-1] - X[1:-3]) - (X[4:] - X[0:-4]) / (12 * h)

    Y0 = -g_u1 * dX**2 + c_u1 * X**2
    Y1 = f_u1 * X**2
    # plt.plot(range(len3+2), X, range(len3+2), dX); plt.show()
    gam = simpson(Y0) / simpson(Y1)

    # return np.sign(gam)*np.sqrt(abs(gam)), X, dX, g_u1, c_u1, f_u1
    return gam, X, dX, g_u1, c_u1, f_u1
