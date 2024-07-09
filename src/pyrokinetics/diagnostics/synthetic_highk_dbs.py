import os

import matplotlib.pyplot as plt
import numpy as np
import xrft
from numpy.typing import ArrayLike

from pyrokinetics import Pyro
from pyrokinetics.typing import PathLike


class SyntheticHighkDBS:
    """
    Synthetic diagnostic for producing synthetic frequency/k-spectra from gyrokinetic simulations

    Steps:
    # 1. Inputs are diagnostic specific (diagnostic, filter, k, location, resolution, local rhos). See example_syn_hk_dbs.py
    # 2. Load equilibrium, kinetics files. Find scattering location theta. See __init__ in class SyntheticHighkDBS
    # 3. Map (kn, kb) to (kx, ky) for all k's / channels specified in 1. See function mapk
    # 4. Load GK output data (fluctuation moment file). See function get_syn_fspec
    # 5. For each input condition (eg. for each k in DBS/highk), filter sim data. See class Filter, functions apply filter, get_syn_fspec
    # 6. Generate synthetic spectra and make plots. See functions get_syn_fspec, plot_syn

    diag: str
        Type of diagnostic: 'highk', 'dbs', 'rcdr', 'bes'
    filter_type: str
        Type of filter - currently only 'gauss' is available   # 'bt_2d', 'bt_scotty' for beam tracing, 'gauss' for Gaussian filter
    Rloc: float, units [length]
        Major radius location of scattering
    Zloc: float, units [length]
        Z location of scattering
    Kn0_exp: ArrayLike, units [length**-1]
        Normal wavenumber component of the scattered turbulence k - see definition in Ruiz Ruiz PPCF 2022
    Kb0_exp: ArrayLike, units [length**-1]
        Binormal wavenumber component of the scattered turbulence k - see definition in Ruiz Ruiz PPCF 2022
    wR: float, units [length]
        Major radius spot size length of the filter function/spread function
    wZ: float, units [length]
        Vertical spot size length of the filter function/spread function
    eq_file: [PathLike]
        Equilibrium file used for the gyrokinetic simulations
    kinetics_file: [PathLike]
        Kinetics file used for the gyrokinetic simulations
    simdir: [PathLike]
        Directory where simulation data is stored
    gk_file: [PathLike]
        File name of gyrokintic input file
    savedir: [PathLike] default "./"
        Directory where to store the synthetic diagnostic output
    fsize: int, default 12
        Size of font for plots
    """

    def __init__(
        self,
        diag: str,
        filter_type: str,
        Rloc: float,
        Zloc: float,
        Kn0_exp: ArrayLike,
        Kb0_exp: ArrayLike,
        wR: float,
        wZ: float,
        eq_file: PathLike,
        kinetics_file: PathLike,
        simdir: PathLike,
        gk_file: PathLike,
        savedir: PathLike = "./",
        fsize: int = 12,
    ):

        # calculate thetaloc
        pyro = Pyro(
            eq_file=eq_file,
            kinetics_file=kinetics_file,
            gk_file=f"{simdir}/{gk_file}",
        )
        self.pyro = pyro
        self.eq = pyro.eq
        self.psin = self.eq._psi_RZ_spline(Rloc, Zloc) / (
            self.eq.psi_lcfs - self.eq.psi_axis
        )
        pyro.load_local(psi_n=self.psin, local_geometry="Miller")
        self.geometry = pyro.local_geometry
        pyro.load_metric_terms()

        self.metric = pyro.metric_terms
        self.diag = diag
        self.filter_type = filter_type
        self.Rloc = Rloc.to(pyro.norms.lref)
        self.Zloc = Zloc.to(pyro.norms.lref)
        self.Kn0_exp = Kn0_exp.to(pyro.norms.cgyro.rhoref**-1, pyro.norms.context)
        self.Kb0_exp = Kb0_exp.to(pyro.norms.cgyro.rhoref**-1, pyro.norms.context)
        self.wR = wR.to(pyro.norms.cgyro.rhoref)
        self.wZ = wZ.to(pyro.norms.cgyro.rhoref)
        self.eq_file = eq_file
        self.kinetics_file = kinetics_file
        self.simdir = simdir
        self.savedir = savedir
        self.fsize = fsize

        self.Pkf = np.zeros(np.size(Kn0_exp))
        self.Pkf_hann = np.zeros(np.size(Kn0_exp))
        self.Pkf_kx0ky0 = np.zeros(np.size(Kn0_exp))
        self.Pks = np.zeros(np.size(Kn0_exp))
        self.Sigma_ks_hann = np.zeros(np.size(Kn0_exp))

        # find thetaloc:
        thetatmp = self.geometry.theta[self.geometry.Z > self.geometry.Z0]
        Rtmp = self.geometry.R[self.geometry.Z > self.geometry.Z0]  # [m]
        Ztmp = self.geometry.Z[self.geometry.Z > self.geometry.Z0]  # [m]
        tmp_ind = np.argmin(np.abs(Rtmp - Rloc))
        self.thetaloc = thetatmp[tmp_ind]
        self.Rtmp = Rtmp[tmp_ind]
        self.Ztmp = Ztmp[tmp_ind]

        print("psin = " + str(self.psin))
        print("theta/pi = " + str(self.thetaloc / np.pi))
        print("Rloc = " + str(Rloc) + " [m]")
        print("Zloc = " + str(Zloc) + " [m]")

    def mapk(self):
        # Function that maps k from (Kn0_exp, Kb0_exp) to (kx0, ky0) in GK code : Will include (kR0, kZ0) in the future
        # Use metric coefficients (grr, etc) for the mapping. These are calculated in pyro.metric_terms.
        # Mapping depends on thetaloc, which we calculated in __init__
        # In pyrokinetics, (kx, ky) is the same as ... (??)  : in CGYRO: kx=2*pi*p/Lx, ky=n*q/r

        # get geo coefficients
        grr = self.metric.toroidal_contravariant_metric("r", "r")  # |grad r|^2
        grtheta = self.metric.toroidal_contravariant_metric(
            "r", "theta"
        )  # gradr . gradtheta
        gthetatheta = self.metric.toroidal_contravariant_metric(
            "theta", "theta"
        )  # |grad theta|^2
        gtheta_cross_gr2 = (self.metric.R / self.metric.Jacobian) ** 2
        gphi_cross_gr2 = grr / (self.metric.R**2)

        # gradients of alpha
        galpha_dot_gr = (
            self.metric.dalpha_dr * grr + self.metric.dalpha_dtheta * grtheta
        )
        galpha2 = (
            self.metric.dalpha_dr**2 * grr
            + 2 * self.metric.dalpha_dr * self.metric.dalpha_dtheta * grtheta
            + self.metric.dalpha_dtheta**2 * gthetatheta
            + 1 / self.metric.R**2
        )  # |grad(alpha)|**2
        galpha_cross_gr_2 = (
            self.metric.dalpha_dtheta**2 * gtheta_cross_gr2 + gphi_cross_gr2
        )  # |grad(alpha) x grad(r)|**2
        bcross_gradr_dot_gradalpha = galpha_dot_gr**2 / (
            np.sqrt(grr) * np.sqrt(galpha_cross_gr_2)
        ) - np.sqrt(grr) * galpha2 / np.sqrt(galpha_cross_gr_2)

        # interpolate to thetaloc
        grr0 = np.interp(self.thetaloc, self.metric.regulartheta, grr)
        dalpha_dtheta0 = np.interp(
            self.thetaloc, self.metric.regulartheta, self.metric.dalpha_dtheta
        )
        galpha_dot_gr0 = np.interp(
            self.thetaloc, self.metric.regulartheta, galpha_dot_gr
        )
        bcross_gradr_dot_gradalpha0 = np.interp(
            self.thetaloc, self.metric.regulartheta, bcross_gradr_dot_gradalpha
        )

        # map (Kn0_exp, Kb0_exp) to (kx0, ky0): [Ruiz PPCF 2022, notes on local mapping in k]. (kn0, kb0) normalized by rhos_loc:
        # Using ky = n q / r here!
        self.ky0 = (
            -(self.metric.q / self.geometry.rho)
            * self.Kb0_exp
            / bcross_gradr_dot_gradalpha0
        )
        self.kx0 = (
            self.Kn0_exp / np.sqrt(grr0)
            + self.ky0 * (self.geometry.rho / self.metric.q) * galpha_dot_gr0 / grr0
        )

        # compute k-resolutions (approx at the OMP): cf. Ruiz Ruiz PPCF 2020, eq. (13)
        if self.filter_type == "gauss":
            self.dkx0 = 2 / (self.wR * np.sqrt(grr0))
            self.dky0 = (
                2 * self.geometry.kappa * self.metric.q / (self.wZ * dalpha_dtheta0)
            )
        else:
            raise ValueError(f"Pyro does not support filter_type = {self.filter_type}")

    def get_syn_fspec(self, t1, t2, savedir, if_save):
        """
        # Function that performs filtering and produces synthetic spectra
        # Steps:
        # 1: Load simulation data: pyro object, grids, moments ...
        # 2: For each case (eg. k in DBS/high-k), define filter. See filter Filter
        # 3: Apply filter on fluctuations. See apply_filter
        """

        self.t1 = t1
        self.t2 = t2
        # fsize = self.fsize

        pyro = self.pyro
        pyro.load_gk_output(load_moments=True, load_fluxes=True, load_fields=False)
        data = pyro.gk_output.data

        # grids
        self.time = data.time

        # TODO need to make this ky = nq/r * rhos_unit, following logic only work if kx is in units of rhoref_pyro

        # Correct for difference in numerical bunit_over_b0 and input file bunit_over_b0
        numerical_factor = (
            pyro.gk_input.get_local_geometry().bunit_over_b0.m
            / pyro.local_geometry.bunit_over_b0
        )

        self.kx = data["kx"].data * numerical_factor * pyro.norms.cgyro.rhoref**-1
        self.ky = data["ky"].data * numerical_factor * pyro.norms.cgyro.rhoref**-1

        # theta index in theta closest to thetaloc
        self.ith = abs(data.theta - self.thetaloc).argmin()
        tmp_time = self.time[self.time > t1 * self.time[-1]]
        self.sim_time = tmp_time[tmp_time < t2 * self.time[-1]]
        density_all = data["density"].sel(species="electron").pint.dequantify()
        dens = density_all.where(density_all.time > t1 * self.time[-1], drop=True)
        dens = dens.where(dens.time < t2 * self.time[-1], drop=True)
        phikxkyt = np.squeeze(dens.sel(theta=data.theta[self.ith]))
        self.phi2kxky = (np.abs(phikxkyt) ** 2).mean(dim="time")

        self.ps_locsyn = []
        self.ps_hann = []
        self.ps_kxavg = []
        self.ps_kxavg_nz = []
        self.ps_kxavg_zon = []
        self.ps_nz = []
        self.ps_zon = []
        self.dne2_locsyn_ky = []
        self.dne2_kxavg_ky = []
        self.ps_dop = []

        self.phi2f_f2_locsyn = []
        self.phi2f_f2_kxavg = []
        self.phi2f_f2_kxavg_nz = []
        self.phi2f_f2_kxavg_zon = []
        self.phi2f_f2_nz = []
        self.phi2f_f2_zon = []
        self.phi2f_f2_hann = []
        self.phi2f_f2_dop = []

        self.filters = []

        for ik in np.arange(np.size(self.Kn0_exp)):
            print(" ")
            print("     Filtering channel = " + str(ik))

            # call filter
            self.filters.append(
                Filter(
                    self.filter_type,
                    self.kx,
                    self.ky,
                    self.kx0[ik],
                    self.ky0[ik],
                    self.dkx0,
                    self.dky0,
                )
            )

            # filter fluct: scattered power
            self.ps_locsyn.append(
                self.apply_filter(phikxkyt, self.filters[ik].F2, dims=["kx", "ky"])
            )

            # ps using hanning window in time
            self.ps_hann.append(
                self.apply_filter(
                    phikxkyt * np.hanning(np.size(self.sim_time)),
                    self.filters[ik].F2,
                    dims=["kx", "ky"],
                )
            )

            self.ps_kxavg.append(
                self.apply_filter(
                    phikxkyt, self.filters[ik].F2_kxavg, dims=["kx", "ky"]
                )
            )

            self.ps_kxavg_nz.append(
                self.apply_filter(
                    phikxkyt, self.filters[ik].F2_kxavg_nz, dims=["kx", "ky"]
                )
            )

            self.ps_kxavg_zon.append(
                self.apply_filter(
                    phikxkyt, self.filters[ik].F2_kxavg_zon, dims=["kx", "ky"]
                )
            )

            self.ps_nz.append(
                self.apply_filter(phikxkyt, self.filters[ik].F2_nz, dims=["kx", "ky"])
            )

            self.ps_zon.append(
                self.apply_filter(phikxkyt, self.filters[ik].F2_zon, dims=["kx", "ky"])
            )

            self.dne2_locsyn_ky.append(
                self.apply_filter(phikxkyt, self.filters[ik].F2, dims=["kx", "time"])
            )

            self.dne2_kxavg_ky.append(
                self.apply_filter(
                    phikxkyt, self.filters[ik].F2_kxavg, dims=["kx", "time"]
                )
            )

            # add Doppler shift to field/moment data
            self.w0 = 0 * pyro.local_species.electron.omega0
            vy = self.geometry.rho / self.geometry.q * self.w0

            # Dimensions don't have units
            phikxkyt_dop = phikxkyt * np.exp(
                phikxkyt.ky * phikxkyt.time * vy * -1j * self.ky.units * self.time.units
            )
            phikxkyfdop = xrft.fft(
                phikxkyt_dop, true_phase=True, true_amplitude=True, dim=["time"]
            )  # Fourier Transform w/ consideration of phase

            phikxkyf = xrft.fft(
                phikxkyt, true_phase=True, true_amplitude=True, dim=["time"]
            )  # Fourier Transform w/ consideration of phase
            phikxkyf_hann = xrft.fft(
                phikxkyt * np.hanning(np.size(self.sim_time)),
                true_phase=True,
                true_amplitude=True,
                dim=["time"],
            )  # Fourier Transform w/ consideration of phase

            self.phi2f_f2_locsyn.append(
                self.apply_filter(phikxkyf, self.filters[ik].F2, dims=["kx", "ky"])
            )
            self.phi2f_f2_kxavg.append(
                self.apply_filter(
                    phikxkyf, self.filters[ik].F2_kxavg, dims=["kx", "ky"]
                )
            )
            self.phi2f_f2_kxavg_nz.append(
                self.apply_filter(
                    phikxkyf, self.filters[ik].F2_kxavg_nz, dims=["kx", "ky"]
                )
            )
            self.phi2f_f2_kxavg_zon.append(
                self.apply_filter(
                    phikxkyf, self.filters[ik].F2_kxavg_zon, dims=["kx", "ky"]
                )
            )
            self.phi2f_f2_nz.append(
                self.apply_filter(phikxkyf, self.filters[ik].F2_nz, dims=["kx", "ky"])
            )
            self.phi2f_f2_zon.append(
                self.apply_filter(phikxkyf, self.filters[ik].F2_zon, dims=["kx", "ky"])
            )
            self.phi2f_f2_hann.append(
                self.apply_filter(phikxkyf_hann, self.filters[ik].F2, dims=["kx", "ky"])
            )
            self.phi2f_f2_dop.append(
                self.apply_filter(phikxkyfdop, self.filters[ik].F2, dims=["kx", "ky"])
            )

            phikx0ky0f = phikxkyf.isel(
                kx=self.filters[ik].indx, ky=self.filters[ik].indy
            )  # Fourier Transform w/ consideration of phase
            pskx0ky0f = np.abs(phikx0ky0f) ** 2

            pkf = np.sum(self.phi2f_f2_locsyn)
            pkf_hann = np.sum(self.phi2f_f2_hann)
            pkf_kx0ky0 = np.sum(pskx0ky0f)
            pks = np.mean(self.ps_locsyn)
            sigma_ks_hann = np.std(self.ps_locsyn)

            self.Pks[ik] = pks
            self.Sigma_ks_hann[ik] = sigma_ks_hann

        return [pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann]

    def plot_syn(self):
        """
        Function that generates all plots in the synthetic diagnostic
        """

        axis_font = {"fontname": "Arial", "size": str(self.fsize)}
        if_save = 0

        fs = self.eq.flux_surface(psi_n=self.psin)
        fs.plot_path(x_label="", y_label="", color="k")
        plt.plot(
            self.geometry.R.to("m"),
            self.geometry.Z.to("m"),
            "-r",
            label="local_geometry",
        )

        plt.plot(fs["R"].pint.to("m"), fs["Z"].pint.to("m"), "--b", label="fs")
        plt.plot(
            self.Rtmp.to("m"), self.Ztmp.to("m"), "ok", markersize=12, label="sc loc"
        )
        plt.axis("equal")
        plt.xlabel("R [m]", fontsize=self.fsize)
        plt.ylabel("Z [m]", fontsize=self.fsize)
        plt.title(" Poloidal location of scattering ", fontsize=self.fsize)
        plt.legend()
        plt.tick_params(labelsize=self.fsize)

        # For loop
        for ik in range(len(self.ky0)):
            plt.figure(102, figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(self.kx, self.filters[ik].Fx2_shift, ".-", label="ch = " + str(ik))
            plt.xlabel(r"$k_x\rho_s$", fontsize=self.fsize)
            plt.title(r"$|F_x(k_x)|^2$", fontsize=self.fsize)
            plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 2, 2)
            # plt.plot(self.ky, self.filters[ik].Fy2, ".-", label="ch = " + str(ik))
            plt.plot(self.ky, self.filters[ik].Fy2_shift, ".-", label="ch = " + str(ik))
            plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
            plt.title(r"$|F_y(k_y)|^2$", fontsize=self.fsize)
            plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.figure(20 + ik, figsize=(16, 7))
            plt.subplot(1, 3, 1)
            plt.plot(self.ky, self.dne2_locsyn_ky[ik], ".-", lw=2, c="b")
            # plt.plot(self.ky, self.dne2_kxavg_ky, ".-", lw=2, c="b", label=r"$avg. \ k_x$")
            plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
            plt.title(
                r"$|\delta \hat{n}|^2(k_y)_{k_{xDBS}}$, ch = " + str(ik),
                fontsize=self.fsize,
            )
            # plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 3, 2)
            plt.plot(self.sim_time, self.ps_locsyn[ik], c="b")
            plt.plot(self.sim_time, self.ps_kxavg[ik], c="k", label="avg. " + r"$P_s$ ")
            plt.plot(
                self.sim_time.data,
                self.Pks[ik] * np.ones(np.size(self.ps_locsyn[ik])),
                "--",
                c="b",
            )
            plt.xlabel(r"$t [c_s/a]$", fontsize=self.fsize)
            plt.title(
                r"$P_s(t) = \sum_{k_x, k_y} |\delta \hat{n}|^2 |F|^2 (t)$, $\omega_0 = $"
                + str(self.w0),
                fontsize=self.fsize,
            )
            # plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 3, 3)
            plt.semilogy(
                self.phi2f_f2_locsyn[ik].freq_time * 2 * np.pi,
                (self.phi2f_f2_locsyn[ik]),
                linestyle="-",
                lw=3,
                c="b",
            )
            plt.semilogy(
                self.phi2f_f2_dop[ik].freq_time * 2 * np.pi,
                (self.phi2f_f2_dop[ik]),
                linestyle="-",
                lw=2,
                c="r",
                label="Doppler shifted ch = " + str(ik),
            )

            # plt.legend()
            plt.title(r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=self.fsize)
            plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
            plt.tick_params(labelsize=self.fsize)

            plt.figure(40 + ik, figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.plot(
                self.ky,
                self.filters[ik].Fy2_shift,
                ".-",
                c="b",
                lw=2,
                label="ch = " + str(ik) + " (shift)",
            )
            plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
            plt.title(r"$|F_y(k_y)|^2$", fontsize=self.fsize)
            # plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 2, 2)
            plt.semilogy(
                self.phi2f_f2_locsyn[ik].freq_time * 2 * np.pi,
                (self.phi2f_f2_locsyn[ik]),
                linestyle="-",
                lw=2,
                c="b",
                label="ch = " + str(ik),
            )
            # plt.legend()
            plt.title(r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=self.fsize)
            plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
            plt.tick_params(labelsize=self.fsize)

            if if_save:
                if not os.path.exists(self.savedir):
                    os.mkdir(self.savedir)
                    os.chdir(self.savedir)
                else:
                    os.chdir(self.savedir)

                    plt.figure(102)
                plt.savefig("f2_kxky_1d.pdf")
                plt.figure(20 + ik)
                plt.savefig("dne2_ky_kxdbs_ps_freq_time_ch" + str(ik) + ".pdf")
                plt.figure(40 + ik)
                # plt.savefig("ps_freq_ky_ch" + str(ik) + "_zon-nz.pdf")
                plt.savefig("pssyn_freq_ky_ch" + str(ik) + "_.pdf")

        # plot (kx,ky) time averaged
        thetaplot = np.linspace(0, 2 * np.pi, 100)

        plt.figure(100, figsize=(8, 8))
        plt.contourf(self.ky, self.kx, np.log10(self.phi2kxky), levels=100)
        plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
        plt.ylabel(r"$k_x\rho_s$", fontsize=self.fsize)
        plt.title(
            r"$|\delta \hat{n}(k_x, k_y)|^2$, "
            + r"$\theta$ ="
            + str(np.round(self.geometry.theta[self.ith] * 180 / np.pi, 2))
            + r" $^o$, t = ["
            + str(np.round(self.t1 * self.time[-1].data, 2))
            + ", "
            + str(np.round(self.t2 * self.time[-2].data, 2))
            + "]",
            fontsize=self.fsize,
        )
        for ik in np.arange(np.size(self.Kn0_exp)):
            plt.plot(
                self.ky0[ik] + self.dky0 * np.sin(thetaplot),
                self.kx0[ik] + self.dkx0 * np.cos(thetaplot),
                linestyle="--",
                color="k",
                linewidth=3,
            )
            plt.plot(self.ky0[ik], self.kx0[ik], ".", markersize=12, color="black")
        plt.colorbar()
        plt.tick_params(labelsize=self.fsize)

        msize = 12

        # plot synthetic P(k)
        plt.figure(37, figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title(r"$P_{syn}(k_y\rho_{s,unit})$ [a.u.]", fontsize=self.fsize)
        plt.semilogy(self.kx0, self.Pks, ".-b", label="pks")
        # plt.legend()
        plt.xlabel(r"$k_x\rho_{s,unit}$", fontsize=self.fsize)
        plt.tick_params(labelsize=self.fsize)

        Kperp = np.sqrt(self.Kn0_exp**2 + self.Kb0_exp**2)

        plt.subplot(1, 2, 2)
        plt.title(r"$P_{syn}(k_\perp)$ [a.u.]", fontsize=self.fsize)
        plt.errorbar(
            Kperp,
            self.Pks,
            yerr=self.Sigma_ks_hann,
            marker=".",
            linestyle="-",
            color="b",
            label="pks",
            markersize=msize,
        )
        plt.xscale("log")
        plt.yscale("log")
        # plt.legend()
        plt.xlabel(r"$k_\perp [cm^{-1}]$", fontsize=self.fsize)
        plt.tick_params(labelsize=self.fsize)

        if if_save:
            os.chdir(self.savedir)
            self.phi2kxky.to_netcdf(path=self.savedir + "/phi2kxky.nc")
            np.savez(
                "phi2kxky_file.npz",
                x=self.phi2kxky.data,
                y=self.phi2kxky.kx.data,
                z=self.phi2kxky.ky.data,
            )
            np.savetxt("Ky0rhosunit.txt", self.ky0, delimiter=" ", newline=os.linesep)
            np.savetxt("Kx0rhosunit.txt", self.kx0, delimiter=" ", newline=os.linesep)
            np.savetxt("Kperprhosunit.txt", Kperp, delimiter=" ", newline=os.linesep)
            np.savetxt("Pks.txt", self.Pks, delimiter=" ", newline=os.linesep)
            np.savetxt(
                "Sigma_ks_hann.txt",
                self.Sigma_ks_hann,
                delimiter=" ",
                newline=os.linesep,
            )
            plt.figure(37)
            plt.savefig("pk_spec.pdf")
            plt.figure(100)
            plt.savefig("dn2kxky_2d.pdf")

        plt.show()

    def apply_filter(self, dn, fkxky, dims):

        # This function applies filter fkxky to fluctuating field dn(kx,ky)

        product = np.abs(dn) ** 2 * fkxky
        dn2_times_f2 = product.sum(dim=dims)
        return dn2_times_f2


class Filter:
    # Filter class that calculates the different filter types |F|^2(kx,ky)
    def __init__(self, filter_type, kx, ky, kx0, ky0, dkx0, dky0):

        # filter function
        KY, KX = np.meshgrid(ky, kx)  # * kx.units
        if filter_type == "gauss":

            self.indx = abs(kx - kx0).argmin()  # kx index closest to kx0

            self.F2 = np.exp(
                -2 * (KX - kx0) ** 2 / dkx0**2 - 2 * (KY - ky0) ** 2 / dky0**2
            )[
                :, :, np.newaxis
            ]  # (kx,ky,t)   slow part of filter |W|^2, see eq (3), (16) on Ruiz Ruiz PPCF 2022
            self.F2_nz = np.exp(
                -2 * (KX - kx0) ** 2 / dkx0**2 - 2 * (KY - ky0) ** 2 / dky0**2
            )[:, :, np.newaxis]
            self.F2_nz[:, 0, :] = 0
            self.F2_zon = self.F2 - self.F2_nz
            self.F2_kxavg = (
                dkx0
                / np.size(kx)
                * np.exp(-2 * (KY - ky0) ** 2 / dky0**2)[:, :, np.newaxis]
            )
            self.F2_kxavg_nz = (
                dkx0
                / np.size(kx)
                * np.exp(-2 * (KY - ky0) ** 2 / dky0**2)[:, :, np.newaxis]
            )
            self.F2_kxavg_nz[:, 0, :] = 0 * kx.units
            self.F2_kxavg_zon = self.F2_kxavg - self.F2_kxavg_nz

            self.indy = abs(ky - ky0).argmin()  # ky index closest to ky0

            self.Fx2 = np.exp(-2 * (kx - kx0) ** 2 / dkx0**2)
            self.Fx2_shift = np.exp(-2 * (kx - kx[self.indx]) ** 2 / dkx0**2)
            self.kx_fine = np.linspace(kx[0], kx[-1], np.size(kx) * 4)
            self.Fy2 = np.exp(-2 * (ky - ky0) ** 2 / dky0**2)
            self.Fy2_shift = np.exp(-2 * (ky - ky[self.indy]) ** 2 / dky0**2)
            self.Fy2_nz = np.exp(-2 * (ky - ky0) ** 2 / dky0**2)
            self.Fy2_nz[0] = 0
            self.Fy2_zon = self.Fy2 - self.Fy2_nz

            print(f"kx0 = {kx0}  kx_close = {kx[self.indx]}")
            print(f"ky0 = {ky0}  ky_close = {ky[self.indy]}\n")

            print(
                f"kx0*rhos_sim = {kx0:.4f}, kx_grid = {kx[self.indx]:.4f}, dkx_grid ={kx[1] - kx[0]:.4f}"
            )
            print(
                f"ky0*rhos_sim = {ky0:.4f}, ky_grid = {ky[self.indy]:.4f}, dky_grid ={ky[1] - ky[0]:.4f}"
            )
