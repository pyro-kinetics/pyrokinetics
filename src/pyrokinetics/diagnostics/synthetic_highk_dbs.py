import os
import sys

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xrft
from numpy.typing import ArrayLike

import pyrokinetics as pk
from pyrokinetics import Pyro, template_dir
from pyrokinetics.typing import PathLike


class SyntheticHighkDBS:
    """
    Synthetic diagnostic for producing synthetic frequency/k-spectra from gyrokinetic simulations

    
    diag: str           # Type of diagnostic: 'highk', 'dbs', 'rcdr', 'bes'
    filter_type: str    # Type of filter:   'gauss'   # 'bt_2d', 'bt_scotty' for beam tracing, 'gauss' for Gaussian filter
    Rloc: float         # Major radius location of scattering [m] :   [1.26] [m]      Bhavin 47107: 1.2689; David 22769: R=1.086, 1,137
    Zloc: float         # Z location of scattering [m] :        Bhavin 47107: 0.1692; David 22769: 0.18
    Kn0_exp:            # Normal wavenumber component of the scattered turbulence k [ArrayLike]     :   see definition in Ruiz Ruiz PPCF 2022
    Kb0_exp:            # Binormal wavenumber component of the scattered turbulence k [ArrayLike]   :   see definition in Ruiz Ruiz PPCF 2022
    wR: float           # Major radius spot size length of the filter function/spread function  [m] :   local sim: 
    wZ: float           # Vertical spot size length of the filter function/spread function  [m] :       
    eq_file:            # Equilibrium file used for the gyrokinetic simulations [PathLike]
    kinetics_file:      # Kinetics file used for the gyrokinetic simulations [PathLike]   
    simdir:             # Directory where simulation data is stored [PathLike]
    savedir:            # Directory where to store the synthetic diagnostic output [PathLike]  
    fsize:              # Size of font for plots

    # Steps:
    # 1. Inputs are diagnostic specific (diagnostic, filter, k, location, resolution, local rhos). See example_syn_hk_dbs.py
    # 2. Load equilibrium, kinetics files. Find scattering location theta. See __init__ in class SyntheticHighkDBS
    # 3. Map (kn, kb) to (kx, ky) for all k's / channels specified in 1. See function mapk
    # 4. Load GK output data (fluctuation moment file). See function get_syn_fspec 
    # 5. For each input condition (eg. for each k in DBS/highk), filter sim data. See class Filter, functions apply filter, get_syn_fspec
    # 6. Generate synthetic spectra and make plots. See functions get_syn_fspec, plot_syn
    """

    def __init__(
        self,
        diag: str,          # 
        filter_type: str,    # 
        Rloc: float,  # [1.26] [m]      Bhavin 47107: 1.2689; David 22769: R=1.086, 1,137
        Zloc: float,  # [m]        Bhavin 47107: 0.1692; David 22769: 0.18
        Kn0_exp: ArrayLike,  # np.asarray([21.637153 ]) # np.asarray([-21.637153])  #np.asarray([0, 0])       # [cm-1]   usually 0 for DBS, finite for high-k
        Kb0_exp: ArrayLike,  # np.asarray([2.701665 ])   # np.asarray([-2.701665])  # np.asarray([1.75, 6.903])       # [cm-1], Bhavin 47107: 6.903
        wR: float,  # [m]    local sim: do sinc function
        wZ: float,  # 2/1711.94563       # [m]    wZ 0.02 MAST-U
        eq_file: PathLike,  #
        kinetics_file: PathLike,  #
        simdir: PathLike,  #
        savedir: PathLike,  #
        fsize: int,
    ):
        
        # calcualte thetaloc
        pyro = Pyro(
            eq_file=eq_file,
            kinetics_file=kinetics_file,
            gk_file=simdir + "/input.cgyro",
        )
        self.pyro = pyro
        self.eq = pk.read_equilibrium(eq_file)
        self.psin = self.eq._psi_RZ_spline(
            Rloc * pyro.norms.units.meter, Zloc * pyro.norms.units.meter
        ) / (self.eq.psi_lcfs - self.eq.psi_axis)
        pyro.load_local(psi_n=self.psin, local_geometry="Miller")
        self.geometry = pyro.local_geometry
        pyro.load_metric_terms()
        
        
        self.metric = pyro.metric_terms
        self.diag = diag
        self.filter_type = filter_type
        self.Rloc = Rloc.to(pyro.norms.rhoref)
        self.Zloc = Zloc.to(pyro.norms.rhoref)
        self.Kn0_exp = Kn0_exp.to(pyro.norms.rhoref**-1)
        self.Kb0_exp = Kb0_exp.to(pyro.norms.rhoref**-1)
        self.wR = wR.to(pyro.norms.rhoref)
        self.wZ = wZ.to(pyro.norms.rhoref)
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

        # calcualte thetaloc
        pyro = Pyro(
            eq_file=eq_file,
            kinetics_file=kinetics_file,
            gk_file=simdir + "/input.cgyro",
        )
        self.pyro = pyro
        self.eq = pk.read_equilibrium(eq_file)
        self.psin = self.eq._psi_RZ_spline(Rloc * pyro.norms.units.meter, Zloc * pyro.norms.units.meter) / (
            self.eq.psi_lcfs - self.eq.psi_axis
        )
        pyro.load_local(psi_n=self.psin, local_geometry="Miller")
        self.geometry = pyro.local_geometry
        pyro.load_metric_terms()
        
        
        self.metric = pyro.metric_terms
        self.diag = diag
        self.filter_type = filter_type
        self.Rloc = Rloc.to(pyro.norms.rhoref)
        self.Zloc = Zloc.to(pyro.norms.rhoref)
        self.Kn0_exp = Kn0_exp.to(pyro.norms.rhoref**-1)
        self.Kb0_exp = Kb0_exp.to(pyro.norms.rhoref**-1)
        self.wR = wR.to(pyro.norms.rhoref)
        self.wZ = wZ.to(pyro.norms.rhoref)
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

        # normalizations
        [
            self.a_minor,
            self.te_ref,
            self.ne_ref,
            self.csa,
            self.rhos_unit,
        ] = self.get_units_norms(pyro)
        Qgbunit = (
            self.ne_ref * self.csa * self.a_minor * self.te_ref * (self.rhos_unit) ** 2
        )  # [W/m^2]
        print("QgB,unit = " + str(np.round(Qgbunit / 1e3, 4)) + " [kW/m^2]")

        # find thetaloc:
        thetatmp = self.geometry.theta[self.geometry.Z > self.geometry.Z0]
        Rtmp = self.geometry.R[self.geometry.Z > self.geometry.Z0] * self.a_minor  # [m]
        Ztmp = self.geometry.Z[self.geometry.Z > self.geometry.Z0] * self.a_minor  # [m]
        tmp_ind = np.argmin(np.abs(Rtmp - Rloc))
        self.thetaloc = thetatmp[tmp_ind]  # np.interp(Zloc, Ztmp, thetatmp)
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

        import matplotlib.pyplot as plt
        import numpy as np

        self.rhos_loc = self.rhos_unit / np.interp(self.thetaloc, self.metric.regulartheta, self.metric.B_magnitude * self.geometry.bunit_over_b0 )
        
        # get geo coefficients
        grr = self.metric.toroidal_contravariant_metric("r", "r")  # |grad r|^2
        grtheta = self.metric.toroidal_contravariant_metric(
            "r", "theta"
        )  # gradr . gradtheta
        gthetatheta = self.metric.toroidal_contravariant_metric(
            "theta", "theta"
        )  # |grad theta|^2
        gzz = self.metric.toroidal_contravariant_metric("zeta", "zeta")
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
        grtheta0 = np.interp(self.thetaloc, self.metric.regulartheta, grtheta)
        gthetatheta0 = np.interp(self.thetaloc, self.metric.regulartheta, gthetatheta)
        # gzz0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.toroidal_contravariant_metric("zeta", "zeta") )
        dalpha_dr0 = np.interp(
            self.thetaloc, self.metric.regulartheta, self.metric.dalpha_dr
        )
        dalpha_dtheta0 = np.interp(
            self.thetaloc, self.metric.regulartheta, self.metric.dalpha_dtheta
        )
        Jr0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.Jacobian)
        dRdr0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.dRdr)
        dRdtheta0 = np.interp(
            self.thetaloc, self.metric.regulartheta, self.metric.dRdtheta
        )
        dZdr0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.dZdr)
        dZdtheta0 = np.interp(
            self.thetaloc, self.metric.regulartheta, self.metric.dZdtheta
        )
        R0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.R)
        gtheta_cross_gr20 = np.interp(
            self.thetaloc, self.metric.regulartheta, gtheta_cross_gr2
        )
        gphi_cross_gr20 = np.interp(
            self.thetaloc, self.metric.regulartheta, gphi_cross_gr2
        )

        galpha_dot_gr0 = np.interp(
            self.thetaloc, self.metric.regulartheta, galpha_dot_gr
        )
        galpha20 = np.interp(self.thetaloc, self.metric.regulartheta, galpha2)
        galpha_cross_gr_20 = np.interp(
            self.thetaloc, self.metric.regulartheta, galpha_cross_gr_2
        )
        bcross_gradr_dot_gradalpha0 = np.interp(
            self.thetaloc, self.metric.regulartheta, bcross_gradr_dot_gradalpha
        )

        # map (Kn0_exp, Kb0_exp) to (kx0, ky0): [Ruiz PPCF 2022, notes on local mapping in k]. (kn0, kb0) normalized by rhos_loc:
        self.ky0 = -( self.metric.q / self.geometry.rho ) * self.Kb0_exp / bcross_gradr_dot_gradalpha0 * self.rhos_unit / self.rhos_loc
        self.kx0 = (
            self.Kn0_exp / np.sqrt(grr0) * self.rhos_unit / self.rhos_loc
            + self.ky0 * (self.geometry.rho / self.metric.q) * galpha_dot_gr0 / grr0
        )

        # compute k-resolutions (approx at the OMP): cf. Ruiz Ruiz PPCF 2020, eq. (13)
        if self.filter_type == "gauss":
            self.dkx0 = 2 / (self.wR * np.sqrt(grr0)) * self.rhos_unit
            self.dky0 = (
                2
                * self.geometry.kappa
                * self.metric.q
                / (self.wZ * dalpha_dtheta0)
                * self.rhos_unit
            )
            # elif filter == 'beam':
            # we need further work on this

    def get_syn_fspec(self, t1, t2, savedir, if_save):

        """    
        # Function that performs filtering and produces synthetic spectra
        # Steps:
        # 1: Load simulation data: pyro object, grids, moments ... 
        # 2: For each case (eg. k in DBS/high-k), define filter. See filter Filter
        # 3: Apply filter on fluctuations. See apply_filter
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        import xrft

        self.t1 = t1
        self.t2 = t2
        # fsize = self.fsize

        pyro = self.pyro
        pyro.load_gk_output(
            load_moments=True, load_fluxes=True, load_fields=False
        )  # pyro.load_gk_output()   #
        data = pyro.gk_output.data  # data = pyro.gk_output   #

        # grids
        self.time = data.time
        self.kx = data["kx"]
        self.ky = data["ky"]

        self.ith = abs(
            data.theta - self.thetaloc
        ).argmin()  # theta index in theta closest to thetaloc
        tmp_time = self.time[self.time > t1 * self.time[-1]]
        self.sim_time = tmp_time[tmp_time < t2 * self.time[-1]]
        density_all = data["density"].sel(species="electron").pint.dequantify()
        dens = density_all.where(density_all.time > t1 * self.time[-1], drop=True)
        dens = dens.where(dens.time < t2 * self.time[-1], drop=True)
        phikxkyt = np.squeeze(
            dens.sel(theta=data.theta[self.ith])
        )  # (kx, spec, ky, t), theta=0
        self.phi2kxky = (np.abs(phikxkyt) ** 2).mean(dim="time")  # phi2(kx,ky,t)

        axis_font = {"fontname": "Arial", "size": str(self.fsize)}
        self.Pks = np.empty(np.size(self.Kn0_exp))

        # continue here
        # define here xarray of quantities to store/plot
        # move plots to plot_syn

        # self.filter.F2 = xr.DataArray( np.zeros(np.shape( [np.size(self.kx), np.size(self.ky), np.size(self.Kn0_exp)] )), dims=[('kx', 'ky', 'k0')])

        for ik in np.arange(np.size(self.Kn0_exp)):
            print(" ")
            print("     Filtering channel = " + str(ik))

            ## call filter
            self.filter = Filter(
                self.filter_type,
                self.kx,
                self.ky,
                self.kx0[ik],
                self.ky0[ik],
                self.dkx0,
                self.dky0,
            )

            ## filter fluct: scattered power
            self.ps_locsyn = self.apply_filter(
                phikxkyt, self.filter.F2, dims=["kx", "ky"]
            )
            self.ps_hann = self.apply_filter(
                phikxkyt * np.hanning(np.size(self.sim_time)),
                self.filter.F2,
                dims=["kx", "ky"],
            )  # ps using hanning window in time
            self.ps_kxavg = self.apply_filter(
                phikxkyt, self.filter.F2_kxavg, dims=["kx", "ky"]
            )
            self.ps_kxavg_nz = self.apply_filter(
                phikxkyt, self.filter.F2_kxavg_nz, dims=["kx", "ky"]
            )
            self.ps_kxavg_zon = self.apply_filter(
                phikxkyt, self.filter.F2_kxavg_zon, dims=["kx", "ky"]
            )
            self.ps_nz = self.apply_filter(
                phikxkyt, self.filter.F2_nz, dims=["kx", "ky"]
            )
            self.ps_zon = self.apply_filter(
                phikxkyt, self.filter.F2_zon, dims=["kx", "ky"]
            )
            self.dne2_locsyn_ky = self.apply_filter(
                phikxkyt, self.filter.F2, dims=["kx", "time"]
            )
            self.dne2_kxavg_ky = self.apply_filter(
                phikxkyt, self.filter.F2_kxavg, dims=["kx", "time"]
            )

            # add Doppler shift to field/moment data
            w0 = 0  #   pyro.local_species.electron.omega0
            vy = self.geometry.rho / self.geometry.q * w0  # vy = r/q*w0
            phikxkyt_dop = phikxkyt * np.exp(-1j * vy * phikxkyt.ky * phikxkyt.time)
            phi2kxkyt_dop = np.abs(phikxkyt_dop) ** 2  # phi2[kx, ky, t]
            ps_dop = np.sum(
                np.sum(phi2kxkyt_dop * self.filter.F2, 0), 0
            )  # sum filter*|dn|2 over kx, ky
            ps_re_dop = np.sum(
                np.sum(np.real(phikxkyt_dop) ** 2 * self.filter.F2, 0), 0
            )

            phikxkyf = xrft.fft(
                phikxkyt, true_phase=True, true_amplitude=True, dim=["time"]
            )  # Fourier Transform w/ consideration of phase
            phikxkyf_hann = xrft.fft(
                phikxkyt * np.hanning(np.size(self.sim_time)),
                true_phase=True,
                true_amplitude=True,
                dim=["time"],
            )  # Fourier Transform w/ consideration of phase

            self.phi2f_f2_locsyn = self.apply_filter(
                phikxkyf, self.filter.F2, dims=["kx", "ky"]
            )
            self.phi2f_f2_kxavg = self.apply_filter(
                phikxkyf, self.filter.F2_kxavg, dims=["kx", "ky"]
            )
            self.phi2f_f2_kxavg_nz = self.apply_filter(
                phikxkyf, self.filter.F2_kxavg_nz, dims=["kx", "ky"]
            )
            self.phi2f_f2_kxavg_zon = self.apply_filter(
                phikxkyf, self.filter.F2_kxavg_zon, dims=["kx", "ky"]
            )
            self.phi2f_f2_nz = self.apply_filter(
                phikxkyf, self.filter.F2_nz, dims=["kx", "ky"]
            )
            self.phi2f_f2_zon = self.apply_filter(
                phikxkyf, self.filter.F2_zon, dims=["kx", "ky"]
            )
            self.phi2f_f2_hann = self.apply_filter(
                phikxkyf_hann, self.filter.F2, dims=["kx", "ky"]
            )

            phikx0ky0f = phikxkyf.sel(
                kx=self.kx[self.filter.indx], ky=self.ky[self.filter.indy]
            )  # Fourier Transform w/ consideration of phase
            pskx0ky0f = np.abs(phikx0ky0f) ** 2

            phikxkyfdop = xrft.fft(
                phikxkyt_dop, true_phase=True, true_amplitude=True, dim=["time"]
            )  # Fourier Transform w/ consideration of phase
            phi2fdop_f2 = self.apply_filter(
                phikxkyfdop, self.filter.F2, dims=["kx", "ky"]
            )
            # phi2fdop_f2 = np.sum(np.sum(np.abs(phikxkyfdop) ** 2 * self.filter.F2, 0), 0)

            f0_avg = (self.phi2f_f2_locsyn.freq_time * self.phi2f_f2_locsyn).sum(
                dim="freq_time"
            ) / (self.phi2f_f2_locsyn).sum(dim="freq_time")
            fdop_avg = (phi2fdop_f2.freq_time * phi2fdop_f2).sum(dim="freq_time") / (
                phi2fdop_f2
            ).sum(dim="freq_time")

            deltaf_dop = fdop_avg - f0_avg  # [vt/a]
            deltaf_theory = (
                self.ky0[ik] * w0 * self.geometry.rho / (2 * np.pi * self.geometry.q)
            )  # [vt/a]

            plt.figure(102, figsize=(12, 6))
            plt.subplot(1, 2, 1)
            # plt.plot(self.kx, self.filter.Fx2, ".-", label="ch = " + str(ik) + " (gauss)")
            plt.plot(self.kx, self.filter.Fx2_shift, ".-", label="ch = " + str(ik))
            plt.xlabel(r"$k_x\rho_s$", fontsize=self.fsize)
            plt.title(r"$|F_x(k_x)|^2$", fontsize=self.fsize)
            plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 2, 2)
            # plt.plot(self.ky, self.filter.Fy2, ".-", label="ch = " + str(ik))
            plt.plot(self.ky, self.filter.Fy2_shift, ".-", label="ch = " + str(ik))
            plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
            plt.title(r"$|F_y(k_y)|^2$", fontsize=self.fsize)
            plt.legend()
            plt.tick_params(labelsize=self.fsize)

            pkf = np.sum(self.phi2f_f2_locsyn)
            pkf_hann = np.sum(self.phi2f_f2_hann)
            pkf_kx0ky0 = np.sum(pskx0ky0f)
            pks = np.mean(self.ps_locsyn)
            sigma_ks_hann = np.std(self.ps_locsyn)

            plt.figure(20 + ik, figsize=(16, 7))
            plt.subplot(1, 3, 1)
            plt.plot(self.ky, self.dne2_locsyn_ky, ".-", lw=2, c="b")
            # plt.plot(self.ky, self.dne2_kxavg_ky, ".-", lw=2, c="b", label=r"$avg. \ k_x$")
            plt.xlabel(r"$k_y\rho_s$", fontsize=self.fsize)
            plt.title(
                r"$|\delta \hat{n}|^2(k_y)_{k_{xDBS}}$, ch = " + str(ik),
                fontsize=self.fsize,
            )
            # plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 3, 2)
            plt.plot(self.sim_time, self.ps_locsyn, c="b")
            # plt.plot(self.sim_time, self.ps_kxavg, c="k", label="avg. " + r"$P_s$ ")
            plt.plot(
                self.sim_time.data,
                pks.data * np.ones(np.size(self.ps_locsyn.data)),
                "--",
                c="b",
            )
            plt.xlabel(r"$t [c_s/a]$", fontsize=self.fsize)
            plt.title(
                r"$P_s(t) = \sum_{k_x, k_y} |\delta \hat{n}|^2 |F|^2 (t)$, $\omega_0 = $"
                + str(w0),
                fontsize=self.fsize,
            )
            # plt.legend()
            plt.tick_params(labelsize=self.fsize)

            plt.subplot(1, 3, 3)
            plt.semilogy(
                self.phi2f_f2_locsyn.freq_time * 2 * np.pi,
                (self.phi2f_f2_locsyn),
                linestyle="-",
                lw=3,
                c="b",
            )
            # plt.semilogy( self.phi2f_f2_locsyn.freq_time * 2 * np.pi, (self.phi2f_f2_kxavg), linestyle="-", lw=3, c="k", label=r"$avg. \ k_x$")

            # plt.legend()
            plt.title(r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=self.fsize)
            plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
            plt.tick_params(labelsize=self.fsize)

            plt.figure(40 + ik, figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.plot(
                self.ky,
                self.filter.Fy2_shift,
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
            label = "ch = " + str(ik)
            plt.semilogy(
                self.phi2f_f2_locsyn.freq_time * 2 * np.pi,
                (self.phi2f_f2_locsyn),
                linestyle="-",
                lw=2,
                c="b",
                label="ch = " + str(ik),
            )
            # plt.legend()
            plt.title(r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=self.fsize)
            plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
            plt.tick_params(labelsize=self.fsize)

            """
            plt.subplot(1, 3, 3)
            plt.semilogy(
                self.phi2f_f2_locsyn.freq_time * 2 * np.pi,
                (self.phi2f_f2_kxavg),
                linestyle="-",
                lw=2,
                c="k",
                label=r"$avg. \ k_x \ (tot)$",
            )
            plt.legend()
            plt.title("Avg. " r"$-k_x \ \tilde{P}_s(f)$, ch = " + str(ik), fontsize=self.fsize)
            plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
            plt.tick_params(labelsize=self.fsize)
            """

            if if_save:
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                    os.chdir(savedir)
                else:
                    os.chdir(savedir)

                plt.figure(102)
                plt.savefig("f2_kxky_1d.pdf")
                plt.figure(20 + ik)
                plt.savefig("dne2_ky_kxdbs_ps_freq_time_ch" + str(ik) + ".pdf")
                plt.figure(40 + ik)
                # plt.savefig("ps_freq_ky_ch" + str(ik) + "_zon-nz.pdf")
                plt.savefig("pssyn_freq_ky_ch" + str(ik) + "_.pdf")

            self.Pks[ik] = pks
            self.Sigma_ks_hann[ik] = sigma_ks_hann

        return [pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann]

    def get_units_norms(self, pyro):
        
        # Is this necessary ??
        
        import numpy as np

        a_minor = (1.0 * pyro.norms.lref).to(
            "m"
        )  # self.geometry.a_minor   # pyro.eq.a_minor       # (1.0*pyro.norms.lref).to('m')
        te_ref = (1 * pyro.norms.tref).to(
            "kg * m**2 / s**2 "
        )  # pyro.local_species.electron.temp.to_base_units()  # [kg*m^2/s^2]
        ne_ref = (1 * pyro.norms.nref).to(
            "m**-3"
        )  # pyro.local_species.electron.dens.to_base_units()  # [m^-3]
        csa = ((1 * pyro.norms.vref) / a_minor).to("second**-1")
        rhos_unit = (1 * pyro.norms.cgyro.rhoref).to("m")

        return [a_minor.m, te_ref.m, ne_ref.m, csa.m, rhos_unit.m]

    def plot_syn(self):
        
        """
        Function that generates all plots in the synthetic diagnostic
        """
        
        axis_font = {"fontname": "Arial", "size": str(self.fsize)}
        if_save = 0
        # geometry = self.geometry

        fs = self.eq.flux_surface(psi_n=self.psin)
        fs.plot_path(x_label="", y_label="", color="k")
        plt.plot(
            self.geometry.R * self.a_minor,
            self.geometry.Z * self.a_minor,
            "-r",
            label="local_geometry",
        )
        plt.plot(fs["R"], fs["Z"], "--b", label="fs")
        plt.plot(self.Rtmp, self.Ztmp, "ok", markersize=12, label="sc loc")
        plt.axis("equal")
        plt.ylabel("R [m]", fontsize=self.fsize)
        plt.xlabel("Z [m]", fontsize=self.fsize)
        plt.title(" Poloidal location of scattering ", fontsize=self.fsize)
        plt.legend()
        plt.tick_params(labelsize=self.fsize)

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

        ## filter function
        KY, KX = np.meshgrid(ky, kx) * kx.units
        if filter_type == "gauss":
            
            self.F2 = np.exp( -2 * (KX - kx0) ** 2 / dkx0**2 - 2 * (KY - ky0) ** 2 / dky0**2)[:, :, np.newaxis]  # (kx,ky,t)   slow part of filter |W|^2, see eq (3), (16) on Ruiz Ruiz PPCF 2022
            self.F2_nz = np.exp( -2 * (KX - kx0) ** 2 / dkx0**2 - 2 * (KY - ky0) ** 2 / dky0**2 )[:, :, np.newaxis]
            self.F2_nz[:, 0, :] = 0
            self.F2_zon = self.F2 - self.F2_nz
            self.F2_kxavg = ( dkx0 / np.size(kx) * np.exp(-2 * (KY - ky0) ** 2 / dky0**2)[:, :, np.newaxis] )
            self.F2_kxavg_nz = ( dkx0 / np.size(kx) * np.exp(-2 * (KY - ky0) ** 2 / dky0**2)[:, :, np.newaxis] )
            self.F2_kxavg_nz[:, 0, :] = 0 * kx.units
            self.F2_kxavg_zon = self.F2_kxavg - self.F2_kxavg_nz

            dkx = kx[1] - kx[0]
            Lx = float(2 * np.pi / dkx)

            print(kx0)
            print(kx)
            print(kx.pint.to("meter"))
            self.indx = abs(kx - kx0).argmin()  # kx index closest to kx0
            self.indy = abs(ky - ky0).argmin()  # ky index closest to ky0

            self.Fx2 = np.exp(-2 * (kx - kx0) ** 2 / dkx0**2)
            self.Fx2_shift = np.exp(-2 * (kx - kx[self.indx]) ** 2 / dkx0**2)
            self.kx_fine = np.linspace(kx[0], kx[-1], np.size(kx) * 4)
            self.Fy2 = np.exp(-2 * (ky - ky0) ** 2 / dky0**2)
            self.Fy2_shift = np.exp(-2 * (ky - ky[self.indy]) ** 2 / dky0**2)
            self.Fy2_nz = np.exp(-2 * (ky - ky0) ** 2 / dky0**2)
            self.Fy2_nz[0] = 0
            self.Fy2_zon = self.Fy2 - self.Fy2_nz

            # grr0 = np.interp(self.thetaloc, self.metric.regulartheta, self.metric.toroidal_contravariant_metric("r", "r") )
            # self.Fx2_sinc = Lx / np.sqrt(grr0) * np.sinc((kx - kx0) * Lx / (2 * np.pi))
            # self.Fx2_sinc_fine = (Lx / np.sqrt(grr0) * np.sinc((kx_fine - kx0) * Lx / (2 * np.pi)))
            # F2_shift = np.exp( - 2*(KX-kx[indx].data)**2/dself.kx0**2 - 2*(KY-ky[indy].data)**2/self.dky0**2 )[:,:,np.newaxis]    # slow part of filter |W|^2, see eq (3), (16) on Ruiz Ruiz PPCF 2022
            # ps_shift = np.sum( np.sum(phi2kxkyt * F2_shift,0), 0 )   # sum filter*|dn|2 over kx, ky

            print("kx0 = " + str(kx0), "        kx_close = " + str(kx[self.indx].data))
            print("ky0 = " + str(ky0), "        ky_close = " + str(ky[self.indy].data))
            print(" ")

            print(
                "kx0*rhos_sim = "
                + str(kx0.round(4))
                + ",   kx_grid = "
                + str(kx[self.indx].values.round(4))
                + ",    dkx_grid = "
                + str((kx[1] - kx[0]).values.round(4))
            )
            print(
                "ky0*rhos_sim = "
                + str(ky0.round(4))
                + ",   ky_grid = "
                + str(ky[self.indy].values.round(4))
                + ",    dky_grid = "
                + str((ky[1] - ky[0]).values.round(4))
            )
