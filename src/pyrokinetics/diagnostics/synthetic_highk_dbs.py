import pyrokinetics as pk
from pyrokinetics import Pyro, template_dir
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import xrft
import xarray as xr
import sys
import os
from numpy.typing import ArrayLike
from pyrokinetics.typing import PathLike


class Synthetic_highk_dbs:
    """
    Rloc: float         # [1.26] [m]      Bhavin 47107: 1.2689; David 22769: R=1.086, 1,137
    Zloc: float         # [m]        Bhavin 47107: 0.1692; David 22769: 0.18
    Kn0_exp:    ArrayLike    #np.asarray([21.637153 ]) # np.asarray([-21.637153])  #np.asarray([0, 0])       # [cm-1]   usually 0 for DBS, finite for high-k
    Kb0_exp:    ArrayLike    #np.asarray([2.701665 ])   # np.asarray([-2.701665])  # np.asarray([1.75, 6.903])       # [cm-1], Bhavin 47107: 6.903
    wR: float       # [m]    local sim: do sinc function
    wZ: float       # 2/1711.94563       # [m]    wZ 0.02 MAST-U
    eq_file: PathLike   #
    kinetics_file: PathLike     #
    simdir: PathLike    #
    savedir: PathLike   #

    # Steps:
    # 1. Inputs are diagnostic specific (diagnostic, filter, k, location, resolution, local rhos)
    # 2. Load equilibrium, kinetics files
    # 3. Map (kn, kb) to (kx, ky) for all k's / channels specified in 1.
    # 4. Load GK output data
    # 5. Filter sim data : synthetic spectra

    """

    def __init__(
        self,
        diag: str,  # 'highk' # 'highk', 'rcdr', 'bes'
        syn_filter: str,  # 'gauss'   # 'bt_2d', 'bt_scotty' for beam tracing, 'gauss' for Gaussian filter
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
        self.diag = diag
        self.syn_filter = syn_filter
        self.Rloc = Rloc
        self.Zloc = Zloc
        self.Kn0_exp = Kn0_exp
        self.Kb0_exp = Kb0_exp
        self.wR = wR
        self.wZ = wZ
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
        meter = pyro.norms.units.meter
        self.psin = self.eq._psi_RZ_spline(Rloc * meter, Zloc * meter) / (
            self.eq.psi_lcfs - self.eq.psi_axis
        )
        pyro.load_local(psi_n=self.psin, local_geometry="Miller")
        self.geo = pyro.local_geometry
        pyro.load_metric_terms()
        self.m = pyro.metric_terms

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
        thetatmp = self.geo.theta[self.geo.Z > self.geo.Z0]
        Rtmp = self.geo.R[self.geo.Z > self.geo.Z0] * self.a_minor  # [m]
        Ztmp = self.geo.Z[self.geo.Z > self.geo.Z0] * self.a_minor  # [m]
        tmp_ind = np.argmin(np.abs(Rtmp - Rloc))
        self.thetaloc = thetatmp[tmp_ind]  # np.interp(Zloc, Ztmp, thetatmp)

        print("psin = " + str(self.psin))
        print("theta/pi = " + str(self.thetaloc / np.pi))
        print("Rloc = " + str(Rloc) + " [m]")
        print("Zloc = " + str(Zloc) + " [m]")

    def mapk(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # maps k from (kn, kb) to k in GK code : (kR, kZ) in the future
        # in pyrokinetics, (kx, ky) is the same as CGYRO: kx=2*pi*p/Lx, ky=n*q/r

        thetaloc = self.thetaloc
        kn0 = self.Kn0_exp
        kb0 = self.Kb0_exp
        geo = self.geo
        syn_filter = self.syn_filter
        m = self.m
        wR = self.wR
        wZ = self.wZ
        rhos_sim = self.rhos_unit
        bmag = m.B_magnitude * geo.bunit_over_b0  # = bmag / B0
        bmag_loc = np.interp(self.thetaloc, m.regulartheta, bmag)
        rhos_loc = rhos_sim / bmag_loc
        self.rhos_loc = rhos_loc

        # geo stuff
        rho = geo.rho
        betap = geo.beta_prime
        dpsidr = geo.dpsidr
        self.bunit_over_b0 = geo.bunit_over_b0
        shat = geo.shat
        kappa = geo.kappa

        # get geo coefficients
        theta_grid = m.regulartheta
        grr = m.toroidal_contravariant_metric("r", "r")  # |grad r|^2
        grtheta = m.toroidal_contravariant_metric("r", "theta")  # gradr . gradtheta
        gthetatheta = m.toroidal_contravariant_metric(
            "theta", "theta"
        )  # |grad theta|^2
        gzz = m.toroidal_contravariant_metric("zeta", "zeta")
        dalpha_dtheta = m.dalpha_dtheta
        dalpha_dr = m.dalpha_dr
        Jr = m.Jacobian

        dRdr = m.dRdr
        dRdtheta = m.dRdtheta
        dZdr = m.dZdr
        dZdtheta = m.dZdtheta
        R = m.R
        q = m.q

        # more fancy terms
        gtheta_cross_gr2 = (R / Jr) ** 2
        gphi_cross_gr2 = grr / (R**2)

        # gradients of alpha
        galpha_dot_gr = dalpha_dr * grr + dalpha_dtheta * grtheta
        galpha2 = (
            dalpha_dr**2 * grr
            + 2 * dalpha_dr * dalpha_dtheta * grtheta
            + dalpha_dtheta**2 * gthetatheta
            + 1 / R**2
        )  # |grad(alpha)|**2
        galpha_cross_gr_2 = (
            dalpha_dtheta**2 * gtheta_cross_gr2 + gphi_cross_gr2
        )  # |grad(alpha) x grad(r)|**2
        bcross_gradr_dot_gradalpha = galpha_dot_gr**2 / (
            np.sqrt(grr) * np.sqrt(galpha_cross_gr_2)
        ) - np.sqrt(grr) * galpha2 / np.sqrt(galpha_cross_gr_2)

        # interpolate to thetaloc
        grr0 = np.interp(thetaloc, theta_grid, grr)
        grtheta0 = np.interp(thetaloc, theta_grid, grtheta)
        gthetatheta0 = np.interp(thetaloc, theta_grid, gthetatheta)
        gzz0 = np.interp(thetaloc, theta_grid, gzz)
        dalpha_dr0 = np.interp(thetaloc, theta_grid, dalpha_dr)
        dalpha_dtheta0 = np.interp(thetaloc, theta_grid, dalpha_dtheta)
        Jr0 = np.interp(thetaloc, theta_grid, Jr)
        dRdr0 = np.interp(thetaloc, theta_grid, dRdr)
        dRdtheta0 = np.interp(thetaloc, theta_grid, dRdtheta)
        dZdr0 = np.interp(thetaloc, theta_grid, dZdr)
        dZdtheta0 = np.interp(thetaloc, theta_grid, dZdtheta)
        R0 = np.interp(thetaloc, theta_grid, R)
        gtheta_cross_gr20 = np.interp(thetaloc, theta_grid, gtheta_cross_gr2)
        gphi_cross_gr20 = np.interp(thetaloc, theta_grid, gphi_cross_gr2)

        galpha_dot_gr0 = np.interp(thetaloc, theta_grid, galpha_dot_gr)
        galpha20 = np.interp(thetaloc, theta_grid, galpha2)
        galpha_cross_gr_20 = np.interp(thetaloc, theta_grid, galpha_cross_gr_2)
        bcross_gradr_dot_gradalpha0 = np.interp(
            thetaloc, theta_grid, bcross_gradr_dot_gradalpha
        )

        # map (kn0, kb0) to (kx0, ky0): [Ruiz PPCF 2022, notes on local mapping in k]. (kn0, kb0) normalized by rhos_loc:
        self.ky0 = -(q / rho) * kb0 / bcross_gradr_dot_gradalpha0 * rhos_sim / rhos_loc
        self.kx0 = (
            kn0 / np.sqrt(grr0) * rhos_sim / rhos_loc
            + self.ky0 * (rho / q) * galpha_dot_gr0 / grr0
        )

        # tests
        R00 = (
            geo.Rmaj
        )  # gk_input_data["theta_grid_parameters"]["rmaj"]   # get actual rmaj0 from input file sim
        eps = rho / R00

        # compute k-resolutions (approx at the OMP): cf. Ruiz Ruiz PPCF 2020, eq. (13)
        if syn_filter == "gauss":
            self.dkx0 = 2 / (wR * np.sqrt(grr0)) * rhos_sim / 1e2
            self.dky0 = 2 * kappa * q / (wZ * dalpha_dtheta0) * rhos_sim / 1e2
            # elif filter == 'beam':
            # we need further work on this

    def get_syn_fspec(self, ik, t1, t2, savedir, if_save):
        import matplotlib.pyplot as plt
        import numpy as np
        import xrft
        import os

        self.t1 = t1
        self.t2 = t2
        fsize = self.fsize

        print(" ")
        print("     Filtering channel = " + str(ik))

        pyro = self.pyro
        pyro.load_gk_output(
            load_moments=True, load_fluxes=True, load_fields=False
        )  # pyro.load_gk_output()   #
        data = pyro.gk_output.data  # data = pyro.gk_output   #

        # grids
        time = data.time
        self.time = time
        self.kx = data["kx"]
        self.ky = data["ky"]
        ky = self.ky
        kx = self.kx

        theta = data.theta
        self.ith = abs(
            theta - self.thetaloc
        ).argmin()  # theta index in theta closest to thetaloc
        tmp_time = time[time > t1 * time[-1]]
        sim_time = tmp_time[tmp_time < t2 * time[-1]]
        density_all = data["density"].sel(species="electron").pint.dequantify()
        dens = density_all.where(density_all.time > t1 * time[-1], drop=True)
        dens = dens.where(dens.time < t2 * time[-1], drop=True)
        phikxkyt = np.squeeze(
            dens.sel(theta=theta[self.ith])
        )  # (kx, spec, ky, t), theta=0
        self.phi2kxky = (np.abs(phikxkyt) ** 2).mean(dim="time")  # phi2(kx,ky,t)

        geo = self.geo
        thetaloc = self.thetaloc
        m = self.m

        axis_font = {"fontname": "Arial", "size": str(fsize)}

        KY, KX = np.meshgrid(ky, kx)

        ## filter function
        F2 = np.exp(
            -2 * (KX - self.kx0[ik]) ** 2 / self.dkx0**2
            - 2 * (KY - self.ky0[ik]) ** 2 / self.dky0**2
        )[
            :, :, np.newaxis
        ]  # (kx,ky,t)   slow part of filter |W|^2, see eq (3), (16) on Ruiz Ruiz PPCF 2022
        F2_nz = np.exp(
            -2 * (KX - self.kx0[ik]) ** 2 / self.dkx0**2
            - 2 * (KY - self.ky0[ik]) ** 2 / self.dky0**2
        )[:, :, np.newaxis]
        F2_nz[:, 0, :] = 0
        F2_zon = F2 - F2_nz
        F2_kxavg = (
            self.dkx0
            / np.size(kx)
            * np.exp(-2 * (KY - self.ky0[ik]) ** 2 / self.dky0**2)[:, :, np.newaxis]
        )
        F2_kxavg_nz = (
            self.dkx0
            / np.size(kx)
            * np.exp(-2 * (KY - self.ky0[ik]) ** 2 / self.dky0**2)[:, :, np.newaxis]
        )
        F2_kxavg_nz[:, 0, :] = 0
        F2_kxavg_zon = F2_kxavg - F2_kxavg_nz

        ## filter fluct: scattered power
        phi2kxkyt = np.abs(phikxkyt) ** 2  # phi2[kx, ky, t]
        ps_locsyn = np.sum(np.sum(phi2kxkyt * F2, 0), 0)  # sum filter*|dn|2 over kx, ky
        ps_kxavg = np.sum(
            np.sum(phi2kxkyt * F2_kxavg, 0), 0
        )  # sum filter*|dn|2 over kx, ky
        ps_kxavg_nz = np.sum(
            np.sum(phi2kxkyt * F2_kxavg_nz, 0), 0
        )  # sum filter*|dn|2 over kx, ky
        ps_kxavg_zon = np.sum(
            np.sum(phi2kxkyt * F2_kxavg_zon, 0), 0
        )  # sum filter*|dn|2 over kx, ky
        ps_nz = np.sum(np.sum(phi2kxkyt * F2_nz, 0), 0)  # sum filter*|dn|2 over kx, ky
        ps_zon = np.sum(
            np.sum(phi2kxkyt * F2_zon, 0), 0
        )  # sum filter*|dn|2 over kx, ky
        dne2_locsyn_ky = np.sum(np.sum(phi2kxkyt * F2, 2), 0)
        dne2_kxavg_ky = np.sum(np.sum(phi2kxkyt * F2_kxavg, 2), 0)
        ps_hann = np.sum(np.sum(phi2kxkyt * F2, 0), 0) * np.hanning(
            np.size(sim_time)
        )  # ps using hanning window in time
        theta_grid = m.regulartheta
        grr = m.toroidal_contravariant_metric("r", "r")  # |grad r|^2
        grr0 = np.interp(thetaloc, theta_grid, grr)
        dkx = kx[1] - kx[0]
        Lx = float(2 * np.pi / dkx)

        indx = abs(kx - self.kx0[ik]).argmin()  # kx index closest to kx0
        indy = abs(ky - self.ky0[ik]).argmin()  # ky index closest to ky0

        Fx2 = np.exp(-2 * (kx - self.kx0[ik]) ** 2 / self.dkx0**2)
        Fx2_shift = np.exp(-2 * (kx - kx[indx]) ** 2 / self.dkx0**2)
        Fx2_sinc = Lx / np.sqrt(grr0) * np.sinc((kx - self.kx0[ik]) * Lx / (2 * np.pi))
        kx_fine = np.linspace(kx[0], kx[-1], np.size(kx) * 4)
        Fx2_sinc_fine = (
            Lx / np.sqrt(grr0) * np.sinc((kx_fine - self.kx0[ik]) * Lx / (2 * np.pi))
        )
        Fy2 = np.exp(-2 * (ky - self.ky0[ik]) ** 2 / self.dky0**2)
        Fy2_shift = np.exp(-2 * (ky - ky[indy]) ** 2 / self.dky0**2)
        Fy2_nz = np.exp(-2 * (ky - self.ky0[ik]) ** 2 / self.dky0**2)
        Fy2_nz[0] = 0
        Fy2_zon = Fy2 - Fy2_nz

        # F2_shift = np.exp( - 2*(KX-kx[indx].data)**2/dself.kx0**2 - 2*(KY-ky[indy].data)**2/self.dky0**2 )[:,:,np.newaxis]    # slow part of filter |W|^2, see eq (3), (16) on Ruiz Ruiz PPCF 2022
        # ps_shift = np.sum( np.sum(phi2kxkyt * F2_shift,0), 0 )   # sum filter*|dn|2 over kx, ky

        print("kx0 = " + str(self.kx0[ik]), "        kx_close = " + str(kx[indx].data))
        print("ky0 = " + str(self.ky0[ik]), "        ky_close = " + str(ky[indy].data))
        print(" ")

        print(
            "kx0*rhos_sim = "
            + str(self.kx0[ik].round(4))
            + ",   kx_grid = "
            + str(kx[indx].values.round(4))
            + ",    dkx_grid = "
            + str((kx[1] - kx[0]).values.round(4))
        )
        print(
            "ky0*rhos_sim = "
            + str(self.ky0[ik].round(4))
            + ",   ky_grid = "
            + str(ky[indy].values.round(4))
            + ",    dky_grid = "
            + str((ky[1] - ky[0]).values.round(4))
        )

        pskx0ky0 = phi2kxkyt[indx, indy, :]  # * np.pi/2 * dkx0 * self.dky0

        # add Doppler shift to field/moment data
        w0 = 0  #   pyro.local_species.electron.omega0
        vy = geo.rho / geo.q * w0  # vy = r/q*w0
        phikxkyt_dop = phikxkyt * np.exp(-1j * vy * phikxkyt.ky * phikxkyt.time)
        phi2kxkyt_dop = np.abs(phikxkyt_dop) ** 2  # phi2[kx, ky, t]
        ps_dop = np.sum(
            np.sum(phi2kxkyt_dop * F2, 0), 0
        )  # sum filter*|dn|2 over kx, ky
        ps_re_dop = np.sum(np.sum(np.real(phikxkyt_dop) ** 2 * F2, 0), 0)

        phikxkyf = xrft.dft(
            phikxkyt, true_phase=True, true_amplitude=True, dim=["time"]
        )  # Fourier Transform w/ consideration of phase
        phikxkyf_hann = xrft.dft(
            phikxkyt * np.hanning(np.size(sim_time)),
            true_phase=True,
            true_amplitude=True,
            dim=["time"],
        )  # Fourier Transform w/ consideration of phase
        phi2f_f2_locsyn = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2, 0), 0)
        phi2f_f2_kxavg = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2_kxavg, 0), 0)
        phi2f_f2_kxavg_nz = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2_kxavg_nz, 0), 0)
        phi2f_f2_kxavg_zon = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2_kxavg_zon, 0), 0)
        phi2f_f2_nz = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2_nz, 0), 0)
        phi2f_f2_zon = np.sum(np.sum(np.abs(phikxkyf) ** 2 * F2_zon, 0), 0)
        phi2f_f2_hann = np.sum(np.sum(np.abs(phikxkyf_hann) ** 2 * F2, 0), 0)
        phikx0ky0f = phikxkyf.sel(
            kx=kx[indx], ky=ky[indy]
        )  # Fourier Transform w/ consideration of phase
        pskx0ky0f = np.abs(phikx0ky0f) ** 2

        phikxkyfdop = xrft.dft(
            phikxkyt_dop, true_phase=True, true_amplitude=True, dim=["time"]
        )  # Fourier Transform w/ consideration of phase
        phi2fdop_f2 = np.sum(np.sum(np.abs(phikxkyfdop) ** 2 * F2, 0), 0)

        f0_avg = (phi2f_f2_locsyn.freq_time * phi2f_f2_locsyn).sum(dim="freq_time") / (
            phi2f_f2_locsyn
        ).sum(dim="freq_time")
        fdop_avg = (phi2fdop_f2.freq_time * phi2fdop_f2).sum(dim="freq_time") / (
            phi2fdop_f2
        ).sum(dim="freq_time")

        deltaf_dop = fdop_avg - f0_avg  # [vt/a]
        deltaf_theory = self.ky0[ik] * w0 * geo.rho / (2 * np.pi * geo.q)  # [vt/a]

        plt.figure(102, figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(kx, Fx2, ".-", label="ch = " + str(ik) + " (gauss)")
        plt.plot(kx, Fx2_shift, ".-", label="ch = " + str(ik) + " (gauss-shift))")
        plt.xlabel(r"$k_x\rho_s$", fontsize=fsize)
        plt.title(r"$|F_x(k_x)|^2$", fontsize=fsize)
        plt.legend()
        plt.tick_params(labelsize=fsize)

        plt.subplot(1, 2, 2)
        plt.plot(ky, Fy2, ".-", label="ch = " + str(ik))
        plt.plot(ky, Fy2_shift, ".-", label="ch = " + str(ik) + " (shift)")
        plt.xlabel(r"$k_y\rho_s$", fontsize=fsize)
        plt.title(r"$|F_y(k_y)|^2$", fontsize=fsize)
        plt.legend()
        plt.tick_params(labelsize=fsize)

        pkf = np.sum(phi2f_f2_locsyn)
        pkf_hann = np.sum(phi2f_f2_hann)
        pkf_kx0ky0 = np.sum(pskx0ky0f)
        pks = np.mean(ps_locsyn)
        sigma_ks_hann = np.std(ps_locsyn)

        plt.figure(20 + ik, figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.plot(ky, dne2_locsyn_ky, lw=2, c="k", label=r"$local \ k_x$")
        plt.plot(ky, dne2_kxavg_ky, ".-", lw=2, c="b", label=r"$avg. \ k_x$")
        plt.xlabel(r"$k_y\rho_s$", fontsize=fsize)
        plt.title(
            "Local & avg. " r"$|\delta n_e|^2(k_y)_{k_{xDBS}}$, ch = " + str(ik),
            fontsize=fsize,
        )
        plt.legend()
        plt.tick_params(labelsize=fsize)

        plt.subplot(1, 3, 2)
        plt.plot(sim_time, ps_locsyn, c="k", label="local " + r"$P_s$ ")
        plt.plot(sim_time, ps_kxavg, c="b", label="avg. " + r"$P_s$ ")
        plt.plot(
            sim_time.data, pks.data * np.ones(np.size(ps_locsyn.data)), "--", c="k"
        )
        plt.xlabel(r"$t [c_s/a]$", fontsize=fsize)
        plt.title(
            r"$P_s(t) = \sum_{k_x, k_y} |\delta \hat{n}|^2 |F|^2 (t)$, $\omega_0 = $"
            + str(w0),
            fontsize=fsize,
        )
        plt.legend()
        plt.tick_params(labelsize=fsize)

        plt.subplot(1, 3, 3)
        plt.semilogy(
            phi2f_f2_locsyn.freq_time * 2 * np.pi,
            (phi2f_f2_locsyn),
            linestyle="-",
            lw=3,
            c="k",
            label=r"$ local \ k_x$",
        )
        plt.semilogy(
            phi2f_f2_locsyn.freq_time * 2 * np.pi,
            (phi2f_f2_kxavg),
            linestyle="-",
            lw=3,
            c="b",
            label=r"$avg. \ k_x$",
        )

        plt.legend()
        plt.title(r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=fsize)
        plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
        plt.tick_params(labelsize=fsize)

        plt.figure(40 + ik, figsize=(14, 7))
        plt.subplot(1, 3, 1)
        plt.plot(ky, Fy2_shift, ".-", c="k", lw=2, label="ch = " + str(ik) + " (shift)")
        plt.xlabel(r"$k_y\rho_s$", fontsize=fsize)
        plt.title("Local " + r"$k_x \ |F_y(k_y)|^2$", fontsize=fsize)
        plt.legend()
        plt.tick_params(labelsize=fsize)

        plt.subplot(1, 3, 2)
        label = "ch = " + str(ik)
        plt.semilogy(
            phi2f_f2_locsyn.freq_time * 2 * np.pi,
            (phi2f_f2_locsyn),
            linestyle="-",
            lw=2,
            c="k",
            label="ch = " + str(ik),
        )
        plt.legend()
        plt.title("Local " + r"$\tilde{P}_s(f)$, ch = " + str(ik), fontsize=fsize)
        plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
        plt.tick_params(labelsize=fsize)

        plt.subplot(1, 3, 3)
        plt.semilogy(
            phi2f_f2_locsyn.freq_time * 2 * np.pi,
            (phi2f_f2_kxavg),
            linestyle="-",
            lw=2,
            c="k",
            label=r"$avg. \ k_x \ (tot)$",
        )
        plt.legend()
        plt.title("Avg. " r"$-k_x \ \tilde{P}_s(f)$, ch = " + str(ik), fontsize=fsize)
        plt.xlabel(r"$\omega \ [a/c_s]$", **axis_font)
        plt.tick_params(labelsize=fsize)

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
            plt.savefig("ps_freq_ky_ch" + str(ik) + "_zon-nz.pdf")

        self.Pks[ik] = pks
        self.Sigma_ks_hann[ik] = sigma_ks_hann

        return [pkf, pkf_hann, pkf_kx0ky0, pks, sigma_ks_hann]

    def get_units_norms(self, pyro):
        import numpy as np

        a_minor = (1.0 * pyro.norms.lref).to(
            "m"
        )  # geo.a_minor   # pyro.eq.a_minor       # (1.0*pyro.norms.lref).to('m')
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
        fsize = self.fsize
        axis_font = {"fontname": "Arial", "size": str(fsize)}
        if_save = 0
        geo = self.geo

        fs = self.eq.flux_surface(psi_n=self.psin)
        fs.plot_path(x_label="", y_label="", color="k")
        plt.plot(
            self.geo.R * self.a_minor,
            self.geo.Z * self.a_minor,
            "-r",
            label="local_geometry",
        )
        plt.plot(fs["R"], fs["Z"], "--b", label="fs")
        plt.plot(Rtmp[tmp_ind], self.Zloc, "ok", markersize=12, label="sc loc")
        plt.axis("equal")
        plt.ylabel("R [m]")
        plt.xlabel("Z [m]")
        plt.legend()

        # plot (kx,ky) time averaged
        thetaplot = np.linspace(0, 2 * np.pi, 100)

        plt.figure(100)
        plt.contourf(self.ky, self.kx, np.log10(self.phi2kxky), levels=100)
        plt.xlabel(r"$k_y\rho_s$", fontsize=fsize)
        plt.ylabel(r"$k_x\rho_s$", fontsize=fsize)
        plt.title(
            r"$|\delta n_e(k_x, k_y)|^2$, "
            + r"$\theta$ ="
            + str(np.round(self.geo.theta[self.ith] * 180 / np.pi, 4))
            + r" $^o$, t = ["
            + str(np.round(self.t1 * self.time[-1], 4))
            + ", "
            + str(np.round(self.t2 * self.time[-2], 4))
            + "]",
            fontsize=fsize,
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
        plt.tick_params(labelsize=fsize)
        ax = plt.gca()

        msize = 12

        # plot synthetic P(k)
        plt.figure(37, figsize=(12, 5.5))
        plt.subplot(1, 2, 1)
        plt.title(r"$P_{syn}(k_y\rho_{s,unit})$ [a.u.]", fontsize=fsize)
        plt.semilogy(self.kx0, self.Pks, ".-b", label="pks")
        plt.legend()
        plt.xlabel(r"$k_x\rho_{s,unit}$", fontsize=fsize)
        plt.tick_params(labelsize=fsize)

        Kperp = np.sqrt(self.Kn0_exp**2 + self.Kb0_exp**2)

        plt.subplot(1, 2, 2)
        plt.title(r"$P_{syn}(k_\perp)$ [a.u.]", fontsize=fsize)
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
        plt.legend()
        plt.xlabel(r"$k_\perp [cm^{-1}]$", fontsize=fsize)
        plt.tick_params(labelsize=fsize)

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
