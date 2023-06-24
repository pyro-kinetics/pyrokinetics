import numpy as np
from scipy.interpolate import RectBivariateSpline
import xrft

from ..pyro import Pyro


class Diagnostics:
    """
    Contains all the diagnostics that can be applied to simulation output data.

    Currently, this class contains only the function to generate a Poincare map,
    but new diagnostics will be available in future.

    Please call "load_gk_output" before attempting to use any diagnostic

    Parameters
    ----------
    pyro: Pyro object containing simulation output data (and geometry)

    """

    def __init__(self, pyro: Pyro):
        if pyro.gk_output is None:
            raise RuntimeError(
                "Diagnostics: Please load gk output files (Pyro.load_gk_output)"
                " before using any diagnostic"
            )
        self.pyro = pyro

    def poincare(
        self,
        xarray: np.ndarray,
        yarray: np.ndarray,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
    ):
        """
        Generates a poincare map. It returns the (x, y) coordinates of
        the Poincare Map.

        This routine may take a while depending on ``nturns`` and on
        the number of magnetic field lines. The parameter rhostar is
        required by the flux-tube boundary condition.
        Available for CGYRO, GENE and GS2 nonlinear simulations

        You need to load the output files of a simulation
        berore calling this function.

        Parameters
        ----------
        xarray: numpy.ndarray, array containing x coordinate of initial
            field line positions
        yarray: numpy.ndarray, array containing y coordinate of initial
            field line positions
        nturns: int, number of intersection points
        time: float, time reference
        rhostar: float, rhostar is needed to set the boundary condition
                 on the magnetic field line
        use_invfft: bool, if True, the inverse Fourier transform is computed
                 every (x, y) points along the magnetic field line. It is much
                 more accurate but very slow.

        Returns
        -------
        coordinates: numpy.ndarray, 4D array of shape (2, nturns, len(yarray), len(xarray))
               containing the x and y coordinates shaped according to the initial
               field line position. See ``example_poincare.py`` for a simple example.

        Raises
        ------
        NotImplementedError: if `gk_code` is not `CGYRO`, `GENE` or `GS2`
        RuntimeError: in case of linear simulation
        """
        if self.pyro.gk_code not in ["CGYRO", "GS2", "GENE"]:
            raise NotImplementedError(
                "Poincare map only available for CGYRO, GENE and GS2"
            )
        if self.pyro.gk_input.is_linear():
            raise RuntimeError("Poincare only available for nonlinear runs")
        apar = self.pyro.gk_output.fields.sel(field="apar").sel(
            time=time, method="nearest"
        )
        kx = apar.kx.values
        ky = apar.ky.values
        ntheta = apar.theta.shape[0]
        nkx = kx.shape[0]
        nky = ky.shape[0]
        dkx = kx[1] - kx[0]
        dky = ky[1]
        ny = 2 * (nky - 1)
        nkx0 = nkx + 1 - np.mod(nkx, 2)
        Lx = 2 * np.pi / dkx
        Ly = 2 * np.pi / dky
        xgrid = np.linspace(-Lx / 2, Lx / 2, nkx0)[:nkx]
        ygrid = np.linspace(-Ly / 2, Ly / 2, ny)
        xmin = np.min(xgrid)
        ymin = np.min(ygrid)
        ymax = np.max(ygrid)

        # Geometrical factors
        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2 * np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)
        nskip = len(geo.theta) // ntheta
        bmag = np.sqrt((1 / geo.R) ** 2 + geo.b_poloidal**2)
        bmag = np.roll(bmag[::nskip], ntheta // 2)
        jacob = self.pyro.metric_terms.Jacobian * geo.dpsidr * geo.bunit_over_b0
        jacob = np.roll(jacob[::nskip], ntheta // 2)
        dq = rhostar * Lx * geo.shat / geo.dpsidr
        qmin = geo.q - dq / 2
        fac1 = 2 * np.pi * geo.dpsidr / rhostar
        fac2 = geo.dpsidr * geo.q / geo.rho

        # Compute bx and by
        ikxapar = 1j * apar.kx * apar
        ikyapar = -1j * apar.ky * apar

        ikxapar = ikxapar.transpose("kx", "ky", "theta")
        ikyapar = ikyapar.transpose("kx", "ky", "theta")

        if use_invfft:
            ikxapar = ikxapar.values
            ikyapar = ikyapar.values
        else:
            byfft = xrft.ifft(
                ikxapar * nkx * ny,
                dim=["kx", "ky"],
                real_dim="ky",
                lag=[0, 0],
                true_amplitude=False,
            )
            bxfft = xrft.ifft(
                ikyapar * nkx * ny,
                dim=["kx", "ky"],
                real_dim="ky",
                lag=[0, 0],
                true_amplitude=False,
            )

            By = [
                RectBivariateSpline(
                    xgrid,
                    ygrid,
                    byfft.sel(theta=theta, method="nearest"),
                    kx=5,
                    ky=5,
                    s=1,
                )
                for theta in byfft.theta
            ]
            Bx = [
                RectBivariateSpline(
                    xgrid,
                    ygrid,
                    bxfft.sel(theta=theta, method="nearest"),
                    kx=5,
                    ky=5,
                    s=1,
                )
                for theta in bxfft.theta
            ]

        # Main loop
        x = xarray[np.newaxis, :]
        y = yarray[:, np.newaxis]
        points = np.empty((2, nturns, len(yarray), len(xarray)))

        for iturn in range(nturns):
            for ith in range(0, ntheta - 1, 2):
                if use_invfft:
                    dby = (
                        self._invfft(ikxapar[:, :, ith], x, y, kx, ky)
                        * bmag[ith]
                        * fac2
                    )
                    dbx = (
                        self._invfft(ikyapar[:, :, ith], x, y, kx, ky)
                        * bmag[ith]
                        * fac2
                    )
                else:
                    dby = By[ith](x, y, grid=False) * bmag[ith] * fac2
                    dbx = Bx[ith](x, y, grid=False) * bmag[ith] * fac2

                xmid = x + 2 * np.pi / ntheta * dbx * jacob[ith]
                ymid = y + 2 * np.pi / ntheta * dby * jacob[ith]

                if use_invfft:
                    dby = (
                        self._invfft(ikxapar[:, :, ith + 1], xmid, ymid, kx, ky)
                        * bmag[ith + 1]
                        * fac2
                    )
                    dbx = (
                        self._invfft(ikyapar[:, :, ith + 1], xmid, ymid, kx, ky)
                        * bmag[ith + 1]
                        * fac2
                    )
                else:
                    dby = By[ith + 1](xmid, ymid, grid=False) * bmag[ith + 1] * fac2
                    dbx = Bx[ith + 1](xmid, ymid, grid=False) * bmag[ith + 1] * fac2

                x = x + 4 * np.pi / ntheta * dbx * jacob[ith + 1]
                y = y + 4 * np.pi / ntheta * dby * jacob[ith + 1]

                y = np.where(y < ymin, ymax - (ymin - y), y)

                y = np.where(y > ymax, ymin + (y - ymax), y)

            y = y + np.mod(fac1 * ((x - xmin) / Lx * dq + qmin), Ly)
            y = np.where(y > ymax, ymin + (y - ymax), y)

            points[0, iturn, :, :] = x
            points[1, iturn, :, :] = y

        return points

    def get_zonal_mixing(
        ky_mix,
        gamma_mix,
        **kw,
    ):
        """
        :param ky_mix: poloidal wavenumber [nk]

        :param gamma_mix: most unstable growth rates [nk]

        :param **kw: keyword list in input.tglf
        """

        nky = len(ky_mix)
        gammamax1 = gamma_mix[0]
        kymax1 = ky_mix[0]
        testmax1 = gammamax1 / kymax1
        testmax2 = 0
        jmax1 = 0
        jmax2 = 0

        j1 = 0
        j2 = 0

        kycut = (
            0.8 * abs(kw["ZS_2"]) / np.sqrt(kw["TAUS_2"] * kw["MASS_2"])
        )  # ITG/ETG-scale separation (for TEM scales see [Creely et al., PPCF, 2019])
        kyhigh = 0.15 * abs(kw["ZS_1"]) / np.sqrt(kw["TAUS_1"] * kw["MASS_1"])

        for j in range(1, nky):
            ky0 = ky_mix[j]
            if ky0 < kycut:
                j1 = j1 + 1
            if ky0 < kyhigh:
                j2 = j2 + 1
            test = gamma_mix[j] / ky0
            if ky0 < kycut:
                if test > testmax1:
                    testmax1 = test
                    kymax1 = ky0
                    jmax1 = j
            if ky0 > kycut:
                if test > testmax2:
                    testmax2 = test
                    kymax2 = ky0
                    jmax2 = j
        # handle exceptions
        if j1 == nky - 1:
            j1 = nky - 2
        if jmax2 == 0:
            jmax2 = j2

        gammamax2 = gamma_mix[jmax2]
        kymax2 = ky_mix[jmax2]
        gammamax1 = gamma_mix[jmax1]
        kymax1 = ky_mix[jmax1]
        vzf1 = gammamax1 / kymax1
        vzf2 = gammamax2 / kymax2

        # Routine for better determination of gamma/ky peak
        if jmax1 > 0 and jmax1 < j1:
            f0 = gamma_mix[jmax1 - 1] / ky_mix[jmax1 - 1]
            f1 = gamma_mix[jmax1] / ky_mix[jmax1]
            f2 = gamma_mix[jmax1 + 1] / ky_mix[jmax1 + 1]
            dky = ky_mix[jmax1 + 1] - ky_mix[jmax1 - 1]
            x0 = (ky_mix[jmax1] - ky_mix[jmax1 - 1]) / dky
            a = f0
            x02 = x0**2
            b = (f1 - f0 * (1 - x02) - f2 * x02) / (x0 - x02)
            c = f2 - f0 - b
            xmax = -b / (2.0 * c)
            if xmax > 1.0:
                kymax1 = ky_mix[jmax1 + 1]
                gammamax1 = f2 * kymax1
            elif xmax < 0.0:
                kymax1 = ky_mix[jmax1 - 1]
                gammamax1 = f0 * kymax1
            else:
                kymax1 = ky_mix[jmax1 - 1] + dky * xmax
                gammamax1 = (a + b * xmax + c * xmax**2) * kymax1
        vzf_mix = gammamax1 / kymax1
        kymax_mix = kymax1
        jmax_mix = jmax1
        return vzf_mix, kymax_mix, jmax_mix

    def get_sat_params(
        sat_rule_in, ky, gammas, mts=5.0, ms=128, small=0.00000001, **kw
    ):
        """
        This function calculates the scalar saturation parameters and spectral shift needed
        for the TGLF saturation rules, dependent on changes to 'tglf_geometry.f90' by Gary Staebler

        :mts: the number of points in the s-grid (flux surface contour)
        :ms: number of points along the arclength
        :ds: the arc length differential on a flux surface
        :R(ms): the major radius on the s-grid
        :Z(ms): the vertical coordinate on the s-grid
        :Bp(ms): the poloidal magnetic field on the s-grid normalized to B_unit
        :**kw: input.tglf
        """
        drmajdx_loc = kw["DRMAJDX_LOC"]
        drmindx_loc = kw["DRMINDX_LOC"]
        kappa_loc = kw["KAPPA_LOC"]
        s_kappa_loc = kw["S_KAPPA_LOC"]
        rmin_loc = kw["RMIN_LOC"]
        rmaj_loc = kw["RMAJ_LOC"]
        zeta_loc = kw["ZETA_LOC"]
        q_s = kw["Q_LOC"]
        q_prime_s = kw["Q_PRIME_LOC"]
        p_prime_s = kw["P_PRIME_LOC"]
        delta_loc = kw["DELTA_LOC"]
        s_delta_loc = kw["S_DELTA_LOC"]
        s_zeta_loc = kw["S_ZETA_LOC"]
        alpha_e_in = kw["ALPHA_E"]
        vexb_shear = kw["VEXB_SHEAR"]
        sign_IT = kw["SIGN_IT"]
        sign_Bt_in = kw["SIGN_BT"]
        units = kw["UNITS"]
        mass_2 = kw["MASS_2"]
        taus_2 = kw["TAUS_2"]
        zs_2 = kw["ZS_2"]
        alpha_ZF_in = kw["ALPHA_ZF"]

        zmaj_loc = 0.0
        dzmajdx_loc = 0.0
        norm_ave = 0.0
        SAT_geo1_out = 0.0
        SAT_geo2_out = 0.0
        dlp = 0.0

        R = np.zeros(ms + 1)
        Z = np.zeros(ms + 1)
        Bp = np.zeros(ms + 1)
        Bt = np.zeros(ms + 1)
        B = np.zeros(ms + 1)
        b_geo = np.zeros(ms + 1)
        qrat_geo = np.zeros(ms + 1)
        sin_u = np.zeros(ms + 1)
        s_p = np.zeros(ms + 1)
        r_curv = np.zeros(ms + 1)
        psi_x = np.zeros(ms + 1)
        costheta_geo = np.zeros(ms + 1)

        pi_2 = 2 * np.pi
        if rmin_loc < 0.00001:
            rmin_loc = 0.00001
        vs_2 = np.sqrt(taus_2 / mass_2)
        gamma_reference_kx0 = gammas[0, :]

        # Miller geo
        rmin_s = rmin_loc
        Rmaj_s = rmaj_loc

        # compute the arclength around the flux surface:
        # initial values define dtheta
        theta = 0.0
        x_delta = np.arcsin(delta_loc)
        arg_r = theta + x_delta * np.sin(theta)
        darg_r = 1.0 + x_delta * np.cos(theta)
        arg_z = theta + zeta_loc * np.sin(2.0 * theta)
        darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
        r_t = -rmin_loc * np.sin(arg_r) * darg_r
        z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
        l_t = np.sqrt(r_t**2 + z_t**2)

        # scale dtheta by l_t to keep mts points in each ds interval of size pi_2/ms
        dtheta = pi_2 / (mts * ms * l_t)
        l_t1 = l_t
        arclength = 0.0

        while theta < pi_2:
            theta = theta + dtheta
            if theta > pi_2:
                theta = theta - dtheta
                dtheta = pi_2 - theta
                theta = pi_2

            arg_r = theta + x_delta * np.sin(theta)
            darg_r = 1.0 + x_delta * np.cos(theta)  # d(arg_r)/dtheta
            r_t = -rmin_loc * np.sin(arg_r) * darg_r  # dR/dtheta
            arg_z = theta + zeta_loc * np.sin(2.0 * theta)
            darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)  # d(arg_z)/dtheta
            z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z  # dZ/dtheta
            l_t = np.sqrt(r_t**2 + z_t**2)  # dl/dtheta
            arclength = (
                arclength + 0.50 * (l_t + l_t1) * dtheta
            )  # arclength along flux surface in poloidal direction
            l_t1 = l_t

        # Find the theta points which map to an equally spaced s-grid of ms points along the arclength
        # going clockwise from the outboard midplane around the flux surface
        # by searching for the theta where dR**2 + dZ**2 >= ds**2 for a centered difference df=f(m+1)-f(m-1).
        # This keeps the finite difference error of dR/ds, dZ/ds on the s-grid small
        ds = arclength / ms
        t_s = np.zeros(ms + 1)
        t_s[ms] = -pi_2

        # Make a first guess based on theta = 0.0
        theta = 0.0
        arg_r = theta + x_delta * np.sin(theta)
        darg_r = 1.0 + x_delta * np.cos(theta)
        arg_z = theta + zeta_loc * np.sin(2.0 * theta)
        darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
        r_t = -rmin_loc * np.sin(arg_r) * darg_r
        z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
        l_t = np.sqrt(r_t**2 + z_t**2)
        dtheta = -ds / l_t
        theta = dtheta
        l_t1 = l_t

        for m in range(1, int(ms / 2) + 1):
            arg_r = theta + x_delta * np.sin(theta)
            darg_r = 1.0 + x_delta * np.cos(theta)
            arg_z = theta + zeta_loc * np.sin(2.0 * theta)
            darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)
            r_t = -rmin_loc * np.sin(arg_r) * darg_r
            z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z
            l_t = np.sqrt(r_t**2 + z_t**2)
            dtheta = -ds / (0.5 * (l_t + l_t1))
            t_s[m] = t_s[m - 1] + dtheta
            theta = t_s[m] + dtheta
            l_t1 = l_t

        # distribute endpoint error over interior points
        dtheta = (t_s[int(ms / 2)] - (-np.pi)) / (ms / 2)

        for m in range(1, int(ms / 2) + 1):
            t_s[m] = t_s[m] - (m) * dtheta
            t_s[ms - m] = -pi_2 - t_s[m]
        # Quinn additions,
        B_unit_out = np.zeros(ms + 1)
        grad_r_out = np.zeros(ms + 1)
        for m in range(0, ms + 1):
            theta = t_s[m]
            arg_r = theta + x_delta * np.sin(theta)
            darg_r = 1.0 + x_delta * np.cos(theta)
            arg_z = theta + zeta_loc * np.sin(2.0 * theta)
            darg_z = 1.0 + zeta_loc * 2.0 * np.cos(2.0 * theta)

            R[m] = rmaj_loc + rmin_loc * np.cos(arg_r)  # = R(theta)
            Z[m] = zmaj_loc + kappa_loc * rmin_loc * np.sin(arg_z)  # = Z(theta)

            R_t = -rmin_loc * np.sin(arg_r) * darg_r  # = dR/dtheta
            Z_t = kappa_loc * rmin_loc * np.cos(arg_z) * darg_z  # = dZ/dtheta

            l_t = np.sqrt(R_t**2 + Z_t**2)  # = dl/dtheta

            R_r = (
                drmajdx_loc
                + drmindx_loc * np.cos(arg_r)
                - np.sin(arg_r)
                * s_delta_loc
                * np.sin(theta)
                / np.sqrt(1.0 - delta_loc**2)
            )  # = dR/dr
            Z_r = (
                dzmajdx_loc
                + kappa_loc * np.sin(arg_z) * (drmindx_loc + s_kappa_loc)
                + kappa_loc * np.cos(arg_z) * s_zeta_loc * np.sin(2.0 * theta)
            )

            det = R_r * Z_t - R_t * Z_r  # Jacobian
            grad_r = abs(l_t / det)
            if m == 0:
                B_unit = 1.0 / grad_r  # B_unit choosen to make qrat_geo(0)/b_geo(0)=1.0
                if drmindx_loc == 1.0:
                    B_unit = 1.0  # Waltz-Miller convention
            B_unit_out[m] = B_unit
            grad_r_out[m] = grad_r

            Bp[m] = (rmin_s / (q_s * R[m])) * grad_r * B_unit
            p_prime_s = p_prime_s * B_unit
            q_prime_s = q_prime_s / B_unit
            psi_x[m] = R[m] * Bp[m]

        delta_s = 12.0 * ds
        ds2 = 12.0 * ds**2
        for m in range(0, ms + 1):
            m1 = (ms + m - 2) % ms
            m2 = (ms + m - 1) % ms
            m3 = (m + 1) % ms
            m4 = (m + 2) % ms
            R_s = (R[m1] - 8.0 * R[m2] + 8.0 * R[m3] - R[m4]) / delta_s
            Z_s = (Z[m1] - 8.0 * Z[m2] + 8.0 * Z[m3] - Z[m4]) / delta_s
            s_p[m] = np.sqrt(R_s**2 + Z_s**2)
            R_ss = (-R[m1] + 16.0 * R[m2] - 30.0 * R[m] + 16.0 * R[m3] - R[m4]) / ds2
            Z_ss = (-Z[m1] + 16.0 * Z[m2] - 30.0 * Z[m] + 16.0 * Z[m3] - Z[m4]) / ds2
            r_curv[m] = (s_p[m] ** 3) / (R_s * Z_ss - Z_s * R_ss)
            sin_u[m] = -Z_s / s_p[m]

        # Compute f=R*Bt such that the eikonal S which solves
        # B*Grad(S)=0 has the correct quasi-periodicity S(s+Ls)=S(s)-2*pi*q_s, where Ls = arclength
        f = 0.0
        for m in range(1, ms + 1):
            f = f + 0.5 * ds * (
                s_p[m - 1] / (R[m - 1] * psi_x[m - 1]) + s_p[m] / (R[m] * psi_x[m])
            )
        f = pi_2 * q_s / f

        for m in range(0, ms + 1):
            Bt[m] = f / R[m]
            B[m] = np.sqrt(Bt[m] ** 2 + Bp[m] ** 2)
            qrat_geo[m] = (rmin_s / R[m]) * (B[m] / Bp[m]) / q_s
            b_geo[m] = B[m]
            costheta_geo[m] = (
                -Rmaj_s
                * (Bp[m] / (B[m]) ** 2)
                * (Bp[m] / r_curv[m] - (f**2 / (Bp[m] * (R[m]) ** 3)) * sin_u[m])
            )

        qrat_geo0 = (rmin_s / R[0]) * (B[0] / Bp[0]) / q_s
        b_geo0 = B[0]

        for m in range(1, ms + 1):
            dlp = s_p[m] * ds * (0.5 / Bp[m] + 0.5 / Bp[m - 1])
            norm_ave += dlp
            SAT_geo1_out += (
                dlp
                * ((b_geo[0] / b_geo[m - 1]) ** 4 + (b_geo[0] / b_geo[m]) ** 4)
                / 2.0
            )
            SAT_geo2_out += (
                dlp
                * (
                    (qrat_geo[0] / qrat_geo[m - 1]) ** 4
                    + (qrat_geo[0] / qrat_geo[m]) ** 4
                )
                / 2.0
            )

        SAT_geo1_out = SAT_geo1_out / norm_ave
        SAT_geo2_out = SAT_geo2_out / norm_ave
        R_unit = Rmaj_s * b_geo[0] / (qrat_geo[0] * costheta_geo[0])
        B_geo0_out = b_geo[0]
        Bt0_out = f / Rmaj_s
        grad_r0_out = b_geo[0] / qrat_geo[0]
        # Additional outputs for SAT2 G1(theta), Gq(theta)
        theta_out = t_s  # theta grid over which everything is calculated.
        Bt_out = B  # total magnetic field matching theta_out grid.

        # Compute spetral shift kx0_e
        vexb_shear_s = vexb_shear * sign_IT
        vexb_shear_kx0 = alpha_e_in * vexb_shear_s

        wd0 = abs(ky / Rmaj_s)
        kx0_factor = abs(b_geo[0] / qrat_geo[0] ** 2)

        if alpha_ZF_in < 0.0:
            kx0_factor = 1.0
        kx0_factor = 1.0 + 0.40 * (kx0_factor - 1.0) ** 2

        kyi = ky * vs_2 * mass_2 / abs(zs_2)
        wE = (
            kx0_factor
            * array([min(x / 0.3, 1.0) for x in kyi])
            * vexb_shear_kx0
            / gamma_reference_kx0
        )
        kx0_e = -(
            0.36 * vexb_shear_kx0 / gamma_reference_kx0
            + 0.38 * wE * np.tanh((0.69 * wE) ** 6)
        )

        if sat_rule_in == 1:
            if units == "CGYRO":
                wE = 0.0
                kx0_factor = 1.0
            kx0_e = -(
                0.53 * vexb_shear_kx0 / gamma_reference_kx0
                + 0.25 * wE * np.tanh((0.69 * wE) ** 6)
            )
        elif sat_rule_in == 2 or sat_rule_in == 3:
            vzf_out, kymax_out, jmax_out = get_zonal_mixing(
                ky, gamma_reference_kx0, **kw
            )
            if abs(kymax_out * vzf_out * vexb_shear_kx0) > small:
                kx0_e = (
                    -0.32 * ((ky / kymax_out) ** 0.3) * vexb_shear_kx0 / (ky * vzf_out)
                )
            else:
                kx0_e = np.zeros(len(ky))
        a0 = 1.3
        if sat_rule_in == 1:
            a0 = 1.45
        elif sat_rule_in == 2 or sat_rule_in == 3:
            a0 = 1.6
        kx0_e = array([min(abs(x), a0) * x / abs(x) for x in kx0_e])
        kx0_e[np.isnan(kx0_e)] = 0

        return (
            kx0_e,
            SAT_geo1_out,
            SAT_geo2_out,
            R_unit,
            Bt0_out,
            B_geo0_out,
            grad_r0_out,
            theta_out,
            Bt_out,
            grad_r_out,
            B_unit_out,
        )

    def mode_transition_function(x, y1, y2, x_ITG, x_TEM):
        if x < x_ITG:
            y = y1
        elif x > x_TEM:
            y = y2
        else:
            y = y1 * ((x_TEM - x) / (x_TEM - x_ITG)) + y2 * (
                (x - x_ITG) / (x_TEM - x_ITG)
            )
        return y

    def linear_interpolation(x, y, x0):
        i = 0
        while x[i] < x0:
            i += 1
        y0 = ((y[i] - y[i - 1]) * x0 + (x[i] * y[i - 1] - x[i - 1] * y[i])) / (
            x[i] - x[i - 1]
        )
        return y0

    def intensity_sat(
        sat_rule_in,
        ky_spect,
        gp,
        kx0_e,
        nmodes,
        QL_data,
        expsub=2.0,
        alpha_zf_in=1.0,
        kx_geo0_out=1.0,
        SAT_geo_out=1.0,
        bz1=0.0,
        bz2=0.0,
        return_phi_params=False,
        **kw,
    ):
        """
        TGLF SAT1 from [Staebler et al., 2016, PoP], takes both GYRO and TGLF outputs as inputs

        :param sat_rule_in: saturation rule [1, 2, 3]

        :param ky_spect: poloidal wavenumber [nk]

        :param gp: growth rates [nk, nm]

        :param kx0_e: spectral shift of the radial wavenumber due to VEXB_SHEAR [nk]

        :param nmodes_in: number of modes stored in quasi-linear weights [1, ..., 5]

        :param QL_data: Quasi-linear weights [ky, nm, ns, nf, type (i.e. particle,energy,stress_tor,stress_para,exchange)]

        :param expsub: scalar exponent in gammaeff calculation [2.0]

        :param alpha_zf_in: scalar switch for the zonal flow coupling coefficient [1.0]

        :param kx_geo_out: scalar switch for geometry [1.0]

        :param SAT_geo_out: scalar switch for geoemtry [1.0]

        :param bz1: scalar correction to zonal flow mixing term [0.0]

        :param bz2: scalar correction to zonal flow mixing term [0.0]

        :param return_phi_params: bool, option to return parameters for calculing the SAT1, SAT2 model for phi [False]

        :param **kw: keyword list in input.tglf
        """

        nky = len(ky_spect)
        if len(np.shape(gp)) > 1:
            gammas1 = gp[
                :, 0
            ]  # SAT1 and SAT2 use the growth rates of the most unstable modes
        else:
            gammas1 = gp
        gamma_net = np.zeros(nky)

        if sat_rule_in == 1:
            etg_streamer = 1.05
            kyetg = (
                etg_streamer * abs(kw["ZS_2"]) / np.sqrt(kw["TAUS_2"] * kw["MASS_2"])
            )
            measure = np.sqrt(kw["TAUS_1"] * kw["MASS_2"])

        czf = abs(alpha_zf_in)
        small = 1.0e-10
        cz1 = 0.48 * czf
        cz2 = 1.0 * czf
        cky = 3.0
        sqcky = np.sqrt(cky)
        cnorm = 14.29

        kycut = (
            0.8 * abs(kw["ZS_2"]) / np.sqrt(kw["TAUS_2"] * kw["MASS_2"])
        )  # ITG/ETG-scale separation (for TEM scales see [Creely et al., PPCF, 2019])
        kyhigh = 0.15 * abs(kw["ZS_1"]) / np.sqrt(kw["TAUS_1"] * kw["MASS_1"])

        vzf_out, kymax_out, jmax_out = get_zonal_mixing(ky_spect, gammas1, **kw)

        if kw["RLNP_CUTOFF"] > 0.0:
            ptot = 0
            dlnpdr = 0
            for i in range(1, kw["NS"] + 1, 1):
                ptot += kw["AS_%s" % i] * kw["TAUS_%s" % i]  # only kinetic species
                dlnpdr += (
                    kw["AS_%s" % i]
                    * kw["TAUS_%s" % i]
                    * (kw["RLNS_%s" % i] + kw["RLTS_%s" % i])
                )
            dlnpdr = kw["RMAJ_LOC"] * dlnpdr / max(ptot, 0.01)

            if dlnpdr >= kw["RLNP_CUTOFF"]:
                dlnpdr = kw["RLNP_CUTOFF"]
            if dlnpdr < 4.0:
                dlnpdr = 4.0
        else:
            dlnpdr = 12.0

        if sat_rule_in == 2 or sat_rule_in == 3:
            # SAT2 fit for CGYRO linear modes NF 2021 paper
            b0 = 0.76
            b1 = 1.22
            b2 = 3.74
            if nmodes > 1:
                b2 = 3.55
            b3 = 1.0
            d1 = (kw["Bt0_out"] / kw["B_geo0_out"]) ** 4  # PPCF paper 2020
            d1 = d1 / kw["grad_r0_out"]
            # WARNING: this is correct, but it's the reciprocal in the paper (typo in paper)
            Gq = kw["B_geo0_out"] / kw["grad_r0_out"]
            d2 = b3 / Gq**2
            cnorm = b2 * (12.0 / dlnpdr)
            kyetg = 1000.0  # does not impact SAT2
            cky = 3.0
            sqcky = np.sqrt(cky)
            kycut = b0 * kymax_out
            cz1 = 0.0
            cz2 = 1.05 * czf
            measure = 1.0 / kymax_out

        if sat_rule_in == 3:
            kmax = kymax_out
            gmax = vzf_out * kymax_out
            kmin = 0.685 * kmax
            aoverb = -1.0 / (2 * kmin)
            coverb = -0.751 * kmax
            kT = 1.0 / kw["rho_ion"]  # SAT3 used up to ky rho_av = 1.0, then SAT2
            k0 = 0.6 * kmin
            kP = 2.0 * kmin
            c_1 = -2.42
            x_ITG = 0.8
            x_TEM = 1.0
            Y_ITG = 3.3 * (gmax**2) / (kmax**5)
            Y_TEM = 12.7 * (gmax**2) / (kmax**4)
            scal = 0.82  # Q(SAT3 GA D) / (2 * QLA(ITG,Q) * Q(SAT2 GA D))

            Ys = np.zeros(nmodes)
            xs = np.zeros(nmodes)

            for k in range(1, nmodes + 1):
                sum_W_i = 0

                # sum over ion species, requires electrons to be species 1
                for is_ in range(2, np.shape(QL_data)[2] + 1):
                    sum_W_i += QL_data[:, k - 1, is_ - 1, 0, 1]

                # check for singularities in weight ratio near kmax
                i = 1
                while ky_spect[i - 1] < kmax:
                    i += 1

                if sum_W_i[i - 1] == 0.0 or sum_W_i[i - 2] == 0.0:
                    x = 0.5
                else:
                    abs_W_ratio = np.abs(QL_data[:, k - 1, 0, 0, 1] / sum_W_i)
                    abs_W_ratio = np.nan_to_num(abs_W_ratio)
                    x = linear_interpolation(ky_spect, abs_W_ratio, kmax)

                xs[k - 1] = x
                Y = mode_transition_function(x, Y_ITG, Y_TEM, x_ITG, x_TEM)
                Ys[k - 1] = Y

        ax = 0.0
        ay = 0.0
        exp_ax = 1
        if kw["ALPHA_QUENCH"] == 0.0:
            if sat_rule_in == 1:
                # spectral shift model parameters
                ax = 1.15
                ay = 0.56
                exp_ax = 4
                units_in = kw["UNITS"]
            elif sat_rule_in == 2 or sat_rule_in == 3:
                ax = 1.21
                ay = 1.0
                exp_ax = 2
                units_in = "CGYRO"

        for j in range(0, nky):
            kx = kx0_e[j]
            if sat_rule_in == 2:
                ky0 = ky_spect[j]
                if ky0 < kycut:
                    kx_width = kycut / kw["grad_r0_out"]
                else:
                    kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
                kx = kx * ky0 / kx_width
            gamma_net[j] = gammas1[j] / (1.0 + abs(ax * kx) ** exp_ax)

        if sat_rule_in == 1:
            vzf_out, kymax_out, jmax_out = get_zonal_mixing(ky_spect, gamma_net, **kw)
        else:
            vzf_out_fp = vzf_out
            vzf_out = vzf_out * gamma_net[jmax_out] / max(gammas1[jmax_out], small)

        gammamax1 = vzf_out * kymax_out
        kymax1 = kymax_out
        jmax1 = jmax_out
        vzf1 = vzf_out

        # include zonal flow effects on growth rate model:
        gamma_mix1 = np.zeros(nky)
        gamma = np.zeros(nky)

        for j in range(0, nky):
            gamma0 = gamma_net[j]
            ky0 = ky_spect[j]
            if sat_rule_in == 1:
                if ky0 < kymax1:
                    gamma[j] = max(gamma0 - cz1 * (kymax1 - ky0) * vzf1, 0.0)
                else:
                    gamma[j] = cz2 * gammamax1 + max(gamma0 - cz2 * vzf1 * ky0, 0.0)
            elif sat_rule_in == 2 or sat_rule_in == 3:
                if ky0 < kymax1:
                    gamma[j] = gamma0
                else:
                    gamma[j] = gammamax1 + max(gamma0 - cz2 * vzf1 * ky0, 0.0)
            gamma_mix1[j] = gamma[j]

        # Mix over ky>kymax with integration weight
        mixnorm1 = np.zeros(nky)
        for j in range(jmax1 + 2, nky):
            gamma_ave = 0.0
            mixnorm1 = ky_spect[j] * (
                np.arctan(sqcky * (ky_spect[nky - 1] / ky_spect[j] - 1.0))
                - np.arctan(sqcky * (ky_spect[jmax1 + 1] / ky_spect[j] - 1.0))
            )
            for i in range(jmax1 + 1, nky - 1):
                ky_1 = ky_spect[i]
                ky_2 = ky_spect[i + 1]
                mix1 = ky_spect[j] * (
                    np.arctan(sqcky * (ky_2 / ky_spect[j] - 1.0))
                    - np.arctan(sqcky * (ky_1 / ky_spect[j] - 1.0))
                )
                delta = (gamma[i + 1] - gamma[i]) / (ky_2 - ky_1)
                mix2 = ky_spect[j] * mix1 + (
                    ky_spect[j] * ky_spect[j] / (2.0 * sqcky)
                ) * (
                    np.log(cky * (ky_2 - ky_spect[j]) ** 2 + ky_spect[j] ** 2)
                    - np.log(cky * (ky_1 - ky_spect[j]) ** 2 + ky_spect[j] ** 2)
                )
                gamma_ave = gamma_ave + (gamma[i] - ky_1 * delta) * mix1 + delta * mix2
            gamma_mix1[j] = gamma_ave / mixnorm1

        if sat_rule_in == 3:
            gamma_fp = np.zeros_like(ky_spect)  # Assuming ky_spect is a numpy array
            gamma = np.zeros_like(ky_spect)  # Assuming ky_spect is a numpy array

            for j in range(1, nky + 1):
                gamma0 = gammas1[j - 1]
                ky0 = ky_spect[j - 1]

                if ky0 < kymax1:
                    gamma[j - 1] = gamma0
                else:
                    gamma[j - 1] = (gammamax1 * (vzf_out_fp / vzf_out)) + max(
                        gamma0 - cz2 * vzf_out_fp * ky0, 0.0
                    )

                gamma_fp[j - 1] = gamma[j - 1]

            # USE_MIX is true by default
            for j in range(
                jmax1 + 3, nky + 1
            ):  # careful: I'm switching here to Fortran indexing, but found jmax1 using python indexing
                gamma_ave = 0.0
                ky0 = ky_spect[j - 1]
                kx = kx0_e[j - 1]

                mixnorm = ky0 * (
                    np.arctan(sqcky * (ky_spect[nky - 1] / ky0 - 1.0))
                    - np.arctan(sqcky * (ky_spect[jmax1 + 1] / ky0 - 1.0))
                )

                for i in range(
                    jmax1 + 2, nky
                ):  # careful: I'm switching here to Fortran indexing, but found jmax1 using python indexing
                    ky1 = ky_spect[i - 1]
                    ky2 = ky_spect[i]
                    mix1 = ky0 * (
                        np.arctan(sqcky * (ky2 / ky0 - 1.0))
                        - np.arctan(sqcky * (ky1 / ky0 - 1.0))
                    )
                    delta = (gamma[i] - gamma[i - 1]) / (ky2 - ky1)
                    mix2 = ky0 * mix1 + (ky0 * ky0 / (2.0 * sqcky)) * (
                        np.log(cky * (ky2 - ky0) ** 2 + ky0**2)
                        - np.log(cky * (ky1 - ky0) ** 2 + ky0**2)
                    )
                    gamma_ave += (gamma[i - 1] - ky1 * delta) * mix1 + delta * mix2
                gamma_fp[j - 1] = gamma_ave / mixnorm

        if sat_rule_in == 3:
            if ky_spect[-1] >= kT:
                dummy_interp = np.zeros_like(ky_spect)
                k = 0
                while ky_spect[k] < kT:
                    k += 1

                for l in range(k - 1, k + 1):
                    gamma0 = gp[l, 0]
                    ky0 = ky_spect[l]
                    kx = kx0_e[l]

                    if ky0 < kycut:
                        kx_width = kycut / kw["grad_r0_out"]
                        sat_geo_factor = kw["SAT_geo0_out"] * d1 * SAT_geo1_out
                    else:
                        kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
                        sat_geo_factor = (
                            kw["SAT_geo0_out"]
                            * (
                                d1 * kw["SAT_geo1_out"] * kycut
                                + (ky0 - kycut) * d2 * kw["SAT_geo2_out"]
                            )
                            / ky0
                        )

                    kx = kx * ky0 / kx_width
                    gammaeff = 0.0
                    if gamma0 > small:
                        gammaeff = gamma_fp[l]
                    # potentials without multimode and ExB effects, added later
                    dummy_interp[l] = (
                        scal * measure * cnorm * (gammaeff / (kx_width * ky0)) ** 2
                    )
                    if units_in != "GYRO":
                        dummy_interp[l] = sat_geo_factor * dummy_interp[l]
                YT = linear_interpolation(ky_spect, dummy_interp, kT)
                YTs = np.array([YT] * nmodes)
            else:
                if (
                    aoverb * (kP**2)
                    + kP
                    + coverb
                    - ((kP - kT) * (2 * aoverb * kP + 1))
                    == 0
                ):
                    YTs = np.zeros(nmodes)
                else:
                    YTs = np.zeros(nmodes)
                    for l in range(1, nmodes + 1):
                        YTs[l - 1] = Ys[l - 1] * (
                            (
                                (aoverb * (k0**2) + k0 + coverb)
                                / (
                                    aoverb * (kP**2)
                                    + kP
                                    + coverb
                                    - ((kP - kT) * (2 * aoverb * kP + 1))
                                )
                            )
                            ** abs(c_1)
                        )

        # preallocate [nky] arrays for phi_params
        gammaeff_out = np.zeros((nky, nmodes))
        kx_width_out = np.zeros(nky)
        sat_geo_factor_out = np.zeros(nky)
        # intensity
        field_spectrum_out = np.zeros((nky, nmodes))
        for j in range(0, nky):
            gamma0 = gp[j, 0]
            ky0 = ky_spect[j]
            kx = kx0_e[j]
            if sat_rule_in == 1:
                sat_geo_factor = kw["SAT_geo0_out"]
                kx_width = ky0
            if sat_rule_in == 2 or sat_rule_in == 3:
                if ky0 < kycut:
                    kx_width = kycut / kw["grad_r0_out"]
                    sat_geo_factor = kw["SAT_geo0_out"] * d1 * kw["SAT_geo1_out"]
                else:
                    kx_width = kycut / kw["grad_r0_out"] + b1 * (ky0 - kycut) * Gq
                    sat_geo_factor = (
                        kw["SAT_geo0_out"]
                        * (
                            d1 * kw["SAT_geo1_out"] * kycut
                            + (ky0 - kycut) * d2 * kw["SAT_geo2_out"]
                        )
                        / ky0
                    )
                kx = kx * ky0 / kx_width

            if sat_rule_in == 1 or sat_rule_in == 2:
                for i in range(0, nmodes):
                    gammaeff = 0.0
                    if gamma0 > small:
                        gammaeff = gamma_mix1[j] * (gp[j, i] / gamma0) ** expsub
                    if ky0 > kyetg:
                        gammaeff = gammaeff * np.sqrt(ky0 / kyetg)
                    field_spectrum_out[j, i] = (
                        measure
                        * cnorm
                        * ((gammaeff / (kx_width * ky0)) / (1.0 + ay * kx**2)) ** 2
                    )
                    if "GYRO" not in units_in:
                        field_spectrum_out[j, i] = (
                            sat_geo_factor * field_spectrum_out[j, i]
                        )
                    # add these outputs
                    gammaeff_out[j, i] = gammaeff
                kx_width_out[j] = kx_width
                sat_geo_factor_out[j] = sat_geo_factor

            elif sat_rule_in == 3:
                # First part
                if gamma_fp[j] == 0:
                    Fky = 0.0
                else:
                    Fky = (gamma_mix1[j] / gamma_fp[j]) ** 2 / (
                        1.0 + ay * (kx**2)
                    ) ** 2
                for i in range(1, nmodes + 1):
                    field_spectrum_out[j, i - 1] = 0.0
                    if gamma0 > small:
                        if ky0 <= kP:  # initial quadratic
                            sig_ratio = (aoverb * (ky0**2) + ky0 + coverb) / (
                                aoverb * (k0**2) + k0 + coverb
                            )
                            field_spectrum_out[j, i - 1] = (
                                Ys[i - 1]
                                * (sig_ratio**c_1)
                                * Fky
                                * (gp[j, i - 1] / gamma0) ** (2 * expsub)
                            )
                        elif ky0 <= kT:  # connecting quadratic
                            if YTs[i - 1] == 0.0 or kP == kT:
                                field_spectrum_out[j, i - 1] = 0.0
                            else:
                                doversig0 = (
                                    (Ys[i - 1] / YTs[i - 1]) ** (1.0 / abs(c_1))
                                ) - (
                                    (
                                        aoverb * (kP**2)
                                        + kP
                                        + coverb
                                        - ((kP - kT) * (2 * aoverb * kP + 1))
                                    )
                                    / (aoverb * (k0**2) + k0 + coverb)
                                )
                                doversig0 = doversig0 * (1.0 / ((kP - kT) ** 2))
                                eoversig0 = -2 * doversig0 * kP + (
                                    (2 * aoverb * kP + 1)
                                    / (aoverb * (k0**2) + k0 + coverb)
                                )
                                foversig0 = (
                                    ((Ys[i - 1] / YTs[i - 1]) ** (1.0 / abs(c_1)))
                                    - eoversig0 * kT
                                    - doversig0 * (kT**2)
                                )
                                sig_ratio = (
                                    doversig0 * (ky0**2) + eoversig0 * ky0 + foversig0
                                )
                                field_spectrum_out[j, i - 1] = (
                                    Ys[i - 1]
                                    * (sig_ratio**c_1)
                                    * Fky
                                    * (gp[j, i - 1] / gamma0) ** (2 * expsub)
                                )
                        else:  # SAT2 for electron scale
                            gammaeff = gamma_mix1[j] * (gp[j, i - 1] / gamma0) ** expsub
                            if ky0 > kyetg:
                                gammaeff = gammaeff * np.sqrt(ky0 / kyetg)
                            field_spectrum_out[j, i - 1] = (
                                scal
                                * measure
                                * cnorm
                                * ((gammaeff / (kx_width * ky0)) / (1.0 + ay * kx**2))
                                ** 2
                            )
                            if units_in != "GYRO":
                                field_spectrum_out[j, i - 1] = (
                                    sat_geo_factor * field_spectrum_out[j, i - 1]
                                )
                    # add these outputs
                    gammaeff_out[j, i - 1] = gammaeff
                kx_width_out[j] = kx_width
                sat_geo_factor_out[j] = sat_geo_factor

        # SAT3 QLA part
        QLA_P = 0.0
        QLA_E = 0.0
        if sat_rule_in == 3:
            QLA_P = np.zeros(nmodes)
            QLA_E = np.zeros(nmodes)
            for k in range(1, nmodes + 1):
                # factor of 2 included for real symmetry
                QLA_P[k - 1] = 2 * mode_transition_function(
                    xs[k - 1], 1.1, 0.6, x_ITG, x_TEM
                )
                QLA_E[k - 1] = 2 * mode_transition_function(
                    xs[k - 1], 0.75, 0.6, x_ITG, x_TEM
                )
            QLA_O = 2 * 0.8
        else:
            QLA_P = 1.0
            QLA_E = 1.0
            QLA_O = 1.0

        phinorm = field_spectrum_out
        # so the normal behavior doesn't change,
        if return_phi_params:
            out = dict(
                kx_width=kx_width_out,  # [nky] kx_model (kx rms width)
                gammaeff=gammaeff_out,  # [nky, nmodes] effective growthrate
                kx0_e=kx0_e,  # [nky] spectral shift in kx
                ax=ax,  # SAT1 (cx), SAT2 (alpha_x)
                ay=ay,  # SAT1 (cy)
                exp_ax=exp_ax,  # SAT2 (sigma_x)
            )
            if sat_rule_in == 2:
                # add bonus geometry params,
                out.update(dict(d1=d1, d2=d2, kycut=kycut, b3=b3))
        else:
            out = phinorm, QLA_P, QLA_E, QLA_O  # SAT123 intensity and QLA params

        return out

    @staticmethod
    def _invfft(f, x, y, kx, ky):
        """
        Returns f(x, y) = ifft[f(kx, ky)]
        """
        nkx = len(kx)
        kx = kx[:, np.newaxis, np.newaxis, np.newaxis]
        ky = ky[np.newaxis, :, np.newaxis, np.newaxis]
        x = x[np.newaxis, np.newaxis, :]
        y = y[np.newaxis, np.newaxis, :]
        f = f[:, :, np.newaxis, np.newaxis]
        rdotk = x * kx + y * ky
        value = (
            f[0, 0, :]
            + 2
            * np.sum(
                np.real(f[:, 1:, :]) * np.cos(rdotk[:, 1:, :])
                - np.imag(f[:, 1:, :]) * np.sin(rdotk[:, 1:, :]),
                axis=(0, 1),
            )
            + 2
            * np.sum(
                np.real(f[1 : (nkx // 2 + 1), 0]) * np.cos(rdotk[1 : (nkx // 2 + 1), 0])
                - np.imag(f[1 : (nkx // 2 + 1), 0])
                * np.sin(rdotk[1 : (nkx // 2 + 1), 0]),
                axis=(0, 1),
            )
        )
        return np.real(value)
