import numpy as np
import xrft
from scipy.integrate import quad, simpson
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import eigs
from ..pyro import Pyro
from .synthetic_highk_dbs import SyntheticHighkDBS
import numba

class Diagnostics:
    """
    Contains diagnostics for GENE/CGYRO/GS2 output, with a fast Poincare map.
    """

    def __init__(self, pyro):
        self.pyro = pyro

    def compute_l_per_turn(self):
        """
        Computes the distance along the field line per poloidal turn.

        This uses the local_geometry routines to integrate the differential
        arclength dLdtheta over a full poloidal period and scales it appropriately.
        In particular, the integration is performed on the dimensionless quantity

            dLdtheta / (R * grad_r)

        and the result is scaled by (Rmaj/(2*pi*rho)).

        Returns
        -------
        l_per_turn : float
            The field line length per turn.
        """

        def bunit_integrand(theta):
            # Get the flux surface R from local_geometry.
            R, _ = self.pyro.local_geometry.get_flux_surface(theta)
            # R_grad_r is R multiplied by the gradient of r.
            R_grad_r = R * self.pyro.local_geometry.get_grad_r(theta)
            # Differential arclength per poloidal angle.
            dLdtheta = self.pyro.local_geometry.get_dLdtheta(theta)
            # Dimensionless integrand.
            result = dLdtheta / R_grad_r
            return result

        # Integrate from 0 to 2*pi.
        integral = quad(bunit_integrand, 0.0, 2 * np.pi)[0]
        # Scale the integral to obtain the physical length.
        l_per_turn = (
            integral
            * self.pyro.local_geometry.Rmaj
            / (2 * np.pi * self.pyro.local_geometry.rho)
        )
        return l_per_turn

    def poincare(
        self,
        xarray: np.ndarray,
        yarray: np.ndarray,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 1.0,
        unwrap: bool = False,
    ):
        """
        Generates a Poincare map. It returns the (x, y) coordinates of the Poincare Map.

        If `unwrap` is False (default) the returned coordinates are wrapped into the
        periodic domain. If `unwrap` is True, the routine does not apply modulo operations
        so that the cumulative displacement is retained.

        You need to load the simulation output files before calling this function.

        Parameters
        ----------
        xarray : np.ndarray
            Array containing x coordinate of initial field line positions.
        yarray : np.ndarray
            Array containing y coordinate of initial field line positions.
        nturns : int
            Number of intersection points.
        time : float
            Time reference.
        rhostar : float
            Parameter required to set the boundary condition on the magnetic field line.
        use_invfft : bool, optional
            If T_irue, the inverse FFT is computed at every (x, y) point along the field line.
        smoothing : float, optional
            Smoothing parameter for RectBivariateSpline interpolation.
        unwrap : bool, optional
            If True, the coordinates are not wrapped into the periodic domain so that
            cumulative displacements are available.

        Returns
        -------
        points : np.ndarray
            4D array of shape (2, nturns, len(yarray), len(xarray)) containing the x and y
            coordinates for each turn. When unwrap is False the coordinates lie within the
            periodic domain.
        """
        import pint_xarray  

        if self.pyro.gk_output is None:
            raise RuntimeError(
                "Diagnostics: Please load gk output files (Pyro.load_gk_output) before using any diagnostic"
            )

        if self.pyro.gk_code not in ["CGYRO", "GS2", "GENE"]:
            raise NotImplementedError(
                "Poincare map only available for CGYRO, GENE and GS2"
            )
        if self.pyro.gk_input.is_linear():
            raise RuntimeError("Poincare only available for nonlinear runs")

        apar = self.pyro.gk_output["apar"].sel(time=time, method="nearest")
        apar = apar.pint.dequantify()
        kx = apar.kx.values
        ky = apar.ky.values
        ntheta = apar.theta.shape[0]
        nkx = kx.shape[0]
        nky = ky.shape[0]
        dkx = kx[1] - kx[0]
        dky = ky[1]
        ny = 2 * (nky - 1)
        nkx0 = nkx + 1 - np.mod(nkx, 2)
        
        # Define domain sizes
        Lx = 2 * np.pi / dkx
        Ly = 2 * np.pi / dky
        xgrid = np.linspace(-Lx / 2, Lx / 2, nkx0)[:nkx]
        ygrid = np.linspace(-Ly / 2, Ly / 2, ny)
        xmin = np.min(xgrid)
        ymin = np.min(ygrid)
        xmax = np.max(xgrid)
        # Recalculate Lx to avoid floating point issues.
        Lx = xmax - xmin

        # Geometrical factors from the simulation's local geometry.
        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2 * np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)
        nskip = len(geo.theta) // ntheta
        bmag = np.sqrt((1 / geo.R.m) ** 2 + geo.b_poloidal.m**2)
        bmag = np.roll(bmag[::nskip], ntheta // 2)
        jacob = self.pyro.metric_terms.Jacobian.m * geo.dpsidr.m * geo.bunit_over_b0.m
        jacob = np.roll(jacob[::nskip], ntheta // 2)
        dq = rhostar * Lx * geo.shat.m / geo.dpsidr.m
        qmin = geo.q.m - dq / 2
        fac1 = 2 * np.pi * geo.dpsidr.m / rhostar
        fac2 = geo.dpsidr.m * geo.q.m / geo.rho.m

        print(apar.kx)

        if use_invfft:
            # pull out the kx values from the DataArray
            kx = apar.kx.values

            # 1) symmetry check on kx
            if not np.allclose(kx, -kx[::-1], atol=1e-12):
                raise ValueError("kx must be symmetric around zero")

            # 2) locate the zero‐frequency index
            zero_idx = int(np.argmin(np.abs(kx)))
            if not np.isclose(kx[zero_idx], 0.0, atol=1e-12):
                raise ValueError("kx_in must contain zero frequency exactly")

            # 3) roll the DataArray (both values and coordinate) along the 'kx' dim
            apar = apar.roll(kx=-zero_idx, roll_coords=True)

            print(f"[invfft_xarray] zero_idx={zero_idx}, rolled so kx[0]= {apar.kx.values[0]}")
            print(apar.kx)
            # now apar.values is in FFT‐order and apar.kx reflects that shift

        # Compute bx and by in Fourier space.
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
                    s=smoothing,
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
                    s=smoothing,
                )
                for theta in bxfft.theta
            ]

        # Initialize positions.
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

                # Apply periodic boundaries if not unwrapping.
                if not unwrap:
                    x = xmin + np.mod(x - xmin, Lx)
                    y = ymin + np.mod(y - ymin, Ly)

            # Final y update from the q-profile twist.
            if not unwrap:
                y = y + np.mod(fac1 * ((x - xmin) / Lx * dq + qmin), Ly)
                y = ymin + np.mod(y - ymin, Ly)

            points[0, iturn, :, :] = x
            points[1, iturn, :, :] = y

        return points

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

    def radial_diffusion_coefficient(
        self,
        xarray: np.ndarray,
        yarray: np.ndarray,
        nturns: int,
        time: float,
        rhostar: float,
        l_per_turn: float = None,
        use_invfft: bool = False,
        smoothing: float = 1.0,
        unwrap: bool = True,
        ):
        """
        Calculates the radial diffusion coefficient using the definition

            D_r = <(r(l) - r(0))^2> / (2 * l_total)

        where r(l) is the radial (x) coordinate at turn l, and the average is taken
        over all field lines. Here, l_total = nturns * l_per_turn.

        Parameters
        ----------
        xarray : np.ndarray
            Array containing initial radial (x) positions.
        yarray : np.ndarray
            Array containing initial y positions.
        nturns : int
            Number of turns over which to integrate.
        time : float
            Time reference.
        rhostar : float
            Parameter for the flux-tube boundary condition.
        l_per_turn : float, optional
            Distance along the field line per turn. If not provided, it is computed using
            the local geometry.
        use_invfft : bool, optional
            Whether to use the inverse FFT method.
        smoothing : float, optional
            Smoothing parameter for interpolation.
        unwrap : bool, optional
            If True, positions are not wrapped so that the cumulative displacement is retained.

        Returns
        -------
        D_r : float
            The estimated radial diffusion coefficient.
        """
        if l_per_turn is None:
            l_per_turn = self.compute_l_per_turn()

        # Obtain the full (cumulative) Poincaré map
        points = self.poincare(
            xarray, yarray, nturns, time, rhostar, use_invfft, smoothing, unwrap
        )

        # r_initial: the initial radial coordinate, taken from xarray
        r_initial = xarray  # shape: (Nx,)

        # r_final: the radial coordinate at the last turn, shape: (Ny, Nx)
        r_final = points[0, -1, :, :]

        # The mean squared displacement (MSD) is the difference of these averages.
        msd = np.mean((r_final - r_initial[np.newaxis, :]) ** 2)

        # Total distance traveled along the field line.
        l_total = nturns * l_per_turn

        # Compute the radial diffusion coefficient.
        D_r = msd / (2 * l_total)

        # Debug prints for checking intermediate values:
        print("l_total =", l_total)
        print("MSD =", msd)
        print("Computed radial diffusion coefficient D_r =", D_r)

        return D_r


    def gs2_geometry_terms(self, ntheta_multiplier: int = 10):
        nperiod = self.pyro.numerics.nperiod
        ntheta = self.pyro.numerics.ntheta * ntheta_multiplier

        theta_max = ((2 * nperiod) - 1) * np.pi
        theta_even = np.linspace(-theta_max, theta_max, ntheta)

        self.pyro.load_metric_terms(theta=theta_even)

        metric = self.pyro.metric_terms

        theta = metric.regulartheta

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

    def compute_half_disp(
        self,
        time: float,
        max_fraction: float = 0.25,
        pad_factor: int = 2,
    ):
        """
        Returns the radial displacement of a magnetic field line
        after half a poloidal turn (0→π), following Pueschel et al. PRL 110, 155005.

        Fixes applied:
         - truly pads the real‐space x grid (pad_factor*nkx)
         - corrects dky = ky[1]-ky[0]
         - wraps y via modulo to avoid mis-wraps
         - monitors per‐step dx and errors out if > max_fraction*Lx
        """
        import pint_xarray  # noqa

        if self.pyro.gk_output is None:
            raise RuntimeError("Load gk output first (Pyro.load_gk_output).")
        if self.pyro.gk_input.is_linear():
            raise RuntimeError("Displacement only available for nonlinear runs")

        # --- pull A_par slice and dequantify ---
        apar = self.pyro.gk_output["apar"].sel(time=time, method="nearest")
        apar = apar.pint.dequantify()

        # --- spectral grid & padding ---
        kx = apar.kx.values
        ky = apar.ky.values
        nkx = len(kx)
        pad_nkx = int(pad_factor * nkx)
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        Lx = 2 * np.pi / dkx
        Ly = 2 * np.pi / dky

        # --- real‐space grid (padded) ---
        xgrid = np.linspace(-Lx/2, Lx/2, pad_nkx, endpoint=False)
        ny = 2 * (len(ky) - 1)
        ygrid = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)

        # --- geometry factors ---
        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2*np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)

        ntheta = apar.theta.size
        nskip = len(geo.theta) // ntheta

        bmag = np.sqrt((1/geo.R.m)**2 + geo.b_poloidal.m**2)
        bmag = np.roll(bmag[::nskip], ntheta//2)

        jacob = (
            self.pyro.metric_terms.Jacobian.m
            * geo.dpsidr.m
            * geo.bunit_over_b0.m
        )
        jacob = np.roll(jacob[::nskip], ntheta//2)

        fac = geo.dpsidr.m * geo.q.m / geo.rho.m

        # precompute Fourier coefficients
        ikxA = (1j * apar.kx * apar)\
            .transpose("theta", "kx", "ky")\
            .values
        ikyA = (-1j * apar.ky * apar)\
            .transpose("theta", "kx", "ky")\
            .values

        dtheta = 2 * np.pi / ntheta
        max_jump = max_fraction * Lx

        disp = np.zeros((pad_nkx, ny))
        for ix, x0 in enumerate(xgrid):
            for iy, y0 in enumerate(ygrid):
                x = x0
                y = y0
                for ith in range(ntheta // 2):
                    # --- first sub‐step ---
                    xx = np.array([[x]], dtype=float)
                    yy = np.array([[y]], dtype=float)

                    dby = (
                        self._invfft(ikxA[ith], xx, yy, kx, ky)[0, 0]
                        * bmag[ith]
                        * fac
                    )
                    dbx = (
                        self._invfft(ikyA[ith], xx, yy, kx, ky)[0, 0]
                        * bmag[ith]
                        * fac
                    )

                    dx_step = abs(dbx * dtheta * jacob[ith])
                    if dx_step > max_jump:
                        raise RuntimeError(
                            f"dx step {dx_step:.3e} > {max_jump:.3e}; "
                            "increase ntheta or decrease pad_factor"
                        )

                    x += dtheta * dbx * jacob[ith]
                    y += dtheta * dby * jacob[ith]
                    # poloidal wrap
                    y = ((y + Ly/2) % Ly) - Ly/2

                    # --- second sub‐step ---
                    idx2 = ith + 1
                    xx = np.array([[x]], dtype=float)
                    yy = np.array([[y]], dtype=float)

                    dby = (
                        self._invfft(ikxA[idx2], xx, yy, kx, ky)[0, 0]
                        * bmag[idx2]
                        * fac
                    )
                    dbx = (
                        self._invfft(ikyA[idx2], xx, yy, kx, ky)[0, 0]
                        * bmag[idx2]
                        * fac
                    )

                    dx_step = abs(dbx * dtheta * jacob[idx2])
                    if dx_step > max_jump:
                        raise RuntimeError(
                            f"dx step {dx_step:.3e} > {max_jump:.3e}; "
                            "increase ntheta or decrease pad_factor"
                        )

                    x += dtheta * dbx * jacob[idx2]
                    y += dtheta * dby * jacob[idx2]
                    y = ((y + Ly/2) % Ly) - Ly/2

                disp[ix, iy] = abs(x - x0)

        return disp

    def compute_half_disp_fast(self, time: float, max_fraction: float = 0.25):
        """
        Fast estimation of mean radial displacement ⟨δr⟩ after half poloidal turn.
        Downsamples x, y, and theta for speed. Suitable for trend analysis.
        """
        import numpy as np
        import pint_xarray

        apar = self.pyro.gk_output["apar"].sel(time=time, method="nearest")
        apar = apar.pint.dequantify()

        # Downsample theta for speed (e.g. 128 → 32)
        apar = apar.isel(theta=slice(None, None, 4))
        ntheta = apar.theta.size
        dtheta = 2 * np.pi / ntheta

        kx = apar.kx.values
        ky = apar.ky.values
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        Lx = 2 * np.pi / dkx
        Ly = 2 * np.pi / dky

        xgrid = np.linspace(-Lx/2, Lx/2, 8)
        ygrid = np.linspace(-Ly/2, Ly/2, 5)

        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2*np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)
        nskip = len(geo.theta) // ntheta

        bmag = np.sqrt((1/geo.R.m)**2 + geo.b_poloidal.m**2)
        bmag = np.roll(bmag[::nskip], ntheta//2)

        jacob = self.pyro.metric_terms.Jacobian.m * geo.dpsidr.m * geo.bunit_over_b0.m
        jacob = np.roll(jacob[::nskip], ntheta//2)

        fac = geo.dpsidr.m * geo.q.m / geo.rho.m

        ikxA = (1j * apar.kx * apar).transpose("theta","kx","ky").values
        ikyA = (-1j * apar.ky * apar).transpose("theta","kx","ky").values

        max_jump = max_fraction * Lx
        disp_vals = []

        for x0 in xgrid:
            for y0 in ygrid:
                x, y = x0, y0
                for ith in range(ntheta // 2):
                    for sub in [0, 1]:
                        idx = ith + sub
                        xx = np.array([[x]])
                        yy = np.array([[y]])
                        dby = self._invfft(ikxA[idx], xx, yy, kx, ky)[0,0].real * bmag[idx] * fac
                        dbx = self._invfft(ikyA[idx], xx, yy, kx, ky)[0,0].real * bmag[idx] * fac
                        dx = dtheta * dbx * jacob[idx]
                        dy = dtheta * dby * jacob[idx]
                        if abs(dx) > max_jump:
                            print(f"[warn] dx step = {dx:.3e} exceeds {max_jump:.3e}")
                        x += dx
                        y = ((y + dy + Ly/2) % Ly) - Ly/2
                disp_vals.append(abs(x - x0))

        return np.mean(disp_vals)

    def compute_corr_length(
        self,
        time: float,
        yarray: np.ndarray,
        Nx: int = 64,
        ndelta: int = None,
    ):
        """
        Fast radial correlation length via Wiener–Khinchin (drops ndelta).

        Returns λ_x at each y in yarray, finding Δ where C(Δ)=1/e.
        """
        # 1) load & dequantify A_par
        apar = self.pyro.gk_output["apar"]\
            .sel(time=time, method="nearest")\
            .pint.dequantify()\
            .transpose("theta", "kx", "ky")

        kx = apar.kx.values
        ky = apar.ky.values
        ntheta = apar.theta.size

        # 2) real-space grid
        dkx = kx[1] - kx[0]
        Lx = 2 * np.pi / dkx
        x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        dx = x[1] - x[0]

        lam_y = np.empty(len(yarray))

        # 3) loop over y, θ
        for iy, y in enumerate(yarray):
            lam_theta = np.empty(ntheta)
            for ith in range(ntheta):
                # build b(x) via one invfft at this θ,y
                A_k = apar.values[ith]  # shape (kx,ky)
                coeff = -1j * apar.ky.values  # shape (ky,)
                b_x = self._invfft(
                    A_k * coeff[None, :],
                    x[None, :],
                    np.array([[y]]),
                    kx, ky
                )[0].real

                # Wiener–Khinchin correlation
                Pk = np.abs(np.fft.fft(b_x)) ** 2
                C = np.fft.ifft(Pk).real
                C /= C[0]

                below = np.where(C < 1/np.e)[0]
                lam_theta[ith] = below[0] * dx if below.size else x[-1]

            lam_y[iy] = lam_theta.mean()

        return lam_y


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
    h = np.diff(theta_ball_u_half)[2]
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
