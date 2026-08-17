import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline
from xrft import ifft

from ..pyro import Pyro
from ..units import ureg


class FieldLine:

    def __init__(self, pyro: Pyro):
        self.pyro = pyro

    def compute_linear_tearing_parameter(
        self,
    ):
        """
        Computes the distance along the field line per poloidal turn.

        Use metric_terms to determine the field aligned covariant metric
        g_theta_theta from which we can get dLdtheta by

        dLdtheta = 1 / sqrt(g_theta_theta)

        This is then integrated over the poloidal turn to determine the
        distance travelled along the field line

        Parameters
        ----------
        ntheta: int
            Number of theta points to be used for MetricTerms

        Returns
        -------
        length_per_turn : float, units [lref]
            The field line length per turn.
        """

        # 1) load & dequantify A_par
        apar = self.pyro.gk_output["apar"].isel(kx=0, ky=0, time=-1).pint.dequantify()

        theta = apar.theta.values

        theta_limits = np.min((np.max(theta), np.abs(np.min(theta))))

        apar = apar.where(apar.theta >= -theta_limits, drop=True)
        apar = apar.where(apar.theta <= theta_limits, drop=True)

        theta = apar.theta.values
        ntheta = apar.theta.size
        nperiod = self.pyro.numerics.nperiod

        self.pyro.load_metric_terms(ntheta=ntheta * 4)
        metric = self.pyro.metric_terms
        theta_metric = metric.regulartheta

        # Ballooning space recreation of metric_terms
        m = np.linspace(-(nperiod - 1), nperiod - 1, 2 * nperiod - 1)
        ntheta_metric = len(theta_metric) - 1
        m = np.repeat(m, ntheta_metric)
        theta_long = np.tile(theta_metric[:-1], 2 * nperiod - 1) + 2.0 * np.pi * m

        # Append final point
        theta_geo_final = 2 * np.pi * (m[-1]) + np.pi
        theta_long = np.append(theta_long, theta_geo_final)

        g_tt = metric.field_aligned_covariant_metric("theta", "theta")
        g_tt_long = np.tile(g_tt[:-1], 2 * nperiod - 1)
        g_tt_long = np.append(g_tt_long, g_tt[-1])

        g_tt_balloon = np.interp(theta, theta_long, g_tt_long)

        apar_dl = apar * np.sqrt(g_tt_balloon)

        linear_tearing_parameter = np.abs(simpson(apar_dl, x=theta)) / simpson(
            np.abs(apar_dl), x=theta
        )

        return linear_tearing_parameter

    def compute_length_per_turn(self, ntheta=256):
        """
        Computes the distance along the field line per poloidal turn.

        Use metric_terms to determine the field aligned covariant metric
        g_theta_theta from which we can get dLdtheta by

        dLdtheta = 1 / sqrt(g_theta_theta)

        This is then integrated over the poloidal turn to determine the
        distance travelled along the field line

        Parameters
        ----------
        ntheta: int
            Number of theta points to be used for MetricTerms

        Returns
        -------
        length_per_turn : float, units [lref]
            The field line length per turn.
        """
        self.pyro.load_metric_terms(ntheta=ntheta)
        metric = self.pyro.metric_terms

        g_tt = metric.field_aligned_covariant_metric("theta", "theta")
        dLdtheta = np.sqrt(g_tt)

        @ureg.wraps(dLdtheta.units, (dLdtheta.units, None))
        def simpson_dLdtheta(dLdtheta, theta):
            return simpson(dLdtheta, x=theta)

        length_per_turn = simpson_dLdtheta(dLdtheta, metric.regulartheta)

        return length_per_turn

    def compute_half_displacement(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 0.0,
        unwrap: bool = False,
        max_fraction: float = 0.25,
        pad_factor: int = 1,
        integration_order: int = 4,
    ):
        """
        Integrates delta B along the field line from a set of starting points.
        It returns the final (x, y) coordinates of the points. You can specify
        how many turns along the field line you wish to integrate

        This routine may take a while depending on ``nturns`` and on
        the number of magnetic field lines. The parameter rhostar is
        required by the flux-tube boundary condition.
        Available for nonlinear simulations
        If `unwrap` is False (default) the returned coordinates are wrapped into the
        periodic domain. If `unwrap` is True, the routine does not apply modulo operations
        so that the cumulative displacement is retained.

        You need to load the simulation output files before calling this function.

        Parameters
        ----------
        xarray: ArrayLike, units [rhoref]
            Array containing x coordinate of initial field line positions
        yarray: ArrayLike, units [rhoref]
            Array containing y coordinate of initial field line positions
        time: float, time reference
        rhostar: float, units [rhoref / lref]
            rhostar is needed to set the boundary conditionon the magnetic field line
        use_invfft: bool
            If True, the inverse Fourier transform is computed
            every (x, y) points along the magnetic field line. It is much
            more accurate but very slow.
        smoothing: float
            Sets level of smoothing done in RectBivariateSpline
        unwrap : Optional[bool]
            If True, the x- coordinates are not wrapped into the periodic domain so that
            cumulative displacements are available.
        theta_min: float
            Sets lower limit of field line integral (default -pi). Can't be set if nturns>1
        theta_max: float
            Sets upper limit of field line integral (default pi). Can't be set if nturns>1
        max_fraction : float, optional
           Maximum allowed x‐step size as a fraction of the full radial domain Lx.
           Steps with ``|Δx|`` > ``max_fraction * Lx`` issue a warning but are retained.
           Default is 0.25.
        pad_factor : int, optional
           Factor by which to pad the kx spectrum before transforming.
           A value of 2 doubles the kx resolution (nx → 2·nx). Larger
           values increase real-space grid resolution but cost more CPU.
           Default is 1.
        Returns
        -------
        points: ArrayLike, units [rhoref]
            4D array of shape (2, nturns, len(yarray), len(xarray))
            containing the x and y coordinates shaped according to the initial
            field line position. See ``example_poincare.py`` for a simple example.

        Raises
        ------
        RuntimeError:
            In case of linear simulation or GKOutput not loaded
        """
        # Set number of turns to 1 and range from 0 to pi
        nturns = 1
        theta_min = 0.0
        theta_max = np.pi

        points = self.follow_field_line(
            xarray=xarray,
            yarray=yarray,
            nturns=nturns,
            time=time,
            rhostar=rhostar,
            use_invfft=use_invfft,
            smoothing=smoothing,
            unwrap=unwrap,
            theta_min=theta_min,
            theta_max=theta_max,
            max_fraction=max_fraction,
            pad_factor=pad_factor,
            integration_order=integration_order,
        )

        displacement = np.abs(xarray - points[0, 0, :, :])

        return displacement

    def poincare(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 0.0,
        unwrap: bool = False,
        max_fraction: float = 0.25,
        pad_factor: int = 1,
        integration_order: int = 4,
    ):
        """
        Integrates delta B along the field line from a set of starting points.
        It returns the final (x, y) coordinates of the points. You can specify
        how many turns along the field line you wish to integrate

        This routine may take a while depending on ``nturns`` and on
        the number of magnetic field lines. The parameter rhostar is
        required by the flux-tube boundary condition.
        Available for nonlinear simulations
        If `unwrap` is False (default) the returned coordinates are wrapped into the
        periodic domain. If `unwrap` is True, the routine does not apply modulo operations
        so that the cumulative displacement is retained.

        You need to load the simulation output files before calling this function.

        Parameters
        ----------
        xarray: ArrayLike, units [rhoref]
            Array containing x coordinate of initial field line positions
        yarray: ArrayLike, units [rhoref]
            Array containing y coordinate of initial field line positions
        nturns: int, number of intersection points
        time: float, time reference
        rhostar: float, units [rhoref / lref]
            rhostar is needed to set the boundary conditionon the magnetic field line
        use_invfft: bool
            If True, the inverse Fourier transform is computed
            every (x, y) points along the magnetic field line. It is much
            more accurate but very slow.
        smoothing: float
            Sets level of smoothing done in RectBivariateSpline
        unwrap : Optional[bool]
            If True, the x- coordinates are not wrapped into the periodic domain so that
            cumulative displacements are available.
        max_fraction: float, optional
           Maximum allowed x‐step size as a fraction of the full radial domain Lx.
           Steps with ``|Δx|`` > ``max_fraction * Lx`` issue a warning but are retained.
           Default is 0.25.
        pad_factor: int, optional
           Factor by which to pad the kx spectrum before transforming.
           A value of 2 doubles the kx resolution (nx → 2·nx). Larger
           values increase real-space grid resolution but cost more CPU.
           Default is 1.
        integration_order: int, optional
            Order of Runge Kutta scheme to be used when integrating along
            field line
        Returns
        -------
        points: ArrayLike, units [rhoref]
            4D array of shape (2, nturns, len(yarray), len(xarray))
            containing the x and y coordinates shaped according to the initial
            field line position. See ``example_poincare.py`` for a simple example.

        Raises
        ------
        RuntimeError:
            In case of linear simulation or GKOutput not loaded
        """

        points = self.follow_field_line(
            xarray=xarray,
            yarray=yarray,
            nturns=nturns,
            time=time,
            rhostar=rhostar,
            use_invfft=use_invfft,
            smoothing=smoothing,
            unwrap=unwrap,
            max_fraction=max_fraction,
            pad_factor=pad_factor,
            integration_order=integration_order,
        )

        return points

    def radial_diffusion_coefficient(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        length_per_turn: float = None,
        use_invfft: bool = False,
        smoothing: float = 0.0,
        unwrap: bool = True,
    ):
        """
        Calculates the radial diffusion coefficient using the definition

            D_r = <(r(l) - r(0))^2> / (2 * l_total)

        where r(l) is the radial (x) coordinate at turn l, and the average is taken
        over all field lines. Here, l_total = nturns * length_per_turn.

        Parameters
        ----------
        xarray : ArrayLike
            Array containing initial radial (x) positions.
        yarray : ArrayLike
            Array containing initial y positions.
        nturns : int
            Number of turns over which to integrate.
        time : float
            Time reference.
        rhostar : float
            Parameter for the flux-tube boundary condition.
        length_per_turn : float, optional
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
        if length_per_turn is None:
            length_per_turn = self.compute_length_per_turn()

        length_per_turn = length_per_turn.to(
            self.pyro.norms.lref, self.pyro.norms.context
        )

        # Obtain the full (cumulative) Poincaré map
        points = self.poincare(
            xarray, yarray, nturns, time, rhostar, use_invfft, smoothing, unwrap
        )

        rhostar *= self.pyro.norms.lref / self.pyro.norms.rhoref

        # r_initial: the initial radial coordinate, taken from xarray
        r_initial = xarray  # shape: (Nx,)

        # r_final: the radial coordinate at the last turn, shape: (Ny, Nx)
        r_final = points[0, -1, :, :]

        # The mean squared displacement (MSD) is the difference of these averages.
        msd = np.mean((r_final - r_initial[np.newaxis, :]) ** 2)

        # Total distance traveled along the field line.
        length_total = nturns * length_per_turn

        # Compute the radial diffusion coefficient.
        D_r = msd / (2 * length_total) * rhostar

        return D_r

    def parallel_correlation_length(
        self,
        time: float,
    ):
        """
        Compute the radial correlation length λ_x at each y using the Wiener–Khinchin theorem.

        #. Reconstructs ``b(x) = ∂y A_par`` via direct Fourier-mode summation
           using ``self._invfft``.

        #. Computes the power spectrum ``P(k) = |FFT[b(x)]|²``.

        #. Obtains the autocorrelation ``C(Δ) = IFFT[P(k)]``,
           normalized so ``C(0)=1``.

        #. Identifies ``λ_x(θ,y)`` as the smallest ``Δ`` where
           ``C(Δ) = 1/e``.

        Finally, λ_x(y) is taken as the mean over θ.

        You need to load the simulation output files before calling this function.

        Parameters
        ----------
        time: float
            Time reference

        Returns
        -------
        points: ArrayLike, units [rhoref]
            4D array of shape (2, nturns, len(yarray), len(xarray))
            containing the x and y coordinates shaped according to the initial
            field line position. See ``example_poincare.py`` for a simple example.

        Raises
        ------
        RuntimeError:
            In case of linear simulation or GKOutput not loaded
        """

        if self.pyro.gk_output is None:
            raise RuntimeError(
                "Diagnostics: Please load gk output files (Pyro.load_gk_output)"
                " before using any diagnostic"
            )

        if self.pyro.gk_input.is_linear():
            raise RuntimeError("Poincare only available for nonlinear runs")

        apar = self.pyro.gk_output["apar"].sel(time=time, method="nearest")

        k_units = self.pyro.gk_output["ky"].units
        apar = apar.pint.dequantify()
        kx = apar.kx.values * k_units

        theta = apar.theta.values

        # Geometrical factors from the simulation's local geometry.
        local_geometry = self.pyro.local_geometry
        local_geometry.normalise(self.pyro.norms)

        theta_metric = np.linspace(-np.pi, np.pi, len(theta) * 4)
        self.pyro.load_metric_terms(theta=theta_metric)
        metric_terms = self.pyro.metric_terms

        Jacobian = np.interp(
            theta,
            metric_terms.regulartheta,
            metric_terms.Jacobian,
        )

        # Compute Bx in Fourier space.
        Bx = -1j * apar.ky * apar
        Jacobian_int = simpson(Jacobian.m, x=theta) * Jacobian.units
        Bx_sq = (np.abs(Bx) ** 2 * Jacobian[None, :, None]).sum(dim="ky").integrate(
            coord="theta"
        ) / Jacobian_int

        corr_length = Bx_sq.integrate(coord="kx") / (np.abs(kx) * Bx_sq).integrate(
            coord="kx"
        )

        return corr_length.data

    def follow_field_line(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 0.0,
        unwrap: bool = False,
        theta_min: float = None,
        theta_max: float = None,
        max_fraction: float = 0.25,
        pad_factor: int = 1,
        integration_order=4,
        return_delta_x: bool = False,
    ):
        """
        Integrates delta B along the field line from a set of starting points.
        It returns the final (x, y) coordinates of the points. You can specify
        how many turns along the field line you wish to integrate

        This routine may take a while depending on ``nturns`` and on
        the number of magnetic field lines. The parameter rhostar is
        required by the flux-tube boundary condition.
        Available for nonlinear simulations
        If `unwrap` is False (default) the returned coordinates are wrapped into the
        periodic domain. If `unwrap` is True, the routine does not apply modulo operations
        so that the cumulative displacement is retained.

        You need to load the simulation output files before calling this function.

        Parameters
        ----------
        xarray: ArrayLike, units [rhoref]
            Array containing x coordinate of initial field line positions
        yarray: ArrayLike, units [rhoref]
            Array containing y coordinate of initial field line positions
        nturns: int, number of intersection points
        time: float, time reference
        rhostar: float, units [rhoref / lref]
            rhostar is needed to set the boundary conditionon the magnetic field line
        use_invfft: bool
            If True, the inverse Fourier transform is computed
            every (x, y) points along the magnetic field line. It is much
            more accurate but very slow.
        smoothing: float
            Sets level of smoothing done in RectBivariateSpline
        unwrap : Optional[bool]
            If True, the x- coordinates are not wrapped into the periodic domain so that
            cumulative displacements are available.
        theta_min: float
            Sets lower limit of field line integral (default -pi). Can't be set if nturns>1
        theta_max: float
            Sets upper limit of field line integral (default pi). Can't be set if nturns>1
        max_fraction : float, optional
           Maximum allowed x‐step size as a fraction of the full radial domain Lx.
           Steps with ``|Δx|`` > ``max_fraction * Lx`` issue a warning but are retained.
           Default is 0.25.
        pad_factor : int, optional
           Factor by which to pad the kx spectrum before transforming.
           A value of 2 doubles the kx resolution (nx → 2·nx). Larger
           values increase real-space grid resolution but cost more CPU.
           Default is 1.
        Returns
        -------
        points: ArrayLike, units [rhoref]
            4D array of shape (2, nturns, len(yarray), len(xarray))
            containing the x and y coordinates shaped according to the initial
            field line position. See ``example_poincare.py`` for a simple example.

        Raises
        ------
        RuntimeError:
            In case of linear simulation or GKOutput not loaded
        """

        if self.pyro.gk_output is None:
            raise RuntimeError(
                "Diagnostics: Please load gk output files (Pyro.load_gk_output)"
                " before using any diagnostic"
            )

        if self.pyro.gk_input.is_linear():
            raise RuntimeError("Poincare only available for nonlinear runs")

        apar = self.pyro.gk_output["apar"].sel(time=time, method="nearest")

        if theta_min and nturns > 1:
            raise ValueError("Can't have a theta_min with nturns > 1")
        if theta_max and nturns > 1:
            raise ValueError("Can't have a theta_max with nturns > 1")

        if theta_min is not None and theta_max is not None:
            apar = apar.where(apar.theta <= theta_max, drop=True)
            apar = apar.where(apar.theta >= theta_min, drop=True)
        else:
            theta_min = -np.pi
            theta_max = np.pi

        # Drop final kx so kx grid is centred when using xrft
        if not use_invfft and not np.isclose(apar.kx.values[len(apar.kx) // 2], 0.0):
            apar = apar.isel(kx=slice(None, -1))

        apar_units = apar.data.units
        k_units = self.pyro.gk_output["ky"].units
        apar = apar.pint.dequantify()
        kx = apar.kx.values * k_units
        ky = apar.ky.values * k_units

        theta = apar.theta.values
        ntheta = apar.theta.shape[0]
        nkx = kx.shape[0]
        nkx = int(pad_factor * nkx)
        nky = ky.shape[0]
        dkx = kx[1] - kx[0]
        dky = ky[1]
        ny = 2 * (nky - 1)
        nkx0 = nkx + 1 - np.mod(nkx, 2)

        if not hasattr(xarray, "units"):
            xarray *= 1.0 / k_units
        elif xarray.units != k_units**-1:
            raise ValueError("Please use the same units for xarray as output")

        if not hasattr(yarray, "units"):
            yarray *= 1.0 / k_units
        elif yarray.units != k_units**-1:
            raise ValueError("Please use the same units for yarray as output")

        x_units = xarray.units
        b_units = self.pyro.norms.pyrokinetics.bref

        if not hasattr(rhostar, "units"):
            rhostar *= k_units * self.pyro.norms.lref

        # Define domain sizes
        Ly = 2 * np.pi / dky
        Lx = 2 * np.pi / dkx
        xgrid = np.linspace(-Lx / 2, Lx / 2, nkx0)[:nkx]
        ygrid = np.linspace(-Ly / 2, Ly / 2, ny)
        xmin = np.min(xgrid)
        ymin = np.min(ygrid)
        xmax = np.max(xgrid)
        # Recalculate Lx to avoid floating point issues.
        Lx = xmax - xmin
        max_jump = max_fraction * Lx

        # Geometrical factors from the simulation's local geometry.
        local_geometry = self.pyro.local_geometry
        local_geometry.normalise(self.pyro.norms)

        theta_metric = np.linspace(-np.pi, np.pi, len(theta) * 4)
        self.pyro.load_metric_terms(theta=theta_metric)
        metric_terms = self.pyro.metric_terms
        theta_metric = metric_terms.regulartheta

        # y = C_y alpha
        dpsidr = metric_terms.dpsidr.to(self.pyro.norms, self.pyro.norms.context)
        C_y = (dpsidr / b_units).to(self.pyro.norms.lref, self.pyro.norms.context)

        alpha = metric_terms.alpha
        alpha_theta_range = np.interp([theta_min, theta_max], theta_metric, alpha)
        dalpha_q = (alpha_theta_range[1] - alpha_theta_range[0]) / metric_terms.q
        twist_prefactor = dalpha_q * C_y / rhostar

        # Delta q over the radial simulation domain
        delta_q = (metric_terms.dqdr * Lx * rhostar).to(
            ureg.dimensionless, self.pyro.norms.context
        )

        qmin = local_geometry.q - delta_q / 2

        # dtheta
        dtheta_grid = np.diff(theta, append=theta[0] + theta_max - theta_min)

        # g_theta_theta (field aligned covariant)
        g_tt = np.interp(
            theta,
            metric_terms.regulartheta,
            metric_terms.field_aligned_covariant_metric("theta", "theta"),
        )

        l_units = np.sqrt(g_tt).units
        g_tt = g_tt.m

        # Compute bx and by in Fourier space.
        ikxapar = 1j * apar.kx * apar
        ikyapar = -1j * apar.ky * apar

        ikxapar = ikxapar.transpose("kx", "ky", "theta") * np.sqrt(g_tt)
        ikyapar = ikyapar.transpose("kx", "ky", "theta") * np.sqrt(g_tt)

        if use_invfft:
            ikxapar = ikxapar.values
            ikyapar = ikyapar.values
        else:
            byfft = ifft(
                ikxapar * nkx * ny,
                dim=["kx", "ky"],
                real_dim="ky",
                lag=[0, 0],
                true_amplitude=False,
            )
            bxfft = ifft(
                ikyapar * nkx * ny,
                dim=["kx", "ky"],
                real_dim="ky",
                lag=[0, 0],
                true_amplitude=False,
            )

            # xrft assume we have a kx such that exp(i 2pi * kx * x) but we actually have
            # exp(i kx * x) with the 2pi embedded in the definition so we need to redefine
            # our real space coordinate
            bxfft = bxfft.assign_coords(
                {
                    "freq_kx": bxfft.freq_kx.data * (2 * np.pi),
                    "freq_ky": bxfft.freq_ky.data * (2 * np.pi),
                }
            )
            byfft = byfft.assign_coords(
                {
                    "freq_kx": byfft.freq_kx.data * (2 * np.pi),
                    "freq_ky": byfft.freq_ky.data * (2 * np.pi),
                }
            )

            By = [
                RectBivariateSpline(
                    byfft.freq_kx.data,
                    byfft.freq_ky.data,
                    byfft.sel(theta=theta, method="nearest").data,
                    kx=5,
                    ky=5,
                    s=smoothing,
                )
                for theta in byfft.theta
            ]
            Bx = [
                RectBivariateSpline(
                    bxfft.freq_kx.data,
                    byfft.freq_ky.data,
                    bxfft.sel(theta=theta, method="nearest").data,
                    kx=5,
                    ky=5,
                    s=smoothing,
                )
                for theta in bxfft.theta
            ]

        # Initialize positions.
        x = xarray[np.newaxis, :]
        y = yarray[:, np.newaxis]

        points = np.empty((2, nturns, len(yarray), len(xarray))) * x_units
        if return_delta_x:
            delta_x = np.empty((ntheta, nturns, len(yarray), len(xarray))) * x_units

        # Handle units for eval_dx_dy, best to do all at once for speed
        dB_units = (1.0 * apar_units * k_units * l_units).to(
            x_units * b_units, self.pyro.norms.context
        )
        dl_units = (dB_units / b_units).to(x_units, self.pyro.norms.context)
        xk_factor = (1 * k_units * x_units).to("dimensionless").m

        def eval_dx_dy(theta_idx, x_in, y_in):
            if use_invfft:
                dBx = self._invfft(
                    ikyapar[:, :, theta_idx], x_in.m, y_in.m, kx.m, ky.m, xk_factor
                )
                dBy = self._invfft(
                    ikxapar[:, :, theta_idx], x_in.m, y_in.m, kx.m, ky.m, xk_factor
                )
            else:
                dBx = Bx[theta_idx](x_in.m, y_in.m, grid=False)
                dBy = By[theta_idx](x_in.m, y_in.m, grid=False)

            dx = dBx * dl_units
            dy = dBy * dl_units

            if np.any(dx > max_jump):
                raise RuntimeError(
                    f"dx step {dx:.3e} > {max_jump:.3e}; "
                    "increase ntheta or decrease pad_factor"
                )

            return dx, dy

        # RK method for integration, note theta goes from -pi to pi
        theta_outboard = np.argmin(np.abs(theta))
        for iturn in range(nturns):
            for ith_loop in range(0, ntheta):
                ith = (ith_loop + theta_outboard) % ntheta
                ith_next = (ith + 1) % ntheta
                dtheta = dtheta_grid[ith]

                # RK1 step (Euler) if you want simplest version
                if integration_order == 1:
                    dx, dy = eval_dx_dy(ith, x, y)

                    dx *= dtheta
                    dy *= dtheta

                # RK2 method (Improved Euler)
                elif integration_order == 2:
                    # First stage (predictor)
                    k1x, k1y = eval_dx_dy(ith, x, y)
                    k1x = dtheta * k1x
                    k1y = dtheta * k1y
                    # Second stage (corrector)
                    k2x, k2y = eval_dx_dy(ith_next, x + k1x, y + k1y)
                    k2x = dtheta * k2x
                    k2y = dtheta * k2y
                    # Update x and y with the average

                    dx = 0.5 * (k1x + k2x)
                    dy = 0.5 * (k1y + k2y)

                elif integration_order == 3:
                    # Should k2x technically be on ith + 1/2?
                    k1x, k1y = eval_dx_dy(ith, x, y)

                    # Interpolate along field line to get halfway point
                    k2x_a, k2y_a = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )
                    k2x_b, k2y_b = eval_dx_dy(
                        ith_next, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )

                    k2x = (k2x_a + k2x_b) / 2
                    k2y = (k2y_a + k2y_b) / 2

                    k3x, k3y = eval_dx_dy(
                        ith_next,
                        x - dtheta * k1x + 2 * dtheta * k2x,
                        y - dtheta * k1y + 2 * dtheta * k2y,
                    )

                    dx = dtheta * (k1x + 4 * k2x + k3x) / 6
                    dy = dtheta * (k1y + 4 * k2y + k3y) / 6

                elif integration_order == 4:
                    # Should k2x and k3x technically be on ith + 1/2?
                    k1x, k1y = eval_dx_dy(ith, x, y)

                    # Interpolate along field line to get halfway point
                    k2x_a, k2y_a = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )
                    k2x_b, k2y_b = eval_dx_dy(
                        ith_next, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )
                    k2x = (k2x_a + k2x_b) / 2
                    k2y = (k2y_a + k2y_b) / 2

                    k3x_a, k3y_a = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k2x, y + 0.5 * dtheta * k2y
                    )
                    k3x_b, k3y_b = eval_dx_dy(
                        ith_next, x + 0.5 * dtheta * k2x, y + 0.5 * dtheta * k2y
                    )

                    k3x = (k3x_a + k3x_b) / 2
                    k3y = (k3y_a + k3y_b) / 2

                    k4x, k4y = eval_dx_dy(ith_next, x + dtheta * k3x, y + dtheta * k3y)

                    dx = dtheta * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
                    dy = dtheta * (k1y + 2 * k2y + 2 * k3y + k4y) / 6

                else:
                    raise ValueError("Only RK order 1, 3, or 4 supported.")

                # Add dx and dy to x and y
                x = x + dx
                y = y + dy

                if return_delta_x:
                    delta_x[ith_loop, iturn, :, :] = dx[0, 0]

                # Apply periodic boundaries on x only if not unwrapping
                if not unwrap:
                    x = xmin + np.mod(x - xmin, Lx)

            # Always apply the q-profile twist to y
            y = y + twist_prefactor * (qmin + (x - xmin) / Lx * delta_q)

            # Always wrap y back into [ymin, ymin + Ly)
            y = ymin + np.mod(y - ymin, Ly)

            points[0, iturn, :, :] = x
            points[1, iturn, :, :] = y

        if return_delta_x:
            return delta_x

        return points

    @staticmethod
    def _invfft(F, x, y, kx, ky, xk_factor=1.0):
        """
        Manual real‐field inverse via direct summation over Fourier modes,
        including automatic rolling of kx/F into FFT order.

        Parameters
        ----------
        F : ArrayLike, shape (nkx, nky)
            Complex half‐spectrum array with ky ≥ 0, and rows of F
            corresponding to kx sorted from negative → 0 → positive.
        x : ArrayLike, shape (1, nx) or (ny, nx)
            Broadcastable x‐coordinates.
        y : ArrayLike, shape (ny, 1) or (ny, nx)
            Broadcastable y‐coordinates.
        kx : ArrayLike, shape (nkx,)
            1D kx vector sorted ascending (negative → positive).
        ky : ArrayLike, shape (nky,)
            1D ky vector (only non-negative values).
        xk_factor : float
            Factor to handle case where x and k are using different
        Returns
        -------
        f_xy : ArrayLike, shape (ny, nx)
            Real‐space field evaluated on the broadcast grid defined by x, y.
        """
        # roll zero‐frequency to index 0 for FFT order
        zero_idx = int(np.argmin(np.abs(kx)))
        F = np.roll(F, -zero_idx, axis=0)
        kx = np.roll(kx, -zero_idx)
        # verify roll succeeded
        if not np.isclose(kx[0], 0.0, atol=1e-12):
            raise ValueError(
                f"_invfft roll failed: kx[0]={kx[0]} is not zero after rolling"
            )

        nkx, nky = F.shape

        # make everything broadcastable to (nkx, nky, ny, nx)
        kx_b = kx[:, None, None, None]
        ky_b = ky[None, :, None, None]
        x_b = x[None, None, :, :]
        y_b = y[None, None, :, :]
        F_b = F[:, :, None, None]

        # compute phase and real/imag parts
        phase = (kx_b * x_b + ky_b * y_b) * xk_factor
        Re = np.real(F_b)
        Im = np.imag(F_b)

        mult = np.ones((nkx, nky, 1, 1), float)
        mult[:, 1:, 0, 0] = 2.0

        # sum over all modes → (ny, nx)
        f_xy = np.sum(mult * (Re * np.cos(phase) - Im * np.sin(phase)), axis=(0, 1))
        return f_xy
