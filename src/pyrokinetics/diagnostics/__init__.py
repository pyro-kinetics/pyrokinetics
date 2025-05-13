import numpy as np
import xarray as xr
import xrft
from numpy.typing import ArrayLike
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import eigs

from ..pyro import Pyro
from ..units import ureg
from .synthetic_highk_dbs import SyntheticHighkDBS


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

    def new_compute_half_displacement(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 1.0,
        unwrap: bool = False,
        max_fraction: float = 0.25,
        pad_factor: int = 1,
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
           Steps with |Δx| > max_fraction * Lx issue a warning but are retained.
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
        )

        displacement = xarray - points[0, 0, :, :]

        return displacement

    def poincare(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 1.0,
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
           Steps with |Δx| > max_fraction * Lx issue a warning but are retained.
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

    def follow_field_line(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        use_invfft: bool = False,
        smoothing: float = 1.0,
        unwrap: bool = False,
        theta_min: float = None,
        theta_max: float = None,
        max_fraction: float = 0.25,
        pad_factor: int = 1,
        integration_order=4,
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
           Steps with |Δx| > max_fraction * Lx issue a warning but are retained.
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
        kx = apar.kx.values
        ky = apar.ky.values

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

        rhostar *= k_units * self.pyro.norms.lref

        # Define domain sizes
        Ly = 2 * np.pi / dky / k_units
        Lx = 2 * np.pi / dkx / k_units
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
                    xgrid.m,
                    ygrid.m,
                    byfft.sel(theta=theta, method="nearest"),
                    kx=5,
                    ky=5,
                    s=smoothing,
                )
                for theta in byfft.theta
            ]
            Bx = [
                RectBivariateSpline(
                    xgrid.m,
                    ygrid.m,
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

        points = np.empty((2, nturns, len(yarray), len(xarray))) * x_units

        def eval_dx_dy(theta_idx, x_in, y_in):
            if use_invfft:
                dBx = self._invfft(ikyapar[:, :, theta_idx], x_in, y_in, kx, ky)
                dBy = self._invfft(ikxapar[:, :, theta_idx], x_in, y_in, kx, ky)
            else:
                dBx = Bx[theta_idx](x_in, y_in, grid=False)
                dBy = By[theta_idx](x_in, y_in, grid=False)

            dBx *= apar_units * k_units * l_units
            dBy *= apar_units * k_units * l_units

            dBx = dBx.to(x_units * b_units, self.pyro.norms.context)
            dBy = dBy.to(x_units * b_units, self.pyro.norms.context)

            dx = (dBx / b_units).to(x_units, self.pyro.norms.context)
            dy = (dBy / b_units).to(x_units, self.pyro.norms.context)

            if np.any(dx > max_jump):
                raise RuntimeError(
                    f"dx step {dx:.3e} > {max_jump:.3e}; "
                    "increase ntheta or decrease pad_factor"
                )

            return dx, dy

        # Work on a staggered grid using halfway point for full integral
        for iturn in range(nturns):
            for ith in range(0, ntheta - 1):
                dtheta = dtheta_grid[ith]

                # RK1 step (Euler) if you want simplest version
                if integration_order == 1:
                    dx, dy = eval_dx_dy(ith, x, y)
                    x = x + dtheta * dx
                    y = y + dtheta * dy

                # RK2 method (Improved Euler)
                elif integration_order == 2:
                    # First stage (predictor)
                    k1x, k1y = eval_dx_dy(ith, x, y)
                    k1x = dtheta * k1x
                    k1y = dtheta * k1y
                    # Second stage (corrector)
                    k2x, k2y = eval_dx_dy(ith + 1, x + k1x, y + k1y)
                    k2x = dtheta * k2x
                    k2y = dtheta * k2y
                    # Update x and y with the average
                    x = x + 0.5 * (k1x + k2x)
                    y = y + 0.5 * (k1y + k2y)

                elif integration_order == 3:
                    # Should k2x technically be on ith + 1/2?
                    k1x, k1y = eval_dx_dy(ith, x, y)
                    k2x, k2y = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )
                    k3x, k3y = eval_dx_dy(
                        ith + 1,
                        x - dtheta * k1x + 2 * dtheta * k2x,
                        y - dtheta * k1y + 2 * dtheta * k2y,
                    )
                    x = x + dtheta * (k1x + 4 * k2x + k3x) / 6
                    y = y + dtheta * (k1y + 4 * k2y + k3y) / 6

                elif integration_order == 4:
                    # Should k2x and k3x technically be on ith + 1/2?
                    k1x, k1y = eval_dx_dy(ith, x, y)
                    k2x, k2y = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k1x, y + 0.5 * dtheta * k1y
                    )
                    k3x, k3y = eval_dx_dy(
                        ith, x + 0.5 * dtheta * k2x, y + 0.5 * dtheta * k2y
                    )
                    k4x, k4y = eval_dx_dy(ith + 1, x + dtheta * k3x, y + dtheta * k3y)
                    x = x + dtheta * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
                    y = y + dtheta * (k1y + 2 * k2y + 2 * k3y + k4y) / 6

                else:
                    raise ValueError("Only RK order 1, 3, or 4 supported.")

                # Apply periodic boundaries on x only if not unwrapping
                if not unwrap:
                    x = xmin + np.mod(x - xmin, Lx)

            # Always apply the q-profile twist to y
            y = y + twist_prefactor * (qmin + (x - xmin) / Lx * delta_q)

            # Always wrap y back into [ymin, ymin + Ly)
            y = ymin + np.mod(y - ymin, Ly)

            points[0, iturn, :, :] = x
            points[1, iturn, :, :] = y

        return points

    @staticmethod
    def _invfft(F, x, y, kx, ky):
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
        phase = kx_b * x_b + ky_b * y_b
        Re = np.real(F_b)
        Im = np.imag(F_b)

        mult = np.ones((nkx, nky, 1, 1), float)
        mult[:, 1:, 0, 0] = 2.0

        # sum over all modes → (ny, nx)
        f_xy = np.sum(mult * (Re * np.cos(phase) - Im * np.sin(phase)), axis=(0, 1))
        return f_xy

    def radial_diffusion_coefficient(
        self,
        xarray: ArrayLike,
        yarray: ArrayLike,
        nturns: int,
        time: float,
        rhostar: float,
        length_per_turn: float = None,
        use_invfft: bool = False,
        smoothing: float = 1.0,
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

        # Debug prints for checking intermediate values:
        print("l_total =", length_total)
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

    def compute_half_disp(
        self,
        time: float,
        max_fraction: float = 0.25,
        pad_factor: int = 2,
    ):
        """
        Returns the radial displacement of a magnetic field line
        after half poloidal turn, which is used to investigate the
        occurance of a nonzonal transition. See Pueschel et al.
        Phys. Rev. Lett. 110, 155005 (2013) for details.
        This routine may take a while.

        This routine steps each field line
        through ntheta/2 segments, applies small RK‐like updates via direct FFT-
        summed B-field increments, and wraps y into the periodic domain. Outliers
        exceeding a maximum allowed jump are warned but still included in the average.

        Available for CGYRO, GENE and GS2 nonlinear simulations

        You need to load the output files of a simulation
        berore calling this function.

        ----------
        time : float
           Simulation time at which to sample the parallel vector potential `apar`.
           Must match a time index in `self.pyro.gk_output["apar"]`.
        max_fraction : float, optional
           Maximum allowed x‐step size as a fraction of the full radial domain Lx.
           Steps with |Δx| > max_fraction * Lx issue a warning but are retained.
           Default is 0.25.
        pad_factor : int, optional
           Factor by which to pad the kx spectrum before transforming.
           A value of 2 doubles the kx resolution (nx → 2·nx). Larger
           values increase real-space grid resolution but cost more CPU.
           Default is 2.

        Returns
        -------
        displacement: ArrayLike, 2D array of shape (nx, ny) containing the
               displacement of each magnetic filed line starting at (x, y).
        """

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
        xgrid = np.linspace(-Lx / 2, Lx / 2, pad_nkx, endpoint=False)
        ny = 2 * (len(ky) - 1)
        ygrid = np.linspace(-Ly / 2, Ly / 2, ny, endpoint=False)

        # --- geometry factors ---
        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2 * np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)

        ntheta = apar.theta.size
        nskip = len(geo.theta) // ntheta

        bmag = np.sqrt((1 / geo.R.m) ** 2 + geo.b_poloidal.m**2)
        bmag = np.roll(bmag[::nskip], ntheta // 2)

        jacob = self.pyro.metric_terms.Jacobian.m * geo.dpsidr.m * geo.bunit_over_b0.m
        jacob = np.roll(jacob[::nskip], ntheta // 2)

        fac = geo.dpsidr.m * geo.q.m / geo.rho.m

        # precompute Fourier coefficients
        ikxapar = 1j * apar.kx * apar
        ikyapar = -1j * apar.ky * apar
        ikxapar = ikxapar.transpose("kx", "ky", "theta").values
        ikyapar = ikyapar.transpose("kx", "ky", "theta").values

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

                    dBy = (
                        self._invfft(ikxapar[:, :, ith], xx, yy, kx, ky)[0, 0]
                        * bmag[ith]
                        * fac
                    )
                    dBx = (
                        self._invfft(ikyapar[:, :, ith], xx, yy, kx, ky)[0, 0]
                        * bmag[ith]
                        * fac
                    )

                    dx_step = abs(dBx * dtheta * jacob[ith])
                    if dx_step > max_jump:
                        raise RuntimeError(
                            f"dx step {dx_step:.3e} > {max_jump:.3e}; "
                            "increase ntheta or decrease pad_factor"
                        )

                    x += dtheta * dBx * jacob[ith]
                    y += dtheta * dBy * jacob[ith]
                    # poloidal wrap
                    y = ((y + Ly / 2) % Ly) - Ly / 2

                    # --- second sub‐step ---
                    idx2 = ith + 1
                    xx = np.array([[x]], dtype=float)
                    yy = np.array([[y]], dtype=float)

                    dBy = (
                        self._invfft(ikxapar[:, :, idx2], xx, yy, kx, ky)[0, 0]
                        * bmag[idx2]
                        * fac
                    )
                    dBx = (
                        self._invfft(ikyapar[:, :, idx2], xx, yy, kx, ky)[0, 0]
                        * bmag[idx2]
                        * fac
                    )

                    dx_step = abs(dBx * dtheta * jacob[idx2])
                    if dx_step > max_jump:
                        raise RuntimeError(
                            f"dx step {dx_step:.3e} > {max_jump:.3e}; "
                            "increase ntheta or decrease pad_factor"
                        )

                    x += dtheta * dBx * jacob[idx2]
                    y += dtheta * dBy * jacob[idx2]
                    y = ((y + Ly / 2) % Ly) - Ly / 2

                disp[ix, iy] = abs(x - x0)

        return disp

    def compute_half_disp_fast(self, time: float, max_fraction: float = 0.25):
        """
        Fast estimation of mean radial displacement ⟨δr⟩ of a set of magnetic field lines
        after half poloidal turn. This routine downsamples x, y, and theta for speed.
        Suitable for trend analysis.

        Parameters
        ----------
        time : float
            Simulation time at which to sample the parallel vector potential `apar`.
            Must match a time index in `self.pyro.gk_output["apar"]`.
        max_fraction : float, optional
            Maximum allowed x‐step size as a fraction of the full radial domain Lx.
            Steps with |Δx| > max_fraction * Lx issue a warning but are retained.
            Default is 0.25.

        Returns
        -------
        float
            The mean absolute radial displacement ⟨|x_final – x_initial|⟩ computed
            over an 8×5 grid of initial (x0,y0) positions after half a poloidal turn.

        Notes
        -----
        - Theta resolution is downsampled by a factor of 4 to accelerate the map.
        - Uses `self._invfft` to reconstruct Bx, By at each (x,y) on-the-fly.
        - Applies periodic wrapping in y but accumulates x displacements unwrapped.
        - Suitable for rapid trend analysis; not for high-precision Poincaré plots.
        """

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

        xgrid = np.linspace(-Lx / 2, Lx / 2, 8)
        ygrid = np.linspace(-Ly / 2, Ly / 2, 5)

        geo = self.pyro.local_geometry
        theta_metric = np.linspace(0, 2 * np.pi, 256)
        self.pyro.load_metric_terms(theta=theta_metric)
        nskip = len(geo.theta) // ntheta

        bmag = np.sqrt((1 / geo.R.m) ** 2 + geo.b_poloidal.m**2)
        bmag = np.roll(bmag[::nskip], ntheta // 2)

        jacob = self.pyro.metric_terms.Jacobian.m * geo.dpsidr.m * geo.bunit_over_b0.m
        jacob = np.roll(jacob[::nskip], ntheta // 2)

        fac = geo.dpsidr.m * geo.q.m / geo.rho.m

        ikxapar = 1j * apar.kx * apar
        ikyapar = -1j * apar.ky * apar
        ikxapar = ikxapar.transpose("kx", "ky", "theta").values
        ikyapar = ikyapar.transpose("kx", "ky", "theta").values

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
                        dBy = (
                            self._invfft(ikxapar[:, :, idx], xx, yy, kx, ky)[0, 0].real
                            * bmag[idx]
                            * fac
                        )
                        dBx = (
                            self._invfft(ikyapar[:, :, idx], xx, yy, kx, ky)[0, 0].real
                            * bmag[idx]
                            * fac
                        )
                        dx = dtheta * dBx * jacob[idx]
                        dy = dtheta * dBy * jacob[idx]
                        if abs(dx) > max_jump:
                            print(f"[warn] dx step = {dx:.3e} exceeds {max_jump:.3e}")
                        x += dx
                        y = ((y + dy + Ly / 2) % Ly) - Ly / 2
                disp_vals.append(abs(x - x0))

        return np.mean(disp_vals)

    def compute_corr_length(
        self,
        time: float,
        yarray: ArrayLike,
        Nx: int = 64,
        ndelta: int = None,
    ):
        """
        Compute the radial correlation length λ_x at each y using the Wiener–Khinchin theorem.

        For each y in `yarray` and each poloidal angle θ, this method:
        1. Reconstructs b(x) = ∂y A_par via direct Fourier‐mode summation using `self._invfft`.
        2. Computes the power spectrum P(k) = |FFT[b(x)]|².
        3. Obtains the autocorrelation C(Δ) = IFFT[P(k)], normalized so C(0)=1.
        4. Identifies λ_x(θ,y) as the smallest Δ where C(Δ) = 1/e.
        Finally, λ_x(y) is taken as the mean over θ.

        Parameters
        ----------
        time : float
            Simulation time at which to select the parallel vector potential `apar`.
        yarray : ArrayLike
            1D array of y positions at which to compute the correlation length.
        Nx : int, optional
            Number of real‐space x grid points used for the FFT. Default is 64.
        ndelta : int, optional
            (Unused) originally intended to limit the Δ‐search range; retained for
            API compatibility.

        Returns
        -------
        lam_y : ArrayLike
            1D array of length len(yarray) giving the mean radial correlation length
            λ_x at each specified y position.

        Notes
        -----
        - Assumes A_par half‐spectrum in ky (ky ≥ 0) is available in `apar`.
        - Uses `np.fft.fft`/`ifft` for the Wiener–Khinchin calculation.
        - Suitable for rapid estimates of radial correlation length; not optimized
          for very high precision.
        """

        # 1) load & dequantify A_par
        apar = (
            self.pyro.gk_output["apar"]
            .sel(time=time, method="nearest")
            .pint.dequantify()
            .transpose("theta", "kx", "ky")
        )

        kx = apar.kx.values
        ky = apar.ky.values
        ntheta = apar.theta.size

        # 2) real-space grid
        dkx = kx[1] - kx[0]
        Lx = 2 * np.pi / dkx
        x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
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
                    A_k * coeff[None, :], x[None, :], np.array([[y]]), kx, ky
                )[0].real

                # Wiener–Khinchin correlation
                Pk = np.abs(np.fft.fft(b_x)) ** 2
                C = np.fft.ifft(Pk).real
                C /= C[0]

                below = np.where(C < 1 / np.e)[0]
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
