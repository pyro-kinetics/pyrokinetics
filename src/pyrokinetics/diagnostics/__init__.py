import warnings

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs

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
        theta_even = np.linspace(-theta_max, theta_max, ntheta)

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
