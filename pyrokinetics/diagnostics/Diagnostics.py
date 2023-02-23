import numpy as np
from scipy.interpolate import RectBivariateSpline
import xrft

from ..pyro import Pyro


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
        nskip = len(geo.theta) // ntheta
        bmag = np.sqrt((1 / geo.R) ** 2 + geo.b_poloidal**2)
        bmag = np.roll(bmag[::nskip], ntheta // 2)
        jacob = geo.jacob * geo.dpsidr * geo.get("bunit_over_b0", 1)
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

        if use_invfft:
            for iturn in range(nturns):
                for ith in range(0, ntheta - 1, 2):
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

                    xmid = x + 2 * np.pi / ntheta * dbx * jacob[ith]
                    ymid = y + 2 * np.pi / ntheta * dby * jacob[ith]

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

                    x = x + 4 * np.pi / ntheta * dbx * jacob[ith + 1]
                    y = y + 4 * np.pi / ntheta * dby * jacob[ith + 1]

                    y = np.where(y < ymin, ymax - (ymin - y), y)

                    y = np.where(y > ymax, ymin + (y - ymax), y)

                y = y + np.mod(fac1 * ((x - xmin) / Lx * dq + qmin), Ly)
                y = np.where(y > ymax, ymin + (y - ymax), y)

                points[0, iturn, :, :] = x
                points[1, iturn, :, :] = y

        else:
            for iturn in range(nturns):
                for ith in range(0, ntheta - 1, 2):
                    dby = By[ith](x, y, grid=False) * bmag[ith] * fac2
                    dbx = Bx[ith](x, y, grid=False) * bmag[ith] * fac2

                    xmid = x + 2 * np.pi / ntheta * dbx * jacob[ith]
                    ymid = y + 2 * np.pi / ntheta * dby * jacob[ith]

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
