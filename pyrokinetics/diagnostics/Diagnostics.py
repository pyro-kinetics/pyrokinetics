import numpy as np
from scipy.interpolate import RectBivariateSpline

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
    def __init__(self, pyro : Pyro):
        if pyro.gk_output is None:
            raise RuntimeError(
                "Diagnostics: Please load gk output files (Pyro.load_gk_output)"
                " before using any diagnostics")
        self.pyro = pyro

    def poincare(self,
                 xarray: np.ndarray,
                 yarray: np.ndarray,
                 nturns: int,
                 time: float,
                 rhostar: float):
        """
        Generate a poincare map. It returns the (x, y) coordinates of
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

        Returns
        -------
        coordinates: numpy.ndarray, 2D array containing the x and y coordinates
                     of the Poincare map

        Raises
        ------
        NotImplementedError: if `gk_code` is not `CGYRO`, `GENE` or `GS2`
        RuntimeError: in case of linear simulation
        """
        if self.pyro.gk_code != "CGYRO" and self.pyro.gk_code != "GS2" and self.pyro.gk_code != "GENE":
            raise NotImplementedError(
                "Poincare map only available for CGYRO, GENE and GS2"
            )
        if self.gk_input.is_linear():
            raise RuntimeError(
                "Poincare only available for nonlinear runs"
            )
        apar = self.pyro.gk_output.fields.sel(field="apar").sel(time=time, method="nearest")
        kymin = apar.ky.values[1]
        shift = np.argmin(np.abs(apar.kx.values))
        apar = apar.roll(kx=shift, roll_coords=True)
        kx = apar.kx.values
        ky = apar.ky.values
        ntheta = apar.theta.shape[0]
        nkx = kx.shape[0]
        nky = ky.shape[0]
        dkx = kx[1] - kx[0]
        dky = kymin
        ny = 2 * (nky - 1)
        nkx0 = nkx + 1 - np.mod(nkx, 2)
        Lx = 2 * np.pi / dkx
        Ly = 2 * np.pi / dky
        xgrid = np.linspace(-Lx / 2, Lx / 2, nkx0)[:nkx]
        ygrid = np.linspace(-Ly / 2, Ly / 2, ny)
        xmin = np.min(xgrid)
        ymin = np.min(ygrid)
        ymax = np.max(ygrid)

        npoints = nturns * xarray.shape[0] * yarray.shape[0]
        points = np.empty((2, npoints))

        # Geometrical factors
        geo = self.pyro.local_geometry
        bmag = np.roll(np.sqrt((1 / geo.R) ** 2 + geo.b_poloidal**2), ntheta // 2)
        jacob = np.roll(
            geo.jacob * geo.dpsidr * geo.get("bunit_over_b0", 1), ntheta // 2
        )
        dq = rhostar * Lx * geo.shat / geo.dpsidr
        qmin = geo.q - dq / 2
        fac1 = 2 * np.pi * geo.dpsidr / rhostar
        fac2 = geo.dpsidr * geo.q / geo.rho

        # Fourier domain
        Ky, Kx = np.meshgrid(ky, kx)
        Apar = np.transpose(apar.values,
                            axes=(
                                apar.get_axis_num("kx"),
                                apar.get_axis_num("ky"),
                                apar.get_axis_num("theta")
                            ))
        By = []
        Bx = []
        for ith in range(ntheta):
            ikxapar = 1j * Kx * Apar[:, :, ith]
            ikyapar = -1j * Ky * Apar[:, :, ith]
            byfft = np.fft.fftshift(
                np.fft.irfft2(ikxapar, norm="forward"),
                axes=(0, 1))
            bxfft = np.fft.fftshift(
                np.fft.irfft2(ikyapar, norm="forward"),
                axes=(0, 1))
            By.append(RectBivariateSpline(xgrid, ygrid, byfft))
            Bx.append(RectBivariateSpline(xgrid, ygrid, bxfft))

        # Main loop
        j = 0
        for x0 in xarray:
            for y0 in yarray:
                x = x0
                y = y0
                for iturn in range(nturns):
                    for ith in range(0, ntheta - 1, 2):
                        dby = By[ith](x, y)
                        dbx = Bx[ith](x, y)

                        dbx = bmag[ith] * dbx * fac2
                        dby = bmag[ith] * dby * fac2

                        xmid = x + 2 * np.pi / ntheta * dbx * jacob[ith]
                        ymid = y + 2 * np.pi / ntheta * dby * jacob[ith]

                        dby = By[ith](xmid, ymid)
                        dbx = Bx[ith](xmid, ymid)

                        dbx = bmag[ith + 1] * dbx * fac2
                        dby = bmag[ith + 1] * dby * fac2

                        x = x + 4 * np.pi / ntheta * dbx * jacob[ith + 1]
                        y = y + 4 * np.pi / ntheta * dby * jacob[ith + 1]

                        if y < ymin:
                            y = ymax - (ymin - y)
                        if y > ymax:
                            y = ymin + (y - ymax)
                    y = y + np.mod(fac1 * ((x - xmin) / Lx * dq + qmin), Ly)
                    if y > ymax:
                        y = ymin + (y - ymax)
                    points[0, j] = x
                    points[1, j] = y
                    j = j + 1
        return points