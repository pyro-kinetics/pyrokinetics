from typing import Dict, Any
from ..typing import PathLike
from .EquilibriumReader import EquilibriumReader

# Can't use xarray, as TRANSP has a variable called X which itself has a dimension called X
import netCDF4 as nc
import numpy as np
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
    RBFInterpolator,
)


class EquilibriumReaderTRANSP(EquilibriumReader):
    def read(
        self,
        filename: PathLike,
        time_index: int = -1,
        time: float = None,
        nr=None,
        nz=None,
        Rmin=None,
        Rmax=None,
        Zmin=None,
        Zmax=None,
    ) -> Dict[str, Any]:
        """

        Read in TRANSP netCDF and populates Equilibrium object

        """

        with nc.Dataset(filename) as data:

            nradial = len(data["XB"][-1, :])

            time_cdf = data["TIME3"][:]

            if time_index != -1 and time is not None:
                raise ValueError("Cannot set both `time` and `time_index`")

            if time is not None:
                time_index = np.argmin(np.abs(time_cdf - time))

            R_axis = data["RAXIS"][time_index] * 1e-2
            Z_axis = data["YAXIS"][time_index] * 1e-2

            ntheta = 256
            theta = np.linspace(0, 2 * np.pi, ntheta)
            theta = theta[:, np.newaxis]

            # No. of moments stored in TRANSP
            nmoments = 17

            # Calculate flux surfaces from moments up to 17
            R_mom_cos = np.empty((nmoments, ntheta, nradial))
            R_mom_sin = np.empty((nmoments, ntheta, nradial))
            Z_mom_cos = np.empty((nmoments, ntheta, nradial))
            Z_mom_sin = np.empty((nmoments, ntheta, nradial))

            for i in range(nmoments):
                try:
                    R_mom_cos[i, :, :] = (
                        np.cos(i * theta) * data[f"RMC{i:02d}"][time_index, :] * 1e-2
                    )
                except IndexError:
                    break
                Z_mom_cos[i, :, :] = (
                    np.cos(i * theta) * data[f"YMC{i:02d}"][time_index, :] * 1e-2
                )

                # TRANSP doesn't stored 0th sin moment = 0.0 by defn
                if i == 0:
                    R_mom_sin[i, :, :] = 0.0
                    Z_mom_sin[i, :, :] = 0.0
                else:
                    R_mom_sin[i, :, :] = (
                        np.sin(i * theta) * data[f"RMS{i:02d}"][time_index, :] * 1e-2
                    )
                    Z_mom_sin[i, :, :] = (
                        np.sin(i * theta) * data[f"YMS{i:02d}"][time_index, :] * 1e-2
                    )

            Rsur = np.sum(R_mom_cos, axis=0) + np.sum(R_mom_sin, axis=0)
            Zsur = np.sum(Z_mom_cos, axis=0) + np.sum(Z_mom_sin, axis=0)

            psi_axis = data["PSI0_TR"][time_index]
            psi_bdry = data["PLFLXA"][time_index]

            current = data["PCUR"][time_index]

            # Load in 1D profiles
            q = data["Q"][time_index, :]
            press = data["PMHD_IN"][time_index, :]

            # F is on a different grid and need to remove duplicated HFS points
            psi_rmajm = data["PLFMP"][time_index, :]
            rmajm_ax = np.argmin(psi_rmajm)
            psi_n_rmajm = psi_rmajm[rmajm_ax:] / psi_rmajm[-1]

            # f = (Bt / |B|) * |B| *  R
            f = (
                data["FBTX"][time_index, rmajm_ax:]
                * data["BTX"][time_index, rmajm_ax:]
                * data["RMAJM"][time_index, rmajm_ax:]
                * 1e-2
            )

            psi = data["PLFLX"][time_index, :]
            psi_n = psi / psi[-1]

            rbdry = Rsur[:, -1]
            zbdry = Zsur[:, -1]

            # Default to netCDF values if None
            if Rmin is None:
                Rmin = min(rbdry)

            if Rmax is None:
                Rmax = max(rbdry)

            if Zmin is None:
                Zmin = min(zbdry)

            if Zmax is None:
                Zmax = max(zbdry)

            if nr is None:
                nr = nradial

            if nz is None:
                nz = nradial

            Rgrid = np.linspace(Rmin, Rmax, nr)
            Zgrid = np.linspace(Zmin, Zmax, nz)

            # Set up 1D profiles

            # Using interpolated splines
            q_interp = InterpolatedUnivariateSpline(psi_n, q)
            press_interp = InterpolatedUnivariateSpline(psi_n, press)
            f_interp = InterpolatedUnivariateSpline(psi_n_rmajm, f)
            f2_interp = InterpolatedUnivariateSpline(psi_n_rmajm, f**2)
            ffprime_interp = f2_interp.derivative()

            # Set up 2D profiles
            # Re-map from R(theta, psi), Z (theta, psi) to psi(R, Z)
            Rmesh, Zmesh = np.meshgrid(Rgrid, Zgrid)
            Rmesh_flat = np.ravel(Rmesh)
            Zmesh_flat = np.ravel(Zmesh)

            Rflat = np.ravel(Rsur.T)
            Zflat = np.ravel(Zsur.T)
            psiflat = np.repeat(psi, ntheta)

            RZflat = np.stack([Rflat, Zflat], -1)
            RZmesh_flat = np.stack([Rmesh_flat, Zmesh_flat], -1)

            # Interpolate using flat data
            psiRZ_interp = RBFInterpolator(RZflat, psiflat, kernel="cubic", neighbors=5)

            # Map data to new grid and reshape
            psiRZ_data = np.reshape(psiRZ_interp(RZmesh_flat), np.shape(Rmesh)).T

            # Set up geometric factors
            rho = (np.max(Rsur, axis=0) - np.min(Rsur, axis=0)) / 2
            R_major = (np.max(Rsur, axis=0) + np.min(Rsur, axis=0)) / 2
            a_minor = rho[-1]
            rho = rho / rho[-1]
            R_major[0] = R_major[1] + psi_n[1] * (R_major[2] - R_major[1]) / (
                psi_n[2] - psi_n[1]
            )

            # return data for Equilibrium object
            return {
                "R_axis": R_axis,
                "Z_axis": Z_axis,
                "psi_axis": psi_axis,
                "psi_bdry": psi_bdry,
                "current": current,
                "a_minor": a_minor,
                "f_psi": f_interp,
                "ff_prime": ffprime_interp,
                "q": q_interp,
                "pressure": press_interp,
                "p_prime": press_interp.derivative(),
                "rho": InterpolatedUnivariateSpline(psi_n, rho),
                "R_major": InterpolatedUnivariateSpline(psi_n, R_major),
                "R": Rgrid,
                "Z": Zgrid,
                "psi_RZ": RectBivariateSpline(Rgrid, Zgrid, psiRZ_data),
                "lcfs_R": rbdry,
                "lcfs_Z": zbdry,
            }

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a TRANSP file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            data = nc.Dataset(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"EquilibriumReaderTRANSP could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"EquilibriumReaderTRANSP must be provided a NetCDF, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the attribute TRANSP_version
        try:
            data.TRANSP_version
        except AttributeError:
            # Failing this, check for s subset of expected data_vars
            var_names = ["TIME3", "XB", "RAXIS", "YAXIS", "PSI0_TR", "PLFLXA", "PLFMP"]
            if not np.all(np.isin(var_names, list(data.variables))):
                raise ValueError(
                    f"EquilibriumReaderTRANSP was provided an invalid NetCDF: {filename}"
                )
        finally:
            data.close()
