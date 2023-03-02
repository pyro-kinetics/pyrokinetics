from typing import Optional

# Can't use xarray, as TRANSP has a variable called X which itself has a dimension
# called X
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RBFInterpolator

from ..readers import Reader
from ..typing import PathLike
from ..units import ureg as units
from .equilibrium import Equilibrium, equilibrium_reader
from .utils import UnitSpline


@equilibrium_reader("TRANSP")
class EquilibriumReaderTRANSP(Reader):
    r"""
    Class that can read TRANSP equilibrium files. Rather than creating instances of this
    class directly, users are recommended to use the function `read_equilibrium`.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an `Equilibrium`.
    """

    def read(
        self,
        filename: PathLike,
        time: Optional[float] = None,
        time_index: Optional[int] = None,
        nR: Optional[int] = None,
        nZ: Optional[int] = None,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None,
    ) -> Equilibrium:
        r"""
        Read in TRANSP netCDF and creates Equilibrium object.

        TRANSP makes use of radial grids, and these are interpolated onto a Cartesian
        RZ grid. Additional keyword-only arguments may be provided to control the
        resolution of the Cartesian grid, and to choose the time at which the
        equilibrium is taken.

        Parameters
        ----------
        filename: PathLike
            Path to the TRANSP netCDF file.
        time: Optional[float]
            The time, in seconds, at which equilibrium data should be taken. Data will
            be drawn from the time closest to the provided value. Users should only
            provide one of ``time`` or ``time_index``. If neither is provided, data is
            drawn at the last time stamp.
        time_index: Optional[int]
            As an alternative to providing the time directly, users may provide the
            index of the desired time stamp.
        nR: Optional[int]
            The number of grid points in the major radius direction. By default, this
            is set to the number of radial grid points in the TRANSP file.
        nZ: Optional[int]
            The number of grid points in the vertical direction. By default, this
            is set to the number of radial grid points in the TRANSP file.
        clockwise_phi: bool, default False
            Determines whether the :math:`\phi` grid increases clockwise or
            anti-clockwise when viewed from above. Used to determine COCOS convention of
            the inputs.
        cocos: Optional[int]
            If set, asserts that the GEQDSK file follows that COCOS convention, and
            neither ``clockwise_phi`` nor the file contents will be used to identify
            the actual convention in use. The resulting Equilibrium is always converted
            to COCOS 11.

        Raises
        ------
        FileNotFoundError
            If ``filename`` is not a valid file.
        RuntimeError
            If the user provides both ``time`` and ``time_index``.
        ValueError
            If nr or nz are negative.
        IndexError
            If ``time_index`` is provided, but is out of bounds.

        Returns
        -------
        Equilibrium
        """
        # Define some units to be used later
        # Note that length units are in centimeters!
        # This is not consistent throughout. Pressure is in Pascal as usual, not
        # Newtons per centimeter^2. However, it does affect our units for F.
        len_units = units.cm
        psi_units = units.weber / units.radian

        if time_index is not None and time is not None:
            raise RuntimeError("Cannot set both 'time' and 'time_index'")

        with nc.Dataset(filename) as data:
            # Determine time at which we should read data
            time_cdf = data["TIME3"][:]
            if time_index is None:
                time_index = -1 if time is None else np.argmin(np.abs(time_cdf - time))

            # Determine factors on the psi grid
            # TRANSP uses several grids: 'XB', the minor radius of each flux surface
            # divided by the minor radius of the last closed flux surface; 'X', the
            # points in between 'XB' and the previous points, with an implied zero at
            # the first point, and "RMAJM", the major radius extents of each flux
            # surface. The latter includes the position at the magnetic axis, so we
            # should prefer quantities on this grid. However, we only need the values on
            # the low-field side (LFS), so we should discard the values on the
            # high-field side (HFS). We wish to have variables and their derivatives in
            # terms of psi.
            axis_idx = np.argmin(data["PLFMP"][time_index])
            rmajm = np.asarray(data["RMAJM"][time_index]) * len_units
            psi = np.asarray(data["PLFMP"][time_index, axis_idx:]) * psi_units

            # F is not given directly, so we must compute it using:
            # F = B_t * R
            # The toroidal B field is also not given directly, so this is calculated
            # using:
            # B_t = (|Bt| / |B|) * |B| * sign(B_t)
            Bt_vacuum = np.asarray(data["BTX"][time_index, axis_idx:])
            Bt_total = np.asarray(data["FBTX"][time_index, axis_idx:]) * Bt_vacuum
            Bt_sign = np.sign(data["BPHI_MSE"][time_index, 0].data)
            F = Bt_sign * Bt_total * units.tesla * rmajm[axis_idx:]
            B_0 = Bt_sign * Bt_vacuum[0] * units.tesla

            # ffprime is determined by fitting a spline and taking its derivative.
            # We'll use UnitSpline to ensure units are carried forward.
            FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)

            # Pressure is on the 'X' grid. We assume that this corresponds to the
            # pressure on each flux surface including the LCFS, but excluding the
            # magnetic axis. To determine p and p_prime, we fit a spline and extrapolate
            # to the get point on the magnetic axis. We can also use this spline to
            # take the derivative.
            # We use 'PMHD_IN', the 'pressure input to MHD solver', as the
            # plasma pressure 'PPLAS' is the thermal pressure only.
            # TODO Should we interpolate from X to XB?
            p_input = np.asarray(data["PMHD_IN"][time_index]) * units.pascal
            p_spline = UnitSpline(psi[1:], p_input)
            p = p_spline(psi)
            p_prime = p_spline(psi, derivative=1)

            # Q is given directly. We use "QMP" instead of "Q" as this includes the
            # magnetic axis
            q = np.asarray(data["QMP"][time_index, axis_idx:]) * units.dimensionless

            # R_major can be obtained by taking the average of the HFS and LFS parts
            # of "RMAJM"
            R_major = (rmajm[axis_idx:] + rmajm[axis_idx::-1]) / 2

            # r_minor can be obtained by taking the difference between the HFS and LFS
            # parts of RMAJM
            r_minor = (rmajm[axis_idx:] - rmajm[axis_idx::-1]) / 2

            # z_mid can be obtained using "YMPA" and "YAXIS"
            Z_mid = np.empty(len(psi)) * len_units
            Z_mid[0] = np.asarray(data["YAXIS"][time_index]) * len_units
            Z_mid[1:] = np.asarray(data["YMPA"][time_index]) * len_units

            # Determine r, z, psi_rz
            # We'll ignore units for now, as they can misbehave around interpolation
            # routines. They're reintroduced at the end.
            # Begin by calculating flux surfaces from moments up to 17
            # The length of the theta grid is fixed at 256.
            ntheta = 256
            theta = np.linspace(0, 2 * np.pi, ntheta)
            # Begin with the 0th moment. Surface grids have shape (nradial, ntheta).
            # There is no 0th sin moment, as it's 0 by definition.
            R_surface = np.outer(np.asarray(data["RMC00"][time_index]), np.ones(ntheta))
            Z_surface = np.outer(np.asarray(data["YMC00"][time_index]), np.ones(ntheta))
            # Add moments 1 to 17
            for mom in range(1, 17):
                # Exit early if this moment doesn't exist
                if f"RMC{mom:02d}" not in data.variables:
                    break
                c = np.cos(mom * theta)
                s = np.sin(mom * theta)
                R_surface += np.outer(np.asarray(data[f"RMC{mom:02d}"][time_index]), c)
                R_surface += np.outer(np.asarray(data[f"RMS{mom:02d}"][time_index]), s)
                Z_surface += np.outer(np.asarray(data[f"YMC{mom:02d}"][time_index]), c)
                Z_surface += np.outer(np.asarray(data[f"YMS{mom:02d}"][time_index]), s)

            # Combine arrays into shape (nradial*ntheta, 2), such that [i,0] is the
            # major radius and [i,1] is the vertical position of coordinate i.
            surface_coords = np.stack((R_surface.ravel(), Z_surface.ravel()), -1)
            # Get psi at each of these coordinates. Discard the value on the mag. axis.
            surface_psi = np.repeat(psi.magnitude[1:], ntheta)
            # Create interpolator we can use to interpolate to RZ grid.
            psi_interp = RBFInterpolator(
                surface_coords, surface_psi, kernel="cubic", neighbors=5
            )

            # Convert to RZ grid.
            # Lengths are the same as the netCDF radial grid if nr, nz not provided.
            nR = R_surface.shape[0] if nR is None else int(nR)
            nZ = R_surface.shape[0] if nZ is None else int(nZ)
            R = np.linspace(min(R_surface[-1, :]), max(R_surface[-1, :]), nR)
            Z = np.linspace(min(Z_surface[-1, :]), max(Z_surface[-1, :]), nZ)
            RZ_coords = np.stack([x.ravel() for x in np.meshgrid(R, Z)], -1)
            psi_RZ = psi_interp(RZ_coords).reshape((nZ, nR)).T

            # Get current and b_axis to determine COCOS
            # Using 'measured plasma current', PCUR. Should this instead be one of the
            # following?
            # - EQ PLASMA CURRENT, PCUREQ (seems to be zero in some files?)
            # - CALCULATED PLASMA CURRENT, PCURC
            # - TOTAL PLASMA CURRENT, CUR (units of amps per centimeter^2)
            I_p = data["PCUR"][time_index] * units.ampere

            return Equilibrium(
                R=R * units.cm,
                Z=Z * units.cm,
                psi_RZ=psi_RZ * psi_units,
                psi=psi,
                F=F,
                FF_prime=FF_prime,
                p=p,
                p_prime=p_prime,
                q=q,
                R_major=R_major,
                r_minor=r_minor,
                Z_mid=Z_mid,
                psi_lcfs=data["PLFLXA"][time_index][()] * psi_units,
                a_minor=r_minor[-1],
                B_0=B_0,
                I_p=I_p,
                clockwise_phi=clockwise_phi,
                cocos=cocos,
                eq_type="TRANSP",
            )

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a TRANSP file without processing"""
        # Try opening data file. If it doesn't exist or isn't netcdf, this will fail.
        data = nc.Dataset(filename)
        try:
            # Given it is a netcdf, check it has the attribute TRANSP_version
            data.TRANSP_version
        except AttributeError:
            # Failing this, check for a subset of expected data_vars
            var_names = ["TIME3", "XB", "RAXIS", "YAXIS", "PSI0_TR", "PLFLXA", "PLFMP"]
            if not np.all(np.isin(var_names, list(data.variables))):
                raise ValueError(
                    f"The netCDF {filename} can't be read by TRANSPEquilibriumReader"
                )
        finally:
            data.close()
