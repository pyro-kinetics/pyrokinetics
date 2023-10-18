from typing import Optional

# Can't use xarray, as TRANSP has a variable called X which itself has a dimension
# called X
import numpy as np
from scipy.interpolate import RBFInterpolator
from pygacode import expro

from ..file_utils import FileReader
from ..typing import PathLike
from ..units import ureg as units, UnitSpline
from .equilibrium import Equilibrium


class EquilibriumReaderTRANSP(FileReader, file_type="GACODE", reads=Equilibrium):
    r"""
    Class that can read input.gacode files. Rather than creating instances of this
    class directly, users are recommended to use the function `read_equilibrium`.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an `Equilibrium`.
    """

    def read_from_file(
        self,
        filename: PathLike,
        nR: Optional[int] = None,
        nZ: Optional[int] = None,
        clockwise_phi: bool = True,
        cocos: Optional[int] = None,
        neighbors: Optional[int] = 64,
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
        neighbors: Optional[int]
            Sets number of nearest neighbours to use when performing the interpolation
            to flux surfaces to R,Z. By default, this is 16

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
        len_units = units.meter
        psi_units = units.weber / units.radian

        # Open data file, get generic data
        expro.expro_read(filename, 0)

        psi = expro.expro_polflux * psi_units
        B_0 = expro.expro_bcentr * units.tesla
        Bt_sign = np.sign(expro.expro_bcentr)
        F = expro.expro_fpol * units.tesla * units.meter
        FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)
        p_input = expro.expro_ptot * units.pascal
        p_spline = UnitSpline(psi, p_input)
        p = p_spline(psi)
        p_prime = p_spline(psi, derivative=1)
        q = expro.expro_q * units.dimensionless

        # z_mid can be obtained using "YMPA" and "YAXIS"
        Z_mid = expro.expro_zmag * len_units
        R_major = expro.expro_rmaj * len_units
        r_minor = expro.expro_rmin * len_units

        ntheta = 256
        theta = np.linspace(0, 2 * np.pi, ntheta)
        Z_surface = np.outer(expro.expro_zmag[1:], np.ones(ntheta)) + np.outer(
            expro.expro_kappa[1:] * expro.expro_rmin[1:], np.sin(theta)
        )

        # Reconstruct thetaR (same as MXH)
        thetaR = np.outer(np.ones(len(R_major)), theta)
        # Add moments 1 to 6
        for mom in range(0, 7):
            c = np.cos(mom * theta)
            s = np.sin(mom * theta)
            thetaR += np.outer(getattr(expro, f"expro_shape_cos{mom}"), c)
            if mom == 0:
                continue
            elif mom == 1:
                x = np.arcsin(expro.expro_delta)
                thetaR += np.outer(x, s)
            elif mom == 2:
                thetaR += np.outer(-expro.expro_zeta, s)
            else:
                thetaR += np.outer(getattr(expro, f"expro_shape_sin{mom}"), s)

        R_surface = np.outer(expro.expro_rmaj[1:], np.ones(ntheta)) + np.outer(
            expro.expro_rmin[1:], np.ones(ntheta)
        ) * np.cos(thetaR[1:])

        # Combine arrays into shape (nradial*ntheta, 2), such that [i,0] is the
        # major radius and [i,1] is the vertical position of coordinate i.
        surface_coords = np.stack((R_surface.ravel(), Z_surface.ravel()), -1)
        # Get psi at each of these coordinates. Discard the value on the mag. axis.
        surface_psi = np.repeat(psi[1:].magnitude, ntheta)

        # Add in magnetic axis
        surface_psi = np.append(surface_psi, psi[0].m)
        surface_coords = np.append(surface_coords, [[R_major[0].m, Z_mid[0].m]], axis=0)

        # Create interpolator we can use to interpolate to RZ grid.
        psi_interp = RBFInterpolator(
            surface_coords,
            surface_psi,
            kernel="cubic",
            neighbors=neighbors,
        )

        # Convert to RZ grid.
        # Lengths are the same as the netCDF radial grid if nr, nz not provided.
        nR = R_surface.shape[0] if nR is None else int(nR)
        nZ = R_surface.shape[0] if nZ is None else int(nZ)
        R = np.linspace(min(R_surface[-1, :]), max(R_surface[-1, :]), nR)
        Z = np.linspace(min(Z_surface[-1, :]), max(Z_surface[-1, :]), nZ)
        RZ_coords = np.stack([x.ravel() for x in np.meshgrid(R, Z)], -1)

        try:
            psi_RZ = psi_interp(RZ_coords).reshape((nZ, nR)).T
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                raise ValueError(
                    "Interpolation resulted in singular matrix. Try increasing number of nearest neighbors in "
                    "eq_kwargs"
                )
            else:
                raise

        I_p = expro.expro_current * units.ampere
        psi_lcfs = psi[-1]

        return Equilibrium(
            R=R * units.meter,
            Z=Z * units.meter,
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
            psi_lcfs=psi_lcfs,
            a_minor=r_minor[-1],
            B_0=B_0,
            I_p=I_p,
            clockwise_phi=clockwise_phi,
            cocos=cocos,
            eq_type="GACODE",
        )

    def verify_file_type(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GACODE file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            expro.expro_read(filename, 0)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderGACODE could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"KineticsReadeGACODE must be provided a GACODE file, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the attribute TRANSP_version
        try:
            expro.expro_name
        except AttributeError:
            raise ValueError(
                f"KineticsReaderGACODE was not able to read {filename} using pygacode"
            )
