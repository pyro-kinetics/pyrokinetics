from typing import Optional

import numpy as np
from scipy.interpolate import RBFInterpolator
from path import Path

from ..file_utils import FileReader
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .equilibrium import Equilibrium

test_keys = ["q", "fpol", "polflux", "bcentr"]


def read_gacode_file(filename: PathLike):
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    data_object = GACODEProfiles()

    data_object.units = {}
    current_key = None
    data_dict = {}

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                current_key = line[1:].strip()

                if "|" in current_key:
                    split_str = current_key.split("|")
                    current_key = split_str[0].strip()
                    data_object.units[current_key] = split_str[1].strip()

                setattr(data_object, current_key, [])
                data_dict[current_key] = []
            elif current_key:
                if line:
                    # Convert data to numpy array of floats or strings
                    data = []
                    for item in line.split():
                        try:
                            data.append(float(item))
                        except ValueError:
                            data.append(item)
                    # Check if data has two columns
                    if len(data) == 1 or current_key in ["mass", "name", "z"]:
                        data_dict[current_key].extend(data)
                    else:
                        data_dict[current_key].append(data[1:])

    # Check if relevant keys exist}
    if len(set(test_keys).intersection(data_dict.keys())) != len(test_keys):
        raise ValueError("EquilibriumReaderGACODE could not find all relevant keys")

    for key, value in data_dict.items():
        # If data has two columns, convert to 2D array
        setattr(data_object, key, np.squeeze(np.array(value)))

    return data_object


class GACODEProfiles:
    def __init__(self):
        self.zeta = None
        self.delta = None
        self.kappa = None
        self.rmin = None
        self.rmaj = None
        self.zmag = None
        self.q = None
        self.ptot = None
        self.fpol = None
        self.bcentr = None
        self.polflux = None


class EquilibriumReaderGACODE(FileReader, file_type="GACODE", reads=Equilibrium):
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
        cocos: Optional[int] = 1,
        neighbors: Optional[int] = 64,
    ) -> Equilibrium:
        r"""
        Read in input.gacode file and creates Equilibrium object.

        GACODE makes use of radial grids, and these are interpolated onto a Cartesian
        RZ grid. Additional keyword-only arguments may be provided to control the
        resolution of the Cartesian grid, and to choose the time at which the
        equilibrium is taken.

        Parameters
        ----------
        filename: PathLike
            Path to the input.gacode file.
        nR: Optional[int]
            The number of grid points in the major radius direction. By default, this
            is set to the number of radial grid points in the input.gacode file.
        nZ: Optional[int]
            The number of grid points in the vertical direction. By default, this
            is set to the number of radial grid points in the input.gacode file.
        clockwise_phi: bool, default False
            Determines whether the :math:`\phi` grid increases clockwise or
            anti-clockwise when viewed from above. Used to determine COCOS convention of
            the inputs.
        cocos: Optional[int]
            If set, asserts that the GEQDSK file follows that COCOS convention, and
            neither ``clockwise_phi`` nor the file contents will be used to identify
            the actual convention in use. The resulting Equilibrium is always converted
            to COCOS 11. BY default this is 1
        neighbors: Optional[int]
            Sets number of nearest neighbours to use when performing the interpolation
            to flux surfaces to R,Z. By default, this is 64

        Raises
        ------
        ValueError
            If ``filename`` is not a valid file or if nr or nz are negative.

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

        profiles = read_gacode_file(filename)

        psi = profiles.polflux * psi_units
        B_0 = profiles.bcentr * units.tesla
        F = profiles.fpol * units.tesla * units.meter
        FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)
        p_input = profiles.ptot * units.pascal
        p_spline = UnitSpline(psi, p_input)
        p = p_spline(psi)
        p_prime = p_spline(psi, derivative=1)
        q = profiles.q * units.dimensionless

        # z_mid can be obtained using "YMPA" and "YAXIS"
        Z_mid = profiles.zmag * len_units
        R_major = profiles.rmaj * len_units
        r_minor = profiles.rmin * len_units
        ntheta = 256
        theta = np.linspace(0, 2 * np.pi, ntheta)
        Z_surface = np.outer(profiles.zmag[1:], np.ones(ntheta)) + np.outer(
            profiles.kappa[1:] * profiles.rmin[1:], np.sin(theta)
        )

        # Reconstruct thetaR (same as MXH)
        thetaR = np.outer(np.ones(len(R_major)), theta)
        # Add moments 1 to 6
        for mom in range(0, 6):
            c = np.cos(mom * theta)
            s = np.sin(mom * theta)
            thetaR += np.outer(getattr(profiles, f"shape_cos{mom}"), c)
            if mom == 0:
                continue
            elif mom == 1:
                x = np.arcsin(profiles.delta)
                thetaR += np.outer(x, s)
            elif mom == 2:
                thetaR += np.outer(-profiles.zeta, s)
            else:
                thetaR += np.outer(getattr(profiles, f"shape_sin{mom}"), s)

        R_surface = np.outer(profiles.rmaj[1:], np.ones(ntheta)) + np.outer(
            profiles.rmin[1:], np.ones(ntheta)
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

        I_p = profiles.current * units.ampere
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
        filename = Path(filename)
        if not filename.isfile():
            raise FileNotFoundError(filename)
        try:
            profiles = read_gacode_file(filename)
            profile_keys = [hasattr(profiles, prof) for prof in test_keys]
            if not np.all(profile_keys):
                raise ValueError(
                    "EquilibriumReaderGACODE could not find all relevant keys"
                )
        except ValueError:
            raise ValueError(f"EquilibriumReaderGACODE could not find {filename}")
