from typing import Optional
from ..typing import PathLike
from copy import deepcopy
import numpy as np

from .EquilibriumReader import equilibrium_readers
from .get_flux_surface import get_flux_surface


class Equilibrium:
    """
    Defines information about Tokamak plasma equilibrium.
    Partially adapted from equilibrium option developed by B. Dudson in FreeGS

    Contains the attributes:

    - supported_equilibrium_types: A list of all supported equilibrium input types.
    - eq_files: Stored reference of the last file read
    - eq_type: Stored reference of the last equilibrium type. May be inferred.
    - nr: No. of radial points in the equilibrium file
    - nz: No. of vertical point in the equilibrium file
    - R: Linearly spaced gridpoints in the radial direction (m)
    - Z: Linearly spaced gridpoints in the vertical direction (m)
    - nr: Number of gridpoints in the R direction
    - nz: Number of gridpoints in the Z direction
    - psi_axis: Poloidal flux on axis (Wb/m2)
    - psi_bdry: Poloidal flux at the plasma boundary (Wb/m2)
    - R_axis: Major radius of magnetic axis (m)
    - Z_axis: Vertical position of magnetic axis (m)
    - lcfs_R: (array) Radial positions of last closed flux surface (m)
    - lcfs_Z: (array) Vertical positions of last closed flus surface (m)
    - a_minor: Minor radius (m)
    - bcentr: Magnetic field at magnetic axis (GEQDSK only) (T)
    - current: Total plasma current (TRANSP ONLY) (A)

    Contains the functions:

    - get_b_radial: Returns radial B field for a given (R,Z)
    - get_b_vertical: Returns vertical B field for a given (R,Z)
    - get_b_toroidal: Returns toroidal B field for a given (R,Z)
    - get_b_poloidal: Returns poloidal B field for a given (R,Z)
    - psi_RZ: Returns poloidal flux for a given (R, Z)
    - get_flux_surface: Returns a flux surface for a given normalised (psiN)
    - f_psi: Returns f = R * Bphi (m*T) for given normalised (psiN)
    - ff_prime: Returns ff' for given a normalised (psiN)
    - q: Returns safety factor for given a normalised (psiN)
    - pressure: Returns total pressure (Pa) for a given normalised (psiN)
    - p_prime: Returns pressure gradient w.r.t psi (Pa /(Wb /m2)) for a given normalised (psiN)
    - rho: Returns Normalised minor radius r/a for a given normalised (psiN)
    - R_major: Returns f = R * Bphi (m*T)  for given normalised (psiN)

    Parameters
    ----------
    eq_file: str or Path
        Filename of an equilibrium file to read from.
    eq_type: str, default None
        Name of the equilibrium input type, such as "TRANSP" or "GEQDSK".
        If left as None, this is inferred from the input file.
    **kwargs:
        Extra arguments to be passed to the reader function. Not used by
        all readers, so only include this if necessary.
    """

    def __init__(
        self,
        eq_file: PathLike,
        eq_type: Optional[str] = None,
        **kwargs,
    ):

        self.eq_file = eq_file

        if eq_type is not None:
            reader = equilibrium_readers[eq_type]
            self.eq_type = eq_type
        else:
            # Infer equilibrium type from file
            reader = equilibrium_readers[eq_file]
            self.eq_type = reader.file_type

        # Store results in a dict. This data is accessible via __getattr__,
        # so eq.R gives the same result as eq._data["R"]
        self._data = reader(eq_file, **kwargs)

    @property
    def supported_equilibrium_types(self):
        return [*equilibrium_readers]

    @property
    def eq_type(self):
        return self._eq_type

    @eq_type.setter
    def eq_type(self, value):
        if value not in self.supported_equilibrium_types:
            raise ValueError(f"Equilibrium type {value} is not currently supported")
        self._eq_type = value

    @property
    def nr(self):
        return len(self.R)

    @property
    def nz(self):
        return len(self.Z)

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError:
            raise AttributeError(f"'Equilibrium' object has no attribute '{attr}'")

    def get_b_radial(self, R, Z):

        b_radial = -1 / R * self.psi_RZ(R, Z, dy=1, grid=False)

        return b_radial

    def get_b_vertical(self, R, Z):

        b_vertical = 1 / R * self.psi_RZ(R, Z, dx=1, grid=False)

        return b_vertical

    def get_b_poloidal(self, R, Z):

        b_radial = self.get_b_radial(R, Z)
        b_vertical = self.get_b_vertical(R, Z)

        b_poloidal = np.sqrt(b_radial**2 + b_vertical**2)

        return b_poloidal

    def get_b_toroidal(self, R, Z):

        psi = self.psi_RZ(R, Z, grid=False)

        psi_n = (psi - self.psi_axis) / (self.psi_bdry - self.psi_axis)
        f = self.f_psi(psi_n)

        b_tor = f / R

        return b_tor

    def generate_local(self, geometry_type=None):
        raise NotImplementedError

    def get_flux_surface(self, psi_n=1.0):
        return get_flux_surface(
            self.R,
            self.Z,
            self.psi_RZ,
            self.psi_axis,
            self.psi_bdry,
            self.R_axis,
            self.Z_axis,
            psi_n,
        )

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Equilibrium object

        Returns
        -------
        Copy of Equilibrium object
        """
        # Create new object without calling __init__
        new_equilibrium = Equilibrium.__new__(Equilibrium)
        # Deep copy each member individually
        for key, value in self.__dict__.items():
            setattr(new_equilibrium, key, deepcopy(value, memodict))
        return new_equilibrium
