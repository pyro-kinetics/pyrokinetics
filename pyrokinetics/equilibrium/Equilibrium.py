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

    Args
    ----
    eq_file (str/Path): Filename of an equilibrium file to read from.
    eq_type (str): Name of the equilibrium input type, such as "TRANSP" or "GEQDSK".
        If left as None, this is inferred from the input file.
    **kwargs: Extra arguments to be passed to the reader function. Not used by
        all readers, so only include this if necessary.

    Attrs
    -----
    supported_equilibrium_types: A list of all supported equilibrium input types.
    eq_files: Stored reference of the last file read
    eq_type: Stored reference of the last equilibrium type. May be inferred.
    nr: TODO
    nz: TODO
    psi_axis: TODO
    psi_bdry: TODO
    R_axis: TODO
    Z_axis: TODO
    f_psi: TODO
    ff_prime: TODO
    q: TODO
    pressure: TODO
    p_prime: TODO
    R: TODO
    Z: TODO
    psi_RZ: TODO
    lcfs_R: TODO
    lcfs_Z: TODO
    a_minor: TODO
    rho: TODO
    R_major: TODO
    bcentr: GEQDSK only, TODO
    psi: TRANSP only, TODO
    current: TRANSP only, TODO


    Functions
    --------
    get_b_radial: TODO
    get_b_vertical: TODO
    get_b_toroidal: TODO
    get_b_poloidal: TODO
    get_flux_surface: TODO
    """

    # Define class level info
    supported_equilibrium_types = [*equilibrium_readers]

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
        self._data = reader(eq_file)

    def __getattr__(self, attr):
        try:
            return self._data[attr]
        except KeyError as e:
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
