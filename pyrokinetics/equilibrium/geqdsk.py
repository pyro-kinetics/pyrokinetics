from contextlib import redirect_stdout

from .equilibrium import Equilibrium, equilibrium_reader
from .flux_surface import _flux_surface_contour
from ..readers import Reader
from ..typing import PathLike
from ..normalisation import ureg as units

import numpy as np
from freegs import _geqdsk


@equilibrium_reader("GEQDSK")
class GEQDSKReader(Reader):
    r"""
    Class that can read G-EQDSK equilibrium files. Rather than creating instances of
    this class directly, users are recommended to use the function `read_equilibrium`.

    Note: Here we assume the convention COCOS 1. However, EFIT uses COCOS 3. Some
    G-EQDSK files may not read correctly depending on the source.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an `Equilibrium`.
    """

    def read(self, filename: PathLike) -> Equilibrium:
        """
        Read in G-EQDSK file and populate Equilibrium object. Should not be invoked
        directly; users should instead use ``read_equilibrium``.

        Parameters
        ----------
        filename: PathLike
           Location of the G-EQDSK file on disk.

        Returns
        -------
        Equilibrium
        """
        # Define some units to use later
        psi_units = units.weber / units.radian
        f_units = units.meter * units.tesla

        # Get geqdsk data in a dict
        with redirect_stdout(None), open(filename) as f:
            data = _geqdsk.read(f)

        # Get RZ grids
        # G-EQDSK uses linearly spaced grids, which we must build ourselves.
        r_0 = data["rleft"] * units.meter
        r_n = (data["rleft"] + data["rdim"]) * units.meter
        len_r = data["nx"]
        r = np.linspace(r_0, r_n, len_r)

        z_0 = 0.5 * (data["zmid"] - data["zdim"]) * units.meter
        z_n = 0.5 * (data["zmid"] + data["zdim"]) * units.meter
        len_z = data["ny"]
        z = np.linspace(z_0, z_n, len_z)

        psi_rz = data["psi"] * psi_units

        # Get info about magnetic axis and LCFS
        r_axis = data["rmagx"] * units.meter
        z_axis = data["zmagx"] * units.meter
        psi_axis = data["simagx"] * psi_units
        psi_lcfs = data["sibdry"] * psi_units

        # Get quantities on the psi grid
        # The number of psi values is the same as the number of r values. The psi grid
        # uniformly increases from psi_axis to psi_lcfs
        psi_grid = np.linspace(psi_axis, psi_lcfs, len(r))
        f = data["fpol"] * f_units
        ff_prime = data["ffprime"] * f_units**2 / psi_units
        p = data["pres"] * units.pascal
        p_prime = data["pprime"] * units.pascal / psi_units
        q = data["qpsi"] * units.dimensionless

        # r_major, r_minor, and z_mid are not provided in the file. They must be
        # determined by fitting contours to the psi_rz grid.
        # TODO This is a major performance bottleneck!
        # - Determine a smaller number of contours and interpolate?
        # - Multiprocessing?
        r_major = np.empty(len(psi_grid)) * units.meter
        r_minor = np.empty(len(psi_grid)) * units.meter
        z_mid = np.empty(len(psi_grid)) * units.meter
        r_major[0] = r_axis
        r_minor[0] = 0.0 * units.meter
        z_mid[0] = data["zmid"] * units.meter
        for idx, psi in enumerate(psi_grid[1:], start=1):
            rc, zc = _flux_surface_contour(r, z, psi_rz, r_axis, z_axis, psi)
            r_max = max(rc)
            r_min = min(rc)
            z_max = max(zc)
            z_min = min(zc)
            r_major[idx] = 0.5 * (r_max + r_min)
            r_minor[idx] = 0.5 * (r_max - r_min)
            z_mid[idx] = 0.5 * (z_max + z_min)

        # Create and return Equilibrium
        return Equilibrium(
            r=r,
            z=z,
            psi_rz=psi_rz,
            psi=psi_grid,
            f=f,
            ff_prime=ff_prime,
            p=p,
            p_prime=p_prime,
            q=q,
            r_major=r_major,
            r_minor=r_minor,
            z_mid=z_mid,
            psi_lcfs=psi_lcfs,
            a_minor=r_minor[-1],
            eq_type="GEQDSK",
        )

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GEQDSK file without processing"""
        # Try opening the GEQDSK file using freegs._geqdsk
        with redirect_stdout(None), open(filename) as f:
            data = _geqdsk.read(f)
        # Check that the correct variables exist
        var_names = ["nx", "ny", "simagx", "sibdry", "rmagx", "zmagx"]
        if not np.all(np.isin(var_names, list(data.keys()))):
            raise ValueError(f"GEQDSKReader was provided an invalid file: {filename}")
