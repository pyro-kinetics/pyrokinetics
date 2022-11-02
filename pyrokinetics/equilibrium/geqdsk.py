import os
import sys
from contextlib import contextmanager

from .equilibrium import Equilibrium, equilibrium_reader
from .flux_surface import _flux_surface_contour
from ..readers import Reader
from ..typing import PathLike

import numpy as np
from freegs import _geqdsk


@equilibrium_reader("GEQDSK")
class GEQDSKReader(Reader):
    """
    Class that can read G-EQDSK equilibrium files. Rather than creating instances of
    this class directly, users are recommended to use the function `read_equilibrium`.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an `Equilibrium`.
    """

    @staticmethod
    @contextmanager
    def _suppress_print():
        """Utility to block freegs IO"""
        # Save stdout locally, temporarily set to /dev/null
        stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w")
            yield
        finally:
            sys.stdout = stdout

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

        # Get geqdsk data in a dict
        with self._suppress_print(), open(filename) as f:
            data = _geqdsk.read(f)

        # Get RZ grids
        # G-EQDSK uses linearly spaced grids, which we must build ourselves.
        r_0 = data["rleft"]
        r_n = data["rleft"] + data["rdim"]
        len_r = data["nx"]
        r = np.linspace(r_0, r_n, len_r)

        z_0 = data["zmid"] - data["zdim"] / 2
        z_n = data["zmid"] + data["zdim"] / 2
        len_z = data["ny"]
        z = np.linspace(z_0, z_n, len_z)

        psi_rz = data["psi"]

        # Get info about magnetic axis and LCFS
        r_axis = data["rmagx"]
        z_axis = data["zmagx"]
        psi_axis = data["simagx"]
        psi_lcfs = data["sibdry"]

        # Get quantities on the psi grid
        # The number of psi values is the same as the number of r values. The psi grid
        # uniformly increases from psi_axis to psi_lcfs
        psi_grid = np.linspace(psi_axis, psi_lcfs, len(r))
        f = data["fpol"]
        ff_prime = data["ffprime"]
        p = data["pres"]
        p_prime = data["pprime"]
        q = data["qpsi"]

        # r_major, r_minor, and z_mid are not provided in the file. They must be
        # determined by fitting contours to the psi_rz grid.
        # TODO This is a major performance bottleneck!
        # - Determine a smaller number of contours and interpolate?
        # - Multiprocessing?
        r_major = np.empty(len(psi_grid))
        r_minor = np.empty(len(psi_grid))
        z_mid = np.empty(len(psi_grid))
        r_major[0] = r_axis
        r_minor[0] = 0.0
        z_mid[0] = data["zmid"]
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
        )

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GEQDSK file without processing"""
        # Try opening the GEQDSK file using freegs._geqdsk
        with self._suppress_print(), open(filename) as f:
            data = _geqdsk.read(f)
        # Check that the correct variables exist
        var_names = ["nx", "ny", "simagx", "sibdry", "rmagx", "zmagx"]
        if not np.all(np.isin(var_names, list(data.keys()))):
            raise ValueError(f"GEQDSKReader was provided an invalid file: {filename}")
