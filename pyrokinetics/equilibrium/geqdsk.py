from contextlib import redirect_stdout

from .equilibrium import Equilibrium, equilibrium_reader
from .utils import UnitSpline
from .flux_surface import _flux_surface_contour
from ..readers import Reader
from ..typing import PathLike
from ..units import ureg as units

import numpy as np
from freegs import _geqdsk


@equilibrium_reader("GEQDSK")
class EquilibriumReaderGEQDSK(Reader):
    r"""
    Class that can read G-EQDSK equilibrium files. Rather than creating instances of
    this class directly, users are recommended to use the function ``read_equilibrium``.

    Note: Here we assume the convention COCOS 1. However, EFIT uses COCOS 3. Some
    G-EQDSK files may not read correctly depending on the source.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an ``Equilibrium``.
    """

    def read(self, filename: PathLike, psi_n_lcfs: float = 1.0) -> Equilibrium:
        """
        Read in G-EQDSK file and populate Equilibrium object. Should not be invoked
        directly; users should instead use ``read_equilibrium``.

        Parameters
        ----------
        filename: PathLike
            Location of the G-EQDSK file on disk.
        psi_n_lcfs: float, default 1.0
            Adjust which flux surface we consider to be the last closed flux surface
            (LCFS). Should take a value between 0.0 and 1.0 inclusive.

        Returns
        -------
        Equilibrium
        """
        # Check inputs
        if psi_n_lcfs < 0.0 or psi_n_lcfs > 1.0:
            raise ValueError("psi_n_lcfs should be between 0.0 and 1.0 (inclusive)")

        # Define some units to use later
        psi_units = units.weber / units.radian
        f_units = units.meter * units.tesla

        # Get geqdsk data in a dict
        with redirect_stdout(None), open(filename) as f:
            data = _geqdsk.read(f)

        # Get RZ grids
        # G-EQDSK uses linearly spaced grids, which we must build ourselves.
        R_0 = data["rleft"] * units.meter
        R_n = (data["rleft"] + data["rdim"]) * units.meter
        len_R = data["nx"]
        R = np.linspace(R_0, R_n, len_R)

        Z_0 = 0.5 * (data["zmid"] - data["zdim"]) * units.meter
        Z_n = 0.5 * (data["zmid"] + data["zdim"]) * units.meter
        len_Z = data["ny"]
        Z = np.linspace(Z_0, Z_n, len_Z)

        psi_RZ = data["psi"] * psi_units

        # Get info about magnetic axis and LCFS
        R_axis = data["rmagx"] * units.meter
        Z_axis = data["zmagx"] * units.meter
        psi_axis = data["simagx"] * psi_units
        psi_lcfs = data["sibdry"] * psi_units

        # Get quantities on the psi grid
        # The number of psi values is the same as the number of r values. The psi grid
        # uniformly increases from psi_axis to psi_lcfs
        psi_grid = np.linspace(psi_axis, psi_lcfs, len(R))
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
        R_major = np.empty(len(psi_grid)) * units.meter
        r_minor = np.empty(len(psi_grid)) * units.meter
        Z_mid = np.empty(len(psi_grid)) * units.meter
        R_major[0] = R_axis
        r_minor[0] = 0.0 * units.meter
        Z_mid[0] = data["zmid"] * units.meter
        for idx, psi in enumerate(psi_grid[1:], start=1):
            Rc, Zc = _flux_surface_contour(R, Z, psi_RZ, R_axis, Z_axis, psi)
            R_max = max(Rc)
            R_min = min(Rc)
            Z_max = max(Zc)
            Z_min = min(Zc)
            R_major[idx] = 0.5 * (R_max + R_min)
            r_minor[idx] = 0.5 * (R_max - R_min)
            Z_mid[idx] = 0.5 * (Z_max + Z_min)

        # Adjust normalising factors if psi_n_lcfs is not 1.0
        if psi_n_lcfs == 1.0:
            a_minor = r_minor[-1]
        else:
            psi_lcfs = psi_n_lcfs * psi_lcfs + (1.0 - psi_n_lcfs) * psi_axis
            a_minor = UnitSpline(psi_grid, r_minor)(psi_lcfs)

        # Create and return Equilibrium
        return Equilibrium(
            R=R,
            Z=Z,
            psi_RZ=psi_RZ,
            psi=psi_grid,
            f=f,
            ff_prime=ff_prime,
            p=p,
            p_prime=p_prime,
            q=q,
            R_major=R_major,
            r_minor=r_minor,
            Z_mid=Z_mid,
            psi_lcfs=psi_lcfs,
            a_minor=a_minor,
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