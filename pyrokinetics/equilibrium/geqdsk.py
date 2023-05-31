from contextlib import redirect_stdout
from typing import Optional

import numpy as np
from freeqdsk import geqdsk

from ..readers import Reader
from ..typing import PathLike
from ..units import ureg as units
from .equilibrium import Equilibrium, equilibrium_reader
from .flux_surface import _flux_surface_contour


@equilibrium_reader("GEQDSK")
class EquilibriumReaderGEQDSK(Reader):
    r"""
    Class that can read G-EQDSK equilibrium files and return ``Equilibrium`` objects.
    Users are not recommended to instantiate this class directly, and should instead use
    the functions ``read_equilibrium`` or ``Equilibrium.from_file``. Keyword arguments
    passed to those functions will be forwarded to this class.

    Here we assume the convention COCOS 1 [Sauter & Medvedev, 2013]. However, EFIT uses
    COCOS 3, and other codes may follow other standards.

    Some G-EQDSK files may not read correctly, as it is not possible to fit a closed
    contour on the Last Closed Flux Surface (LCFS). In these cases, the user may provide
    the argument ``psi_n_lcfs=0.99`` (or something similarly close to 1) which adjusts
    the :math:`\psi` grid so that only values in the range
    :math:`[\psi_\text{axis},\psi_\text{axis}+0.99(\psi_\text{LCFS}-\psi_\text{axis})]`
    are included.

    It is not possible to determine the coordinate system used by a G-EQDSK file from
    its own data alone. By default, we assume that the toroidal angle :math:`\phi`
    increases in an anti-clockwise direction when the tokamak is viewed from above. If
    the G-EQDSK file originates from a code that uses the opposite convention, the
    user should set ``clockwise_phi`` to ``True``. Alternatively, if the COCOS
    convention of the G-EQDSK file is known, this should be supplied to the optional
    ``cocos`` argument.

    See Also
    --------
    Equilibrium: Class representing a global tokamak equilibrium.
    read_equilibrium: Read an equilibrium file, return an ``Equilibrium``.
    """

    def read(
        self,
        filename: PathLike,
        psi_n_lcfs: float = 1.0,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None,
    ) -> Equilibrium:
        r"""
        Read in G-EQDSK file and populate Equilibrium object. Should not be invoked
        directly; users should instead use ``read_equilibrium``.

        Parameters
        ----------
        filename: PathLike
            Location of the G-EQDSK file on disk.
        psi_n_lcfs: float, default 1.0
            Adjust which flux surface we consider to be the last closed flux surface
            (LCFS). Should take a value between 0.0 and 1.0 inclusive.
        clockwise_phi: bool, default False
            Determines whether the :math:`\phi` grid increases clockwise or
            anti-clockwise when the tokamak is viewed from above.
        cocos: Optional[int]
            If set, asserts that the GEQDSK file follows that COCOS convention, and
            neither ``clockwise_phi`` nor the file contents will be used to identify
            the actual convention in use. The resulting Equilibrium is always converted
            to COCOS 11.

        Returns
        -------
        Equilibrium
        """
        # Define some units to use later
        # GEQDSK should use COCOS 1 standards, though not all codes do so.
        # Most (all?) use COCOS 1 -> 8, so psi is in Webers per radian.
        # Equilibrium should be able to handle the conversion to Webers itself.
        psi_units = units.weber / units.radian
        F_units = units.meter * units.tesla

        # Get geqdsk data in a dict
        with redirect_stdout(None), open(filename) as f:
            data = geqdsk.read(f)

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
        B_0 = data["bcentr"] * units.tesla
        I_p = data["cpasma"] * units.ampere

        # Get quantities on the psi grid
        # The number of psi values is the same as the number of r values. The psi grid
        # uniformly increases from psi_axis to psi_lcfs
        psi_grid = np.linspace(psi_axis, psi_lcfs, len(R))
        F = data["fpol"] * F_units
        FF_prime = data["ffprime"] * F_units**2 / psi_units
        p = data["pres"] * units.pascal
        p_prime = data["pprime"] * units.pascal / psi_units
        q = data["qpsi"] * units.dimensionless

        #  Adjust grids if psi_n_lcfs is not 1
        if psi_n_lcfs != 1.0:
            if psi_n_lcfs > 1.0 or psi_n_lcfs < 0.0:
                raise ValueError(f"psi_n_lcfs={psi_n_lcfs} must be in the range [0,1]")
            psi_lcfs_new = psi_axis + psi_n_lcfs * (psi_lcfs - psi_axis)
            # Find the index at which psi_lcfs_new would be inserted.
            lcfs_idx = np.searchsorted(psi_grid, psi_lcfs_new)
            # Discard elements off the end of the grid, insert new psi_lcfs
            psi_grid_new = np.concatenate((psi_grid[:lcfs_idx], [psi_lcfs_new]))
            # Linearly interpolate each grid onto the new psi_grid
            F = np.interp(psi_grid_new, psi_grid, F)
            FF_prime = np.interp(psi_grid_new, psi_grid, FF_prime)
            p = np.interp(psi_grid_new, psi_grid, p)
            p_prime = np.interp(psi_grid_new, psi_grid, p_prime)
            q = np.interp(psi_grid_new, psi_grid, q)
            # Replace psi_grid and psi_lcfs with the new versions
            psi_grid = psi_grid_new
            psi_lcfs = psi_lcfs_new

        # r_major, r_minor, and z_mid are not provided in the file. They must be
        # determined by fitting contours to the psi_rz grid.
        R_major = np.empty(len(psi_grid)) * units.meter
        r_minor = np.empty(len(psi_grid)) * units.meter
        Z_mid = np.empty(len(psi_grid)) * units.meter
        R_major[0] = R_axis
        r_minor[0] = 0.0 * units.meter
        Z_mid[0] = data["zmid"] * units.meter
        for idx, psi in enumerate(psi_grid[1:], start=1):
            Rc, Zc = _flux_surface_contour(R, Z, psi_RZ, R_axis, Z_axis, psi)
            R_min, R_max = min(Rc), max(Rc)
            Z_min, Z_max = min(Zc), max(Zc)
            R_major[idx] = 0.5 * (R_max + R_min)
            r_minor[idx] = 0.5 * (R_max - R_min)
            Z_mid[idx] = 0.5 * (Z_max + Z_min)

        a_minor = r_minor[-1]

        # Create and return Equilibrium
        return Equilibrium(
            R=R,
            Z=Z,
            psi_RZ=psi_RZ,
            psi=psi_grid,
            F=F,
            FF_prime=FF_prime,
            p=p,
            p_prime=p_prime,
            q=q,
            R_major=R_major,
            r_minor=r_minor,
            Z_mid=Z_mid,
            psi_lcfs=psi_lcfs,
            a_minor=a_minor,
            B_0=B_0,
            I_p=I_p,
            clockwise_phi=clockwise_phi,
            cocos=cocos,
            eq_type="GEQDSK",
        )

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GEQDSK file without processing"""
        # Try opening the GEQDSK file using freeqdsk.geqdsk
        with redirect_stdout(None), open(filename) as f:
            data = geqdsk.read(f)
        # Check that the correct variables exist
        var_names = ["nx", "ny", "simagx", "sibdry", "rmagx", "zmagx"]
        if not np.all(np.isin(var_names, list(data.keys()))):
            raise ValueError(f"GEQDSKReader was provided an invalid file: {filename}")
