from typing import Dict, Any
from ..typing import PathLike
from .EquilibriumReader import EquilibriumReader
from .get_flux_surface import get_flux_surface

import numpy as np
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
)
from freegs import _geqdsk


class EquilibriumReaderGEQDSK(EquilibriumReader):
    def read(
        self,
        filename: PathLike,
        psi_n_lcfs: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Read in GEQDSK file and populates Equilibrium object
        """
        with open(filename) as f:
            gdata = _geqdsk.read(f)

        nr = gdata["nx"]
        nz = gdata["ny"]
        psi_n = np.linspace(0.0, psi_n_lcfs, nr)
        psi_axis = gdata["simagx"]
        psi_bdry = gdata["sibdry"]
        R_axis = (gdata["rmagx"],)
        Z_axis = (gdata["zmagx"],)

        # Set up 1D profiles as interpolated functions
        f_psi = InterpolatedUnivariateSpline(psi_n, gdata["fpol"])
        ff_prime = InterpolatedUnivariateSpline(psi_n, gdata["ffprime"])
        q = InterpolatedUnivariateSpline(psi_n, gdata["qpsi"])
        pressure = InterpolatedUnivariateSpline(psi_n, gdata["pres"])
        p_prime = pressure.derivative()

        # Set up 2D psi_RZ grid
        R = np.linspace(gdata["rleft"], gdata["rleft"] + gdata["rdim"], nr)
        Z = np.linspace(
            gdata["zmid"] - gdata["zdim"] / 2,
            gdata["zmid"] + gdata["zdim"] / 2,
            nz,
        )

        psi_RZ = RectBivariateSpline(R, Z, gdata["psi"])

        rho = np.zeros(len(psi_n))
        R_major = np.zeros(len(psi_n))

        for i, i_psiN in enumerate(psi_n[1:]):

            surface_R, surface_Z = get_flux_surface(
                R, Z, psi_RZ, psi_axis, psi_bdry, R_axis, Z_axis, psi_n=i_psiN
            )

            rho[i + 1] = (max(surface_R) - min(surface_R)) / 2
            R_major[i + 1] = (max(surface_R) + min(surface_R)) / 2

        lcfs_R = surface_R
        lcfs_Z = surface_Z

        a_minor = rho[-1]

        rho = rho / rho[-1]

        R_major[0] = R_major[1] + psi_n[1] * (R_major[2] - R_major[1]) / (
            psi_n[2] - psi_n[1]
        )

        # Return dict of equilibrium data
        return {
            "bcentr": gdata["bcentr"],
            "psi_axis": psi_axis,
            "psi_bdry": psi_bdry,
            "R_axis": R_axis,
            "Z_axis": Z_axis,
            "f_psi": f_psi,
            "ff_prime": ff_prime,
            "q": q,
            "pressure": pressure,
            "p_prime": p_prime,
            "R": R,
            "Z": Z,
            "psi_RZ": psi_RZ,
            "lcfs_R": lcfs_R,
            "lcfs_Z": lcfs_Z,
            "a_minor": a_minor,
            "rho": InterpolatedUnivariateSpline(psi_n, rho),
            "R_major": InterpolatedUnivariateSpline(psi_n, R_major),
        }

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GEQDSK file without processing"""
        # Try opening the GEQDSK file using freegs._geqdsk
        with open(filename) as f:
            gdata = _geqdsk.read(f)
        # Check that the correct variables exist
        var_names = ["nx", "ny", "simagx", "sibdry", "rmagx", "zmagx"]
        if not np.all(np.isin(var_names, list(gdata.keys()))):
            raise ValueError(
                f"EquilibriumReaderGEQDSK was provided an invalid file: {filename}"
            )
