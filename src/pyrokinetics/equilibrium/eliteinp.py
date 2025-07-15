import numpy as np
from typing import Optional, Union, IO

from ..file_utils import FileReader
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .equilibrium import Equilibrium


def _parse_eqin(filename_or_file: Union[str, IO]) -> dict:
    """
    Parse a .eqin ELITEINP file and return a dictionary of variables.
    """
    def token_generator(fp):
        for line in fp:
            for tok in line.split():
                yield tok

    if isinstance(filename_or_file, str):
        with open(filename_or_file, "r") as f:
            return _parse_eqin(f)

    f = filename_or_file
    if not f.readline():
        raise IOError("Cannot read from file")

    tokens = token_generator(f)

    try:
        npsi, npol = int(next(tokens)), int(next(tokens))
    except Exception:
        raise IOError("Header must contain npsi and npol")

    data = {"npsi": npsi, "npol": npol}

    while True:
        try:
            var = next(tokens).rstrip(":")
        except StopIteration:
            break

        try:
            if var.lower() in {"r", "z"} or var.startswith("B"):
                arr = [float(next(tokens)) for _ in range(npsi * npol)]
                data[var] = np.reshape(arr, (npsi, npol), order="F")
            elif var in {"Zeff", "Zimp", "Aimp", "Amain"}:
                data[var] = float(next(tokens))
            else:
                data[var] = np.array([float(next(tokens)) for _ in range(npsi)])
        except Exception as e:
            raise IOError(f"Error reading variable {var}: {e}")

    f.close()

    # Derive Bt = fpol * R
    if "fpol" in data and "R" in data:
        data["Bt"] = data["fpol"][:, None] * data["R"]

    # Fill in Ti/Te if one is missing
    if "Te" in data and "Ti" not in data:
        data["Ti"] = data["Te"]
    if "Ti" in data and "Te" not in data:
        data["Te"] = data["Ti"]

    # Compute pressure if needed
    if "p" not in data:
        if "ne" in data and "Te" in data:
            data["p"] = data["ne"] * (data["Te"] + data["Ti"]) * 1.602e-19 * (4e-7 * np.pi)
        else:
            raise IOError("No pressure and insufficient data to compute it")

    return data


class EquilibriumReaderELITEINP(FileReader, file_type="ELITEINP", reads=Equilibrium):
    """
    Reader for ELITEINP (.eqin) ASCII equilibrium files. These files contain profiles of
    equilibrium quantities and a 2D R-Z grid on a (ψ, θ) mesh.
    """

    def read_from_file(
        self,
        filename: PathLike,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None,
    ) -> Equilibrium:
        """
        Read in ELITEINP .eqin file and return an Equilibrium object.

        Parameters
        ----------
        filename : PathLike
            Path to the .eqin file.
        clockwise_phi : bool, optional
            Direction of φ (clockwise vs. anti-clockwise).
        cocos : int, optional
            Optional override for COCOS convention.

        Returns
        -------
        Equilibrium
        """
        data = _parse_eqin(filename)

        len_units = units.meter
        psi_units = units.weber / units.radian
        pressure_units = units.pascal
        ff_units = units.tesla * units.meter

        psi = data["psi"] * psi_units
        F = data["fpol"] * ff_units
        p = data["p"] * pressure_units
        q = data["q"] * units.dimensionless

        FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)
        p_prime = UnitSpline(psi, p)(psi, derivative=1)

        # Geometry from 2D arrays: take mean over poloidal angle
        R2D = data["R"] * len_units
        Z2D = data["Z"] * len_units

        R_major = R2D.mean(axis=1)
        Z_mid = Z2D.mean(axis=1)

        # Estimate r_minor as √(2 * ψ_N)
        r_minor = np.sqrt(2 * (psi / psi[-1]).magnitude) * len_units

        # Simple grid for now (1D psi extended into Cartesian mesh)
        nR = nZ = len(psi)
        R = np.linspace(R_major.min().magnitude, R_major.max().magnitude, nR)
        Z = np.linspace(Z_mid.min().magnitude, Z_mid.max().magnitude, nZ)
        psi_RZ = np.outer(psi.magnitude, np.ones(nR)).T * psi_units

        B_0 = F[0] / R_major[0]

        return Equilibrium(
            R=R * len_units,
            Z=Z * len_units,
            psi_RZ=psi_RZ,
            psi=psi,
            F=F,
            FF_prime=FF_prime,
            p=p,
            p_prime=p_prime,
            q=q,
            R_major=R_major,
            r_minor=r_minor,
            Z_mid=Z_mid,
            psi_lcfs=psi[-1],
            a_minor=r_minor[-1],
            B_0=B_0,
            I_p=np.nan * units.ampere,  # Not given
            clockwise_phi=clockwise_phi,
            cocos=cocos,
            eq_type="ELITEINP",
        )

    def verify_file_type(self, filename: PathLike) -> None:
        with open(filename, "r") as f:
            head = f.read(500).upper()
        if "FPOL" not in head or "PSI" not in head:
            raise ValueError(f"{filename} does not appear to be an ELITEINP .eqin file.")

