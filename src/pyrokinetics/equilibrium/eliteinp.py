import numpy as np
from typing import Optional, Union, IO
from pathlib import Path
from scipy.interpolate import interp1d
from ..file_utils import FileReader
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .equilibrium import Equilibrium
from scipy.constants import mu_0


def _parse_eqin(filename_or_file: Union[str, Path, IO]) -> dict:
    """
    Parse a .eqin ELITEINP file and return a dictionary of variables.

    Parameters
    ----------
    filename_or_file : str, Path, or file-like
        The ELITEINP file to read.

    Returns
    -------
    dict
        Dictionary of variables including npsi, npol, psi, fpol, R, Z, etc.
    """

    def token_generator(fp):
        for line in fp:
            for tok in line.split():
                yield tok

    if isinstance(filename_or_file, (str, Path)):
        with open(str(filename_or_file), "r") as f:
            return _parse_eqin(f)

    f = filename_or_file
    # Skip header lines until npsi, npol found
    while True:
        line = f.readline()
        if not line:
            raise IOError("Cannot read from file or no data found.")
        try:
            tokens = line.split()
            npsi, npol = int(tokens[0]), int(tokens[1])
            break
        except (ValueError, IndexError):
            continue

    data = {"npsi": npsi, "npol": npol}
    tokens = token_generator(f)

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

    # Normalize variable names
    if "r" in data:
        data["R"] = data.pop("r")
    if "z" in data:
        data["Z"] = data.pop("z")

    key_renames = {
        "Psi": "psi",
        "R": "R",
        "Z": "Z",
    }
    for old, new in key_renames.items():
        if old in data and new not in data:
            data[new] = data.pop(old)

    # Derive Bt = fpol * R
    if "fpol" in data and "R" in data:
        data["Bt"] = data["fpol"][:, None] * data["R"]

    # Fill Ti or Te if missing
    if "Te" in data and "Ti" not in data:
        data["Ti"] = data["Te"]
    if "Ti" in data and "Te" not in data:
        data["Te"] = data["Ti"]

    # Compute pressure if missing
    if "p" not in data:
        if "ne" in data and "Te" in data:
            data["p"] = data["ne"] * (data["Te"] + data["Ti"]) * 1.602e-19 * (4e-7 * np.pi)
        else:
            raise IOError("No pressure and insufficient data to compute it")

    return data


def interpolate_flux_surface(data_2d, psi_array, psi_n):
    """
    Interpolate 2D data on (psi, poloidal angle) to 1D at given psi_n.
    """
    psi_norm = psi_array / psi_array[-1]
    n_pol = data_2d.shape[1]
    data_1d = np.empty(n_pol)
    for j in range(n_pol):
        interp_func = interp1d(psi_norm, data_2d[:, j], kind="cubic", fill_value="extrapolate")
        data_1d[j] = interp_func(psi_n)
    return data_1d


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
        psi_n: float = 0.5,
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
        psi_n : float, optional
            Normalized flux surface value (0 to 1) to interpolate poloidal profiles.

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

        R2D = data["R"] * len_units
        Z2D = data["Z"] * len_units

        R_major = R2D.mean(axis=1)
        Z_mid = Z2D.mean(axis=1)

        r_minor = np.sqrt(2 * (psi / psi[-1]).magnitude) * len_units

        nR = nZ = len(psi)
        R = np.linspace(R_major.min().magnitude, R_major.max().magnitude, nR)
        Z = np.linspace(Z_mid.min().magnitude, Z_mid.max().magnitude, nZ)
        psi_RZ = np.outer(psi.magnitude, np.ones(nR)).T * psi_units

        B_0 = F[0] / R_major[0]


        try:
            Ip = data["Ip"]
        except KeyError:
            try:
                if "psi" in data:
                    raw_psi = data["psi"]
                elif "Psi" in data:
                    raw_psi = data["Psi"]
                else:
                    raise IOError("Missing psi")

                raw_q = data["q"]
                raw_Bt = np.mean(data["Bt"])
                raw_R = np.mean(data["R"])
                dpsi = np.gradient(raw_psi)

                I_profile = 2 * np.pi * raw_R * raw_Bt / (mu_0 * raw_q) * dpsi
                Ip = np.trapz(I_profile, raw_psi)
                data["Ip"] = Ip
            except Exception as e:
                raise IOError(f"Could not infer total current: {e}")


        # Interpolate poloidal profiles at desired psi_n
        psi_array = psi.magnitude

        R_eq_1d = interpolate_flux_surface(R2D.magnitude, psi_array, psi_n) * len_units
        Z_eq_1d = interpolate_flux_surface(Z2D.magnitude, psi_array, psi_n) * len_units

        # Use b_poloidal_eq if present, else Bp if present
        if "b_poloidal_eq" in data:
            bpol_2d = data["b_poloidal_eq"] * units.tesla
        elif "Bp" in data:
            bpol_2d = data["Bp"] * units.tesla
        else:
            raise IOError("Poloidal magnetic field data not found for interpolation")

        bpol_eq_1d = interpolate_flux_surface(bpol_2d.magnitude, psi_array, psi_n) * units.tesla

        # Attach to equilibrium for Miller fit
        self.R_eq = R_eq_1d
        self.Z_eq = Z_eq_1d
        self.b_poloidal_eq = bpol_eq_1d

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
            I_p=Ip * units.ampere,
            clockwise_phi=clockwise_phi,
            cocos=cocos,
            eq_type="ELITEINP",
        )

    def verify_file_type(self, filename: PathLike) -> None:
        with open(filename, "r") as f:
            head = f.read(500).upper()
        if "FPOL" not in head or "PSI" not in head:
            raise ValueError(f"{filename} does not appear to be an ELITEINP .eqin file.")

