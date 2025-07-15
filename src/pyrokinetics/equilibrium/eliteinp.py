from pathlib import Path
from typing import Optional

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import griddata

from ..file_utils import FileReader
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .equilibrium import Equilibrium


def read_eqin(filename_or_file):
    def token_generator(fp):
        for line in fp:
            for tok in line.split():
                yield tok

    if isinstance(filename_or_file, (str, Path)):
        with open(str(filename_or_file), "r") as fh:
            return read_eqin(fh)

    f = filename_or_file
    if not f.readline():
        raise IOError("Cannot read from input file")

    tokens = token_generator(f)

    try:
        npsi, npol = int(next(tokens)), int(next(tokens))
    except (StopIteration, ValueError):
        raise IOError("Second line should contain Npsi and Npol")

    data_dict = {"npsi": npsi, "npol": npol}

    while True:
        try:
            varname = next(tokens).rstrip(":")
        except StopIteration:
            break

        try:
            if varname.lower() in {"r", "z"} or varname.startswith("B"):
                data = np.array([float(next(tokens)) for _ in range(npsi * npol)])
                data = data.reshape((npsi, npol), order="F")
            elif varname in {"Zeff", "Zimp", "Aimp", "Amain"}:
                data = float(next(tokens))
            else:
                data = np.array([float(next(tokens)) for _ in range(npsi)])
        except (StopIteration, ValueError):
            raise IOError(f"Error while reading {varname}")

        data_dict[varname] = data

    f.close()

    if "Psi" in data_dict:
        data_dict["Psi"] = data_dict["Psi"] * units.weber / units.radian

    if "dp/dpsi" in data_dict:
        data_dict["p_prime"] = (
            data_dict["dp/dpsi"] * units.pascal / (units.weber / units.radian)
        )

    if "ffp" in data_dict:
        data_dict["ff_prime"] = data_dict["ffp"] * (
            units.tesla**2 * units.meter**2 * units.radian / units.weber
        )

    if "Bt" not in data_dict and "fpol" in data_dict and "R" in data_dict:
        fpol = data_dict["fpol"]
        R = data_dict["R"]  # shape (npsi, npol)
        data_dict["Bt"] = fpol[:, None] * R  # shape (npsi, npol)

    if "p" not in data_dict:
        if "ne" in data_dict and "Te" in data_dict:
            data_dict["p"] = (
                data_dict["ne"] * (data_dict["Te"] + data_dict["Ti"]) * 1.602e-19
            ) * units.pascal
        else:
            raise IOError("Cannot reconstruct pressure without ne and Te")

    return data_dict


class EquilibriumReaderELITEINP(FileReader, file_type="ELITEINP", reads=Equilibrium):
    def read_from_file(
        self,
        filename: PathLike,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None,
    ) -> Equilibrium:
        data = read_eqin(filename)

        print("Parsed keys from ELITEINP:", data.keys())

        # Units
        len_units = units.meter
        psi = data["Psi"]
        F = data["fpol"] * units.tesla * units.meter
        p = data["p"]
        q = data["q"] * units.dimensionless
        FF_prime = data["ff_prime"]
        p_prime = data["p_prime"]

        R2D = data["R"] * len_units
        Z2D = data["z"] * len_units
        R_major = R2D.mean(axis=1)
        Z_mid = Z2D.mean(axis=1)

        # Normalised psi
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
        r_minor = np.sqrt(psi_n) * R2D.max()  # full profile
        a_minor = r_minor[-1]

        # Flatten known grid
        R_flat = R2D.to_base_units().magnitude.ravel()
        Z_flat = Z2D.to_base_units().magnitude.ravel()
        psi_flat = np.repeat(psi.to_base_units().magnitude, R2D.shape[1])

        # Regular grid
        nR, nZ = 256, 256
        R = np.linspace(R_flat.min(), R_flat.max(), nR) * len_units
        Z = np.linspace(Z_flat.min(), Z_flat.max(), nZ) * len_units
        grid_R, grid_Z = np.meshgrid(R.magnitude, Z.magnitude, indexing="ij")

        # Interpolate ψ(R, Z)
        psi_RZ_vals = griddata(
            points=np.column_stack([R_flat, Z_flat]),
            values=psi_flat,
            xi=(grid_R, grid_Z),
            method="cubic",
        )
        psi_RZ_vals = np.nan_to_num(psi_RZ_vals, nan=psi_flat.min())
        psi_RZ = psi_RZ_vals * psi.units

        # Magnetic field strength at axis
        B_0 = F[0] / R_major[0]

        # Infer total current
        Ip = data.get("Ip")
        if Ip is None:
            try:
                dpsi = np.gradient(psi.magnitude)
                I_profile = (
                    2
                    * np.pi
                    * np.mean(data["R"])
                    * np.mean(data["Bt"])
                    / (mu_0 * data["q"])
                    * dpsi
                )
                Ip = np.trapz(I_profile, psi.magnitude)
            except Exception as e:
                raise IOError(f"Could not infer total current: {e}")

        return Equilibrium(
            R=R,
            Z=Z,
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
            a_minor=a_minor,
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
            raise ValueError(
                f"{filename} does not appear to be an ELITEINP .eqin file."
            )
