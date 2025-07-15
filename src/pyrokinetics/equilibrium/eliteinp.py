import numpy as np
from typing import Optional
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.constants import mu_0
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
    except StopIteration:
        raise IOError("Unexpected EOF while reading grid size")
    except ValueError:
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
        except StopIteration:
            raise IOError(f"Unexpected EOF while reading {varname}")
        except ValueError:
            raise IOError(f"Expected float while reading {varname}")

        data_dict[varname] = data

    f.close()

    if "Bt" not in data_dict:
        try:
            fpol, R = data_dict["fpol"], data_dict["R"]
            data_dict["Bt"] = fpol[:, None] * R
        except KeyError:
            raise IOError("Need fpol and R to calculate Bt")

    if "Te" in data_dict and "Ti" not in data_dict:
        data_dict["Ti"] = data_dict["Te"]
    elif "Ti" in data_dict and "Te" not in data_dict:
        data_dict["Te"] = data_dict["Ti"]

    if "p" not in data_dict:
        if "ne" in data_dict and "Te" in data_dict:
            data_dict["p"] = (
                data_dict["ne"] * (data_dict["Te"] + data_dict["Ti"])
                * 1.602e-19 * (4.0e-7 * np.pi)
            )
        else:
            raise IOError("Cannot calculate pressure without ne and Te")

    return data_dict


class EquilibriumReaderELITEINP(FileReader, file_type="ELITEINP", reads=Equilibrium):
    def read_from_file(
        self,
        filename: PathLike,
        clockwise_phi: bool = False,
        cocos: Optional[int] = None
    ) -> Equilibrium:
        data = read_eqin(filename)

        len_units = units.meter
        psi_units = units.weber / units.radian
        pressure_units = units.pascal
        ff_units = units.tesla * units.meter

        psi = data["Psi"] * psi_units
        F = data["fpol"] * ff_units
        p = data["p"] * pressure_units
        q = data["q"] * units.dimensionless

        FF_prime = F * UnitSpline(psi, F)(psi, derivative=1)
        p_prime = UnitSpline(psi, p)(psi, derivative=1)

        R2D = data["R"] * len_units
        Z2D = data["z"] * len_units

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
                raw_psi = data["Psi"]
                raw_q = data["q"]
                raw_Bt = np.mean(data["Bt"])
                raw_R = np.mean(data["R"])
                dpsi = np.gradient(raw_psi)
                I_profile = 2 * np.pi * raw_R * raw_Bt / (mu_0 * raw_q) * dpsi
                Ip = np.trapz(I_profile, raw_psi)
            except Exception as e:
                raise IOError(f"Could not infer total current: {e}")

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

