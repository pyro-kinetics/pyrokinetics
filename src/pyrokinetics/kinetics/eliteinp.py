# eliteinp_kinetics.py

import numpy as np
from pathlib import Path

from ..constants import deuterium_mass, electron_mass
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics


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
                data_dict["ne"]
                * (data_dict["Te"] + data_dict["Ti"])
                * 1.602e-19
                * (4.0e-7 * np.pi)
            )
        else:
            raise IOError("Cannot calculate pressure without ne and Te")

    return data_dict


class KineticsReaderELITEINP(FileReader, file_type="ELITEINP", reads=Kinetics):
    def read_from_file(self, filename: PathLike) -> Kinetics:
        data = read_eqin(filename)

        psi = data["Psi"]
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0]) * units.dimensionless
        unit_charge_array = np.ones_like(psi_n)

        rho = np.sqrt(psi_n) * units.lref_minor_radius
        rho_func = UnitSpline(psi_n, rho)

        Te_data = data["Te"] * units.eV
        ne_data = data["ne"] * units.meter**-3  

        Te_func = UnitSpline(psi_n, Te_data)
        ne_func = UnitSpline(psi_n, ne_data)

        Ti_func = (
            UnitSpline(psi_n, data["Ti"] * units.eV)
            if "Ti" in data
            else Te_func
        )

        omega_data = np.zeros_like(psi_n) * units.second**-1
        omega_func = UnitSpline(psi_n, omega_data)

        electron_charge_func = UnitSpline(
            psi_n, -1 * unit_charge_array * units.elementary_charge
        )

        electron = Species(
            species_type="electron",
            charge=electron_charge_func,
            mass=electron_mass,
            dens=ne_func,
            temp=Te_func,
            omega0=omega_func,
            rho=rho_func,
        )

        deuteron_charge_func = UnitSpline(
            psi_n, 1 * unit_charge_array * units.elementary_charge
        )
        deuteron_dens_func = ne_func  # Assume quasi-neutrality for now

        deuterium = Species(
            species_type="deuterium",
            charge=deuteron_charge_func,
            mass=deuterium_mass,
            dens=deuteron_dens_func,
            temp=Ti_func,
            omega0=omega_func,
            rho=rho_func,
        )

        return Kinetics(kinetics_type="ELITEINP", electron=electron, deuterium=deuterium)

    def verify_file_type(self, filename: PathLike) -> None:
        with open(filename, "r") as f:
            first_line = f.readline()
            if not first_line.strip().isdigit():
                raise ValueError("ELITEINP file must start with integer grid size")

