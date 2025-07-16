# eliteinp.py

from pathlib import Path

import numpy as np
from textwrap import dedent

from ..constants import deuterium_mass, electron_mass, hydrogen_mass
from ..equilibrium import Equilibrium
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
    def read_from_file(self, filename: PathLike, eq: Equilibrium = None) -> Kinetics:

        # Use Equilibrium to obtain rho_func.
        if eq is None:
            raise ValueError(
                dedent(
                    f"""\
                    {self.__class__.__name__} must be provided with an Equilibrium object via
                    the keyword argument 'eq'. Please load an Equilibrium.
                    """
                )
            )

        data = read_eqin(filename)

        psi = data["Psi"]
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0]) * units.dimensionless
        unit_charge_array = np.ones_like(psi_n)

        rho = eq.r_minor(psi_n)
        rho = rho / rho[-1] * units.lref_minor_radius
        rho_func = UnitSpline(psi_n, rho)

        Te_data = data["Te"] * units.eV
        Ti_data = data["Ti"] * units.eV
        ne_data = data["ne"] * units.meter**-3
        ni_data = data["nMainIon"] * units.meter**-3

        Te_func = UnitSpline(psi_n, Te_data)
        Ti_func = UnitSpline(psi_n, Ti_data)
        ne_func = UnitSpline(psi_n, ne_data)
        ni_func = UnitSpline(psi_n, ni_data)

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
        deuterium = Species(
            species_type="deuterium",
            charge=deuteron_charge_func,
            mass=deuterium_mass,
            dens=ni_func,
            temp=Ti_func,
            omega0=omega_func,
            rho=rho_func,
        )

        result = {
            "electron": electron,
            "deuterium": deuterium,
        }

        # Optional: add impurity
        if "Zimp" in data and "Aimp" in data and "nZ" in data:
            impurity_charge_func = UnitSpline(
                psi_n, data["Zimp"] * unit_charge_array * units.elementary_charge
            )
            impurity_mass = data["Aimp"] * hydrogen_mass
            impurity_dens_func = UnitSpline(psi_n, data["nZ"] * units.meter**-3)
            impurity_temp_func = Ti_func  # fallback to Ti

            impurity = Species(
                species_type="impurity",
                charge=impurity_charge_func,
                mass=impurity_mass,
                dens=impurity_dens_func,
                temp=impurity_temp_func,
                omega0=omega_func,
                rho=rho_func,
            )

            result["impurity"] = impurity

        return Kinetics(kinetics_type="ELITEINP", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        with open(filename, "r") as f:
            first_line = f.readline()
            if not first_line.strip().isdigit():
                raise ValueError("ELITEINP file must start with integer grid size")
