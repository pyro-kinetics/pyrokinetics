# eliteinp.py

from textwrap import dedent

import numpy as np

from ..constants import deuterium_mass, electron_charge, electron_mass, hydrogen_mass
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

    with open(filename_or_file) as f:
        f.readline()
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

    necessary_keys = ["Psi", "ne", "Te", "nMainIon", "Ti"]
    if not all(key in data_dict.keys() for key in necessary_keys):
        raise ValueError(
            "Missing kinetics data in ELITEINP file."
            " Currently only HELENA ELITEINP files are supported, please"
            " raise issue if another source (like) SCENE is needed"
        )

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
        if "ne" not in data_dict or "Te" not in data_dict:
            raise ValueError(
                "No electron density and temperature data found in ELITEINP"
            )

        if "nMainIon" not in data_dict or "Ti" not in data_dict:
            raise ValueError(
                "No main ion density and temperature data found in ELITEINP"
            )

        data_dict["p"] = (
            data_dict["ne"] * (data_dict["Te"] * electron_charge.m)
        ) * units.pascal

        data_dict["p"] += (
            data_dict["nMainIon"] * (data_dict["Ti"] * electron_charge.m)
        ) * units.pascal

        if "nZ" in data_dict and "Ti" in data_dict:
            data_dict["p"] += (
                data_dict["nZ"] * (data_dict["Ti"] * electron_charge.m)
            ) * units.pascal

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
            head = f.read(50).upper()
        if "HELENA GENERATED INPUT" not in head and "Psi:" not in head:
            raise ValueError(
                f"{filename} does not appear to be an ELITEINP .eqin file."
                " Currently only HELENA ELITEINP files are supported, please"
                " raise issue if another source (like) SCENE is needed"
            )
