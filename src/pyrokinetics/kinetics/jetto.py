from ..typing import PathLike
from .kinetics import Kinetics
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass, electron_charge
from ..units import ureg as units, UnitSpline
from ..file_utils import FileReader
import numpy as np
from jetto_tools.binary import read_binary_file


class KineticsReaderJETTO(FileReader, file_type="JETTO", reads=Kinetics):
    def read_from_file(
        self, filename: PathLike, time_index: int = -1, time: float = None
    ) -> Kinetics:
        """
        Reads in JETTO profiles NetCDF file
        """
        # Open data file, get generic data
        try:
            kinetics_data = read_binary_file(filename)
            jssfile = str(filename).replace("jsp", "jss")
            jetto_jss = read_binary_file(jssfile)

        except Exception as e:
            if "not found. Abort" in str(e):
                raise FileNotFoundError(
                    f"KineticsReaderJETTO could not find {filename}"
                ) from e
            elif "Extention of file" in str(e):
                raise ValueError(
                    f"Extention of file {filename} not in allowed list. Abort."
                )
            else:
                raise e

        time_cdf = kinetics_data["TIME"].T[:]

        if time_index != -1 and time is not None:
            raise ValueError("Cannot set both `time` and `time_index`")

        if time is not None:
            time_index = np.argmin(np.abs(time_cdf - time))

        psi = kinetics_data["PSI"][time_index, :]
        psi = psi - psi[0]
        psi_n = psi / psi[-1] * units.dimensionless

        unit_charge_array = np.ones(len(psi_n))

        Rmax = kinetics_data["R"][time_index, :]
        Rmin = kinetics_data["RI"][time_index, :]

        r = (Rmax - Rmin) / 2
        rho = r / r[-1] * units.lref_minor_radius
        rho_func = UnitSpline(psi_n, rho)

        # Electron data
        electron_temp_data = kinetics_data["TE"][time_index, :] * units.eV
        electron_temp_func = UnitSpline(psi_n, electron_temp_data)

        electron_dens_data = kinetics_data["NETF"][time_index, :] * units.meter**-3
        electron_dens_func = UnitSpline(psi_n, electron_dens_data)

        omega_data = kinetics_data["ANGF"][time_index, :] * units.second**-1

        omega_func = UnitSpline(psi_n, omega_data)

        electron_charge_func = UnitSpline(
            psi_n, -1 * unit_charge_array * units.elementary_charge
        )

        electron = Species(
            species_type="electron",
            charge=electron_charge_func,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            omega0=omega_func,
            rho=rho_func,
        )

        result = {"electron": electron}

        # JETTO only has one temp for impurities and main ions
        thermal_temp_data = kinetics_data["TI"][time_index, :] * units.eV
        thermal_temp_func = UnitSpline(psi_n, thermal_temp_data)
        if not np.all(kinetics_data["NALF"][time_index, :] == 0):
            fast_temp_data = (
                np.nan_to_num(
                    2.0
                    / 3.0
                    * kinetics_data["WALD"][time_index, :]
                    / kinetics_data["NALF"][time_index, :]
                    / electron_charge.m
                )
                * units.eV
            )
            fast_temp_func = UnitSpline(psi_n, fast_temp_data)

        possible_species = [
            {
                "species_name": "deuterium",
                "jetto_name": "NID",
                "charge": UnitSpline(
                    psi_n, 1 * unit_charge_array * units.elementary_charge
                ),
                "mass": deuterium_mass,
            },
            {
                "species_name": "tritium",
                "jetto_name": "NIT",
                "charge": UnitSpline(
                    psi_n, 1 * unit_charge_array * units.elementary_charge
                ),
                "mass": 1.5 * deuterium_mass,
            },
            {
                "species_name": "alpha",
                "jetto_name": "NALF",
                "charge": UnitSpline(
                    psi_n, 2 * unit_charge_array * units.elementary_charge
                ),
                "mass": 4 * hydrogen_mass,
            },
        ]

        # Go through each species output in JETTO
        impurity_keys = [key for key in kinetics_data.keys() if "ZIA" in key]

        for i_imp, impurity_z in enumerate(impurity_keys):
            # impurity charge can have a profile variation
            impurity_charge_data = (
                kinetics_data[impurity_z][time_index, :] * units.elementary_charge
            )
            impurity_charge_func = UnitSpline(psi_n, impurity_charge_data)

            # mass unchanged with profile
            impurity_mass = jetto_jss[f"AIM{i_imp+1}"][0][0] * hydrogen_mass

            possible_species.append(
                {
                    "species_name": f"impurity{i_imp+1}",
                    "jetto_name": f"NIM{i_imp+1}",
                    "charge": impurity_charge_func,
                    "mass": impurity_mass,
                }
            )

        for species in possible_species:
            density_data = (
                kinetics_data[species["jetto_name"]][time_index, :] * units.meter**-3
            )
            if not any(density_data):
                continue

            density_func = UnitSpline(psi_n, density_data)

            if species["species_name"] == "alpha":
                ion_temp_func = fast_temp_func
            else:
                ion_temp_func = thermal_temp_func

            result[species["species_name"]] = Species(
                species_type=species["species_name"],
                charge=species["charge"],
                mass=species["mass"],
                dens=density_func,
                temp=ion_temp_func,
                omega0=omega_func,
                rho=rho_func,
            )

        return Kinetics(kinetics_type="JETTO", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a JETTO file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            data = read_binary_file(filename)
        except Exception as e:
            if "not found. Abort" in str(e):
                raise FileNotFoundError(
                    f"KineticsReaderJETTO could not find {filename}"
                ) from e
            elif "Extention of file" in str(e):
                raise ValueError(
                    f"Extention of file {filename} not in allowed list. Abort."
                )
            else:
                raise e
        try:
            if "JSP" not in data["DDA NAME"]:
                raise ValueError
        except (AttributeError, ValueError):
            # Failing this, check for expected data_vars
            var_names = ["PSI", "TIME", "TE", "TI", "NE", "VTOR"]
            if not np.all(np.isin(var_names, list(data.keys()))):
                raise ValueError(
                    f"KineticsReaderJETTO was provided an invalid JETTO file: {filename}"
                )
