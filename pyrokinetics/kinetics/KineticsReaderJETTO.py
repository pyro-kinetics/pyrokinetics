from pathlib import Path
from typing import Union, Dict
from .KineticsReader import KineticsReader
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass

# Can't use xarray, as JETTO has a variable called X which itself has a dimension called X
import netCDF4 as nc
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class KineticsReaderJETTO(KineticsReader):

    impurity_charge_to_mass = dict(
        zip(
            [2, 6, 8, 10, 18, 54, 74],
            [4, 12, 16, 20, 40, 132, 184],
        )
    )

    impurity_mass_to_charge = {
        value: key for key, value in impurity_charge_to_mass.items()
    }

    def read(self, filename: Union[str, Path]) -> Dict[str, Species]:
        """
        Reads in JETTO profiles NetCDF file
        """
        # Open data file, get generic data
        with nc.Dataset(filename) as kinetics_data:
            psi = kinetics_data["PSI"][-1, :].data
            psi = psi - psi[0]
            psi_n = psi / psi[-1]

            rho = kinetics_data["RMNMP"][-1, :].data
            rho_func = InterpolatedUnivariateSpline(psi_n, rho)

            # Electron data
            electron_temp_data = kinetics_data["TE"][-1, :].data
            electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

            electron_dens_data = kinetics_data["NE"][-1, :].data
            electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

            rotation_data = kinetics_data["VTOR"][-1, :].data
            rotation_func = InterpolatedUnivariateSpline(psi_n, rotation_data)

            electron = Species(
                species_type="electron",
                charge=-1,
                mass=electron_mass,
                dens=electron_dens_func,
                temp=electron_temp_func,
                rot=rotation_func,
                rho=rho_func,
            )

            result = {"electron": electron}

            # JETTO only has one ion temp
            ion_temp_data = kinetics_data["TI"][-1, :].data
            ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data)

            # Go through each species output in JETTO
            try:
                impurity_charge = int(kinetics_data["ZIA1"][-1, 0].data)
                impurity_mass = (
                    self.impurity_charge_to_mass[impurity_charge] * hydrogen_mass
                )
            except IndexError:
                impurity_charge = 0
                impurity_mass = 0

            possible_species = [
                {
                    "species_name": "deuterium",
                    "jetto_name": "NID",
                    "charge": 1,
                    "mass": deuterium_mass,
                },
                {
                    "species_name": "tritium",
                    "jetto_name": "NIT",
                    "charge": 1,
                    "mass": 1.5 * deuterium_mass,
                },
                {
                    "species_name": "helium",
                    "jetto_name": "NALF",
                    "charge": 2,
                    "mass": 4 * hydrogen_mass,
                },
                {
                    "species_name": "impurity",
                    "jetto_name": "NIMP",
                    "charge": impurity_charge,
                    "mass": impurity_mass,
                },
            ]

            for species in possible_species:
                density_data = kinetics_data[species["jetto_name"]][-1, :].data
                if not any(density_data):
                    continue

                density_func = InterpolatedUnivariateSpline(psi_n, density_data)

                result[species["species_name"]] = Species(
                    species_type=species["species_name"],
                    charge=species["charge"],
                    mass=species["mass"],
                    dens=density_func,
                    temp=ion_temp_func,
                    rot=rotation_func,
                    rho=rho_func,
                )

            return result

    def verify(self, filename: Union[str, Path]) -> None:
        """Quickly verify that we're looking at a JETTO file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            data = nc.Dataset(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderJETTO could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"KineticsReaderJETTO must be provided a NetCDF, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the attribute 'description'
        try:
            description = data.description
            if "JETTO" not in description:
                raise ValueError
        except (AttributeError,ValueError):
            # Failing this, check for expected data_vars
            var_names = ["PSI", "RMNMP", "TE", "TI", "NE", "VTOR"]
            if not np.all(np.isin(var_names, list(data.variables))):
                raise ValueError(
                    f"KineticsReaderJETTO was provided an invalid NetCDF: {filename}"
                )
        finally:
            data.close()
