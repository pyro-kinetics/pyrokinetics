from typing import Dict
from ..typing import PathLike
from .KineticsReader import KineticsReader
from ..species import Species
from ..constants import electron_mass, deuterium_mass

import numpy as np
import xarray as xr
from scipy.interpolate import InterpolatedUnivariateSpline


class KineticsReaderSCENE(KineticsReader):
    def read(self, filename: PathLike) -> Dict[str, Species]:
        """Reads NetCDF file from SCENE code. Assumes 3 species: e, D, T"""
        # Open data file, get generic data
        with xr.open_dataset(filename) as kinetics_data:

            psi = kinetics_data["Psi"][::-1]
            psi_n = psi / psi[-1]

            rho = kinetics_data["TGLF_RMIN"][::-1]
            rho_func = InterpolatedUnivariateSpline(psi_n, rho)

            # Determine electron data
            electron_temp_data = kinetics_data["Te"][::-1]
            electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

            electron_density_data = kinetics_data["Ne"][::-1]
            electron_density_func = InterpolatedUnivariateSpline(
                psi_n, electron_density_data
            )

            electron_rotation_data = electron_temp_data * 0.0
            electron_rotation_func = InterpolatedUnivariateSpline(
                psi_n, electron_rotation_data
            )

            electron = Species(
                species_type="electron",
                charge=-1,
                mass=electron_mass,
                dens=electron_density_func,
                temp=electron_temp_func,
                rot=electron_rotation_func,
                rho=rho_func,
            )

            # Determine ion data
            ion_temperature_func = electron_temp_func
            ion_rotation_func = electron_rotation_func

            ion_density_func = InterpolatedUnivariateSpline(
                psi_n, electron_density_data / 2
            )

            deuterium = Species(
                species_type="deuterium",
                charge=1,
                mass=deuterium_mass,
                dens=ion_density_func,
                temp=ion_temperature_func,
                rot=ion_rotation_func,
                rho=rho_func,
            )

            tritium = Species(
                species_type="tritium",
                charge=1,
                mass=1.5 * deuterium_mass,
                dens=ion_density_func,
                temp=ion_temperature_func,
                rot=ion_rotation_func,
                rho=rho_func,
            )

            # Return dict of species
            return {
                "electron": electron,
                "deuterium": deuterium,
                "tritium": tritium,
            }

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a SCENE file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            data = xr.open_dataset(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderSCENE could not find {filename}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"KineticsReaderSCENE must be provided a NetCDF, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the expected data_vars
        try:
            software_name = data.software_name
            if "SCENE" not in software_name:
                raise ValueError
        except (AttributeError, ValueError):
            # Failing this, check for expected variables
            if not np.all(np.isin(["Psi", "TGLF_RMIN", "Te", "Ne"], data.data_vars)):
                raise ValueError(
                    f"KineticsReaderSCENE was provided an invalid NetCDF: {filename}"
                )
        finally:
            data.close()
