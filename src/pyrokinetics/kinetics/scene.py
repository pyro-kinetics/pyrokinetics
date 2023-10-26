from ..typing import PathLike
from .kinetics import Kinetics
from ..species import Species
from ..constants import electron_mass, deuterium_mass
from ..file_utils import FileReader

import numpy as np
import xarray as xr
from ..units import ureg as units, UnitSpline


class KineticsReaderSCENE(FileReader, file_type="SCENE", reads=Kinetics):
    def read_from_file(self, filename: PathLike) -> Kinetics:
        """Reads NetCDF file from SCENE code. Assumes 3 species: e, D, T"""
        # Open data file, get generic data
        with xr.open_dataset(filename) as kinetics_data:
            psi = kinetics_data["Psi"][::-1]
            psi_n = (psi / psi.isel(rho_psi=-1)).pint.quantify(units.dimensionless)

            rho = kinetics_data["TGLF_RMIN"][::-1].pint.quantify(
                units.lref_minor_radius
            )

            rho_func = UnitSpline(psi_n, rho)

            unit_charge_array = np.ones(len(psi_n))

            # Determine electron data
            electron_temp_data = kinetics_data["Te"][::-1] * units.eV
            electron_temp_func = UnitSpline(psi_n, electron_temp_data)

            electron_density_data = kinetics_data["Ne"][::-1] * units.meter**-3
            electron_density_func = UnitSpline(psi_n, electron_density_data)

            electron_omega_data = (
                electron_temp_data.pint.dequantify() * 0.0 / units.second
            )
            electron_omega_func = UnitSpline(psi_n, electron_omega_data)

            electron_charge = UnitSpline(
                psi_n, -1 * unit_charge_array * units.elementary_charge
            )

            electron = Species(
                species_type="electron",
                charge=electron_charge,
                mass=electron_mass,
                dens=electron_density_func,
                temp=electron_temp_func,
                omega0=electron_omega_func,
                rho=rho_func,
            )

            # Determine ion data
            ion_temperature_func = electron_temp_func
            ion_omega_func = electron_omega_func

            ion_density_func = UnitSpline(psi_n, electron_density_data / 2)

            deuterium_charge = UnitSpline(
                psi_n, 1 * unit_charge_array * units.elementary_charge
            )

            deuterium = Species(
                species_type="deuterium",
                charge=deuterium_charge,
                mass=deuterium_mass,
                dens=ion_density_func,
                temp=ion_temperature_func,
                omega0=ion_omega_func,
                rho=rho_func,
            )

            tritium_charge = UnitSpline(
                psi_n, 1 * unit_charge_array * units.elementary_charge
            )

            tritium = Species(
                species_type="tritium",
                charge=tritium_charge,
                mass=1.5 * deuterium_mass,
                dens=ion_density_func,
                temp=ion_temperature_func,
                omega0=ion_omega_func,
                rho=rho_func,
            )

            # Return dict of species
            return Kinetics(
                kinetics_type="SCENE",
                electron=electron,
                deuterium=deuterium,
                tritium=tritium,
            )

    def verify_file_type(self, filename: PathLike) -> None:
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
