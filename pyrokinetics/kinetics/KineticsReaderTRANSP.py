from typing import Dict
from ..typing import PathLike
from .KineticsReader import KineticsReader
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass

# Can't use xarray, as TRANSP has a variable called X which itself has a dimension called X
import netCDF4 as nc
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class KineticsReaderTRANSP(KineticsReader):
    def read(
        self, filename: PathLike, time_index: int = -1, time: float = None
    ) -> Dict[str, Species]:
        """
        Reads in TRANSP profiles NetCDF file
        """
        # Open data file, get generic data
        with nc.Dataset(filename) as kinetics_data:

            time_cdf = kinetics_data["TIME3"][:]

            if time_index != -1 and time is not None:
                raise ValueError("Cannot set both `time` and `time_index`")

            if time is not None:
                time_index = np.argmin(np.abs(time_cdf - time))

            psi = kinetics_data["PLFLX"][time_index, :].data
            psi = psi - psi[0]
            psi_n = psi / psi[-1]

            rho = kinetics_data["RMNMP"][time_index, :].data
            rho = rho / rho[-1]

            rho_func = InterpolatedUnivariateSpline(psi_n, rho)

            electron_temp_data = kinetics_data["TE"][time_index, :].data
            electron_temp_func = InterpolatedUnivariateSpline(psi_n, electron_temp_data)

            electron_dens_data = kinetics_data["NE"][time_index, :].data * 1e6
            electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

            try:
                omega_data = kinetics_data["OMEG_VTR"][time_index, :].data
            except IndexError:
                omega_data = electron_dens_data * 0.0

            omega_func = InterpolatedUnivariateSpline(psi_n, omega_data)

            electron = Species(
                species_type="electron",
                charge=-1,
                mass=electron_mass,
                dens=electron_dens_func,
                temp=electron_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

            result = {"electron": electron}

            # TRANSP only has one ion temp
            ion_temp_data = kinetics_data["TI"][time_index, :].data
            ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data)

            # Go through each species output in TRANSP
            try:
                impurity_charge = int(kinetics_data["XZIMP"][time_index].data)
                impurity_mass = (
                    int(kinetics_data["AIMP"][time_index].data) * hydrogen_mass
                )
            except IndexError:
                impurity_charge = 0
                impurity_mass = 0

            possible_species = [
                {
                    "species_name": "deuterium",
                    "transp_name": "ND",
                    "charge": 1,
                    "mass": deuterium_mass,
                },
                {
                    "species_name": "tritium",
                    "transp_name": "NT",
                    "charge": 1,
                    "mass": 1.5 * deuterium_mass,
                },
                {
                    "species_name": "helium",
                    "transp_name": "NI4",
                    "charge": 2,
                    "mass": 4 * hydrogen_mass,
                },
                {
                    "species_name": "helium3",
                    "transp_name": "NI4",
                    "charge": 2,
                    "mass": 3 * hydrogen_mass,
                },
                {
                    "species_name": "impurity",
                    "transp_name": "NIMP",
                    "charge": impurity_charge,
                    "mass": impurity_mass,
                },
            ]

            for species in possible_species:
                if species["transp_name"] not in kinetics_data.variables:
                    continue

                density_data = (
                    kinetics_data[species["transp_name"]][time_index, :].data * 1e6
                )
                density_func = InterpolatedUnivariateSpline(psi_n, density_data)

                result[species["species_name"]] = Species(
                    species_type=species["species_name"],
                    charge=species["charge"],
                    mass=species["mass"],
                    dens=density_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rho=rho_func,
                )

            return result

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a TRANSP file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            data = nc.Dataset(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderTRANSP could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"KineticsReaderTRANSP must be provided a NetCDF, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the attribute TRANSP_version
        try:
            data.TRANSP_version
        except AttributeError:
            # Failing this, check for expected data_vars
            var_names = ["TIME3", "PLFLX", "RMNMP", "TE", "TI", "NE"]
            if not np.all(np.isin(var_names, list(data.variables))):
                raise ValueError(
                    f"KineticsReaderTRANSP was provided an invalid NetCDF: {filename}"
                )
        finally:
            data.close()
