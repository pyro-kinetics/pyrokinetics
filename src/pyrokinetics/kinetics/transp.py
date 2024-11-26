# Can't use xarray, as TRANSP has a variable called X which itself has a dimension called X
import netCDF4 as nc
import numpy as np

from ..constants import deuterium_mass, electron_mass, hydrogen_mass
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics

species_mapping = {
    "C": ["carbon", 12.0],
    "BE": ["beryllium", 9.0],
    "O": ["oxygen", 16.0],
    "NE": ["neon", 20.0],
    "AR": ["argon", 40.0],
    "W": ["tungsten", 184.0],
}


class KineticsReaderTRANSP(FileReader, file_type="TRANSP", reads=Kinetics):
    def read_from_file(
        self, filename: PathLike, time_index: int = -1, time: float = None
    ) -> Kinetics:
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
            psi_n = psi / psi[-1] * units.dimensionless

            unit_charge_array = np.ones(len(psi_n))

            rho = kinetics_data["RMNMP"][time_index, :].data
            rho = rho / rho[-1] * units.lref_minor_radius

            rho_func = UnitSpline(psi_n, rho)

            electron_temp_data = kinetics_data["TE"][time_index, :].data * units.eV
            electron_temp_func = UnitSpline(psi_n, electron_temp_data)

            electron_dens_data = (
                kinetics_data["NE"][time_index, :].data * 1e6 * units.meter**-3
            )
            electron_dens_func = UnitSpline(psi_n, electron_dens_data)

            if "OMEG_VTR" in kinetics_data.variables.keys():
                omega_data = (
                    kinetics_data["OMEG_VTR"][time_index, :].data * units.second**-1
                )
            elif "OMEGA" in kinetics_data.variables.keys():
                omega_data = (
                    kinetics_data["OMEGA"][time_index, :].data * units.second**-1
                )
            else:
                omega_data = electron_dens_data.m * 0.0 * units.second**-1

            omega_func = UnitSpline(psi_n, omega_data)

            electron_charge = UnitSpline(
                psi_n, -1 * unit_charge_array * units.elementary_charge
            )

            electron = Species(
                species_type="electron",
                charge=electron_charge,
                mass=electron_mass,
                dens=electron_dens_func,
                temp=electron_temp_func,
                omega0=omega_func,
                rho=rho_func,
            )

            result = {"electron": electron}

            # TRANSP only has one ion temp
            ion_temp_data = kinetics_data["TI"][time_index, :].data * units.eV
            ion_temp_func = UnitSpline(psi_n, ion_temp_data)

            possible_species = [
                {
                    "species_name": "hydrogen",
                    "transp_name": "NH",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": hydrogen_mass,
                },
                {
                    "species_name": "deuterium",
                    "transp_name": "ND",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": deuterium_mass,
                },
                {
                    "species_name": "tritium",
                    "transp_name": "NT",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": 1.5 * deuterium_mass,
                },
                {
                    "species_name": "helium",
                    "transp_name": "NI4",
                    "charge": UnitSpline(
                        psi_n, 2 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": 4 * hydrogen_mass,
                },
                {
                    "species_name": "helium3",
                    "transp_name": "NI3",
                    "charge": UnitSpline(
                        psi_n, 2 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": 3 * hydrogen_mass,
                },
            ]

            for species in possible_species:
                if species["transp_name"] not in kinetics_data.variables:
                    continue

                density_data = (
                    kinetics_data[species["transp_name"]][time_index, :].data
                    * 1e6
                    * units.meter**-3
                )
                density_func = UnitSpline(psi_n, density_data)

                result[species["species_name"]] = Species(
                    species_type=species["species_name"],
                    charge=species["charge"],
                    mass=species["mass"],
                    dens=density_func,
                    temp=ion_temp_func,
                    omega0=omega_func,
                    rho=rho_func,
                )

            # Add in impurities
            impurity_keys = [key for key in kinetics_data.variables if "NIMP_" in key]
            if "NIMP_NC" in impurity_keys:
                impurity_keys.remove("NIMP_NC")

            # Go through each species output in TRANSP
            for impurity_key in impurity_keys:

                split_name = impurity_key.split("_")

                if split_name[-1] == "SINGL":
                    name = "impurity"
                    impurity_charge = int(kinetics_data["XZIMP"][time_index].data)
                    impurity_charge_func = UnitSpline(
                        psi_n,
                        impurity_charge * unit_charge_array * units.elementary_charge,
                    )

                    impurity_dens_data = (
                        kinetics_data["NIMP"][time_index, :].data * 1e6 * units.m**-3
                    )
                    impurity_dens_func = UnitSpline(psi_n, impurity_dens_data)

                    impurity_temp_data = (
                        kinetics_data["TX"][time_index, :].data * units.eV
                    )
                    impurity_temp_func = UnitSpline(psi_n, impurity_temp_data)

                    impurity_mass = (
                        int(kinetics_data["AIMP"][time_index].data) * hydrogen_mass
                    )

                else:

                    element = split_name[1]
                    name = species_mapping[element][0]

                    impurity_charge_data = (
                        kinetics_data[f"ZIMPS_{element}"][time_index, :].data
                        * units.elementary_charge
                    )
                    impurity_charge_func = UnitSpline(psi_n, impurity_charge_data)

                    impurity_dens_data = (
                        kinetics_data[f"NIMP_{split_name[1]}_{split_name[2]}"][
                            time_index, :
                        ].data
                        * 1e6
                        * units.m**-3
                    )
                    impurity_dens_func = UnitSpline(psi_n, impurity_dens_data)

                    impurity_temp_data = (
                        kinetics_data["TX"][time_index, :].data * units.eV
                    )
                    impurity_temp_func = UnitSpline(psi_n, impurity_temp_data)

                    impurity_mass = species_mapping[element][1] * hydrogen_mass

                result[name] = Species(
                    species_type=name,
                    charge=impurity_charge_func,
                    mass=impurity_mass,
                    dens=impurity_dens_func,
                    temp=impurity_temp_func,
                    omega0=omega_func,
                    rho=rho_func,
                )

            possible_fast_species = [
                {
                    "species_name": "hydrogen_fast",
                    "transp_name": "BDENS_H",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": hydrogen_mass,
                },
                {
                    "species_name": "deuterium_fast",
                    "transp_name": "BDENS_D",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": deuterium_mass,
                },
                {
                    "species_name": "tritium_fast",
                    "transp_name": "BDENS_T",
                    "charge": UnitSpline(
                        psi_n, 1 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": 1.5 * deuterium_mass,
                },
                {
                    "species_name": "alpha",
                    "transp_name": "FDENS_4",
                    "charge": UnitSpline(
                        psi_n, 2 * unit_charge_array * units.elementary_charge
                    ),
                    "mass": 4 * hydrogen_mass,
                },
            ]

            for species in possible_fast_species:
                if species["transp_name"] not in kinetics_data.variables:
                    continue

                density_data = (
                    kinetics_data[species["transp_name"]][time_index, :].data
                    * 1e6
                    * units.meter**-3
                )

                density_func = UnitSpline(psi_n, density_data)

                prefix = species["transp_name"][0]
                suffix = species["transp_name"].split("_")[-1]

                # Work out fast ion pressure
                pressure_data = (
                    (
                        0.5
                        * kinetics_data[f"U{prefix}PRP_{suffix}"][time_index, :].data
                        + kinetics_data[f"U{prefix}PAR_{suffix}"][time_index, :].data
                    )
                    * units.joules
                    / units.cm**3
                )

                # Take "temperature" as ratio of pressure to density
                fast_ion_temp_data = (pressure_data / density_data).to("eV")

                fast_ion_temp_func = UnitSpline(psi_n, fast_ion_temp_data)

                result[species["species_name"]] = Species(
                    species_type=species["species_name"],
                    charge=species["charge"],
                    mass=species["mass"],
                    dens=density_func,
                    temp=fast_ion_temp_func,
                    omega0=omega_func,
                    rho=rho_func,
                )

            return Kinetics(kinetics_type="TRANSP", **result)

    def verify_file_type(self, filename: PathLike) -> None:
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
