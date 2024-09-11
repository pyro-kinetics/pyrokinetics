from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np
from periodictable import elements

from ..constants import electron_mass, hydrogen_mass
from ..equilibrium import Equilibrium
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics


class KineticsReaderIMAS(FileReader, file_type="IMAS", reads=Kinetics):
    def read_from_file(
        self,
        filename: PathLike,
        time_index: int = -1,
        time: float = None,
        eq: Equilibrium = None,
    ) -> Kinetics:
        """
        Reads in IMAS profiles NetCDF file
        """

        if eq is None:
            raise ValueError(
                dedent(
                    f"""\
                    {self.__class__.__name__} must be provided with an Equilibrium object via
                    the keyword argument 'eq'. Please load an Equilibrium.
                    """
                )
            )

        if time_index is not None and time is not None:
            raise RuntimeError("Cannot set both 'time' and 'time_index'")

        with h5py.File(filename, "r") as raw_file:

            data = raw_file["core_profiles"]

            time_h5 = data["time"][:]
            if time_index is None:
                time_index = -1 if time is None else np.argmin(np.abs(time_h5 - time))

            psi = data["profiles_1d[]&grid&psi"][time_index]
            psi = psi - psi[0]
            psi_n = psi / psi[-1] * units.dimensionless

            unit_charge_array = np.ones(len(psi_n))

            rho = eq.rho(psi_n) * units.lref_minor_radius

            rho_func = UnitSpline(psi_n, rho)

            electron_temp_data = (
                data["profiles_1d[]&electrons&temperature"][time_index, ...] * units.eV
            )
            electron_temp_func = UnitSpline(psi_n, electron_temp_data)

            electron_dens_data = (
                data["profiles_1d[]&electrons&density_thermal"][time_index, ...]
                * units.meter**-3
            )
            electron_dens_func = UnitSpline(psi_n, electron_dens_data)

            if "profiles_1d[]&ion[]&rotation_frequency_tor" in data.keys():
                omega_data = (
                    data["profiles_1d[]&ion[]&rotation_frequency_tor"][
                        time_index,
                        0,
                    ]
                    * units.second**-1
                )
            elif "profiles_1d[]&ion[]&velocity&toroidal" in data.keys():
                Rmaj = eq.R_major(psi_n).m
                omega_data = (
                    data["profiles_1d[]&ion[]&velocity&toroidal"][
                        time_index,
                        0,
                    ]
                    / Rmaj
                    * units.second**-1
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

            # IMAS only has one ion temp

            ion_full_temp_data = (
                data["profiles_1d[]&ion[]&temperature"][time_index, ...] * units.eV
            )

            n_ions = ion_full_temp_data.shape[0]
            unit_array = np.ones(len(psi_n))

            for i_ion in range(n_ions):
                ion_temp_data = ion_full_temp_data[i_ion, :]
                ion_temp_func = UnitSpline(psi_n, ion_temp_data)

                ion_dens_data = (
                    data["profiles_1d[]&ion[]&density"][time_index, i_ion]
                    / units.meter**3
                )
                ion_dens_func = UnitSpline(psi_n, ion_dens_data)

                ion_charge_data = data["profiles_1d[]&ion[]&element[]&z_n"][
                    time_index, i_ion, 0
                ]
                ion_charge_func = UnitSpline(
                    psi_n, ion_charge_data * unit_array * units.elementary_charge
                )

                ion_mass = (
                    data["profiles_1d[]&ion[]&element[]&a"][time_index, i_ion, 0]
                    * hydrogen_mass
                )
                ion_name = data["profiles_1d[]&ion[]&label"][time_index, i_ion].decode(
                    "utf-8"
                )

                ion_name = ion_name.split("+")[0]
                try:
                    ion_name = getattr(elements, ion_name).name
                except AttributeError:
                    ion_name = ion_name

                result[ion_name] = Species(
                    species_type=ion_name,
                    charge=ion_charge_func,
                    mass=ion_mass,
                    dens=ion_dens_func,
                    temp=ion_temp_func,
                    omega0=omega_func,
                    rho=rho_func,
                )

        return Kinetics(kinetics_type="IMAS", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a IMAS file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(filename)
        try:
            raw_data = h5py.File(filename, "r")
        except Exception as exc:
            raise ValueError("Couldn't read IMAS file. Is the format correct?") from exc
        # Check that the correct variables exist
        if "core_profiles" not in list(raw_data.keys()):
            raise ValueError(
                "IMAS file was missing electron density data. Is the format correct?"
            )
