from ..typing import PathLike
from .kinetics import Kinetics
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass
from ..file_utils import FileReader

# Can't use xarray, as TRANSP has a variable called X which itself has a dimension called X
import numpy as np
from ..units import ureg as units, UnitSpline
from pygacode import expro


class KineticsReaderGACODE(FileReader, file_type="GACODE", reads=Kinetics):
    def read_from_file(
        self,
        filename: PathLike,
    ) -> Kinetics:
        """
        Reads in GACODE profiles
        """
        # Open data file, get generic data
        expro.expro_read(filename, 0)

        psi = expro.expro_polflux
        psi_n = psi / psi[-1] * units.dimensionless

        unit_charge_array = np.ones(len(psi_n))

        rmin = expro.expro_rmin * units.meter
        rho = expro.expro_rho * units.lref_minor_radius

        rho_func = UnitSpline(psi_n, rho)

        electron_temp_data = expro.expro_te * units.keV
        electron_temp_func = UnitSpline(psi_n, electron_temp_data)

        electron_dens_data = expro.expro_ne * 1e19 * units.meter**-3
        electron_dens_func = UnitSpline(psi_n, electron_dens_data)

        omega_data = expro.expro_w0 * units.radians / units.second

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
            ang=omega_func,
            rho=rho_func,
        )

        result = {"electron": electron}

        ion_temp_data = expro.expro_ti * units.keV
        ion_dens_data = expro.expro_ni * 1e19 * units.meter**-3

        # TODO not always deuterium
        ion_mass_data = expro.expro_mass * deuterium_mass
        ion_charge_data = expro.expro_z
        ion_name_data = [
            name.decode().strip().lower() for name in expro.expro_name if name
        ]
        n_ion = expro.expro_n_ion

        for i_ion in range(n_ion):
            ion_temp_func = UnitSpline(psi_n, ion_temp_data[i_ion, :])
            ion_dens_func = UnitSpline(psi_n, ion_dens_data[i_ion, :])
            ion_charge_func = UnitSpline(
                psi_n,
                ion_charge_data[i_ion] * unit_charge_array * units.elementary_charge,
            )

            result[ion_name_data[i_ion]] = Species(
                species_type=ion_name_data[i_ion],
                charge=ion_charge_func,
                mass=ion_mass_data[i_ion],
                dens=ion_dens_func,
                temp=ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

        return Kinetics(kinetics_type="GACODE", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GACODE file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        try:
            expro.expro_read(filename, 0)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderGACODE could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"KineticsReadeGACODE must be provided a GACODE file, was given {filename}"
            ) from e
        # Given it is a netcdf, check it has the attribute TRANSP_version
        try:
            expro.expro_name
        except AttributeError:
            raise ValueError(
                f"KineticsReaderGACODE was not able to read {filename} using pygacode"
            )
