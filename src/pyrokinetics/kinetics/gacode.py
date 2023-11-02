from ..typing import PathLike
from .kinetics import Kinetics
from ..species import Species
from ..constants import electron_mass, deuterium_mass
from ..file_utils import FileReader

import numpy as np
from ..units import ureg as units, UnitSpline
from pygacode import expro
import subprocess


class KineticsReaderGACODE(FileReader, file_type="GACODE", reads=Kinetics):
    def read_from_file(
        self,
        filename: PathLike,
    ) -> Kinetics:
        """
        Reads in GACODE profiles and creates Kinetics object

        Parameters
        ----------
        filename: PathLike
            Path to the input.gacode file.

        Raises
        ------
        ValueError
            If ``filename`` is not a valid file or if nr or nz are negative.

        Returns
        -------
        Kinetics

        """

        # Calls fortran code which can cause segfault so need to run subprocess
        # to catch any erros
        read_gacode = f"from pygacode import expro; expro.expro_read('{filename}', 0)"
        try:
            subprocess.run(["python", "-c", read_gacode], check=True)
        except subprocess.CalledProcessError:
            raise ValueError(f"KineticsReaderGACODE could not read {filename}")

        # Open data file, get generic data
        expro.expro_read(filename, 0)

        psi = expro.expro_polflux
        psi_n = psi / psi[-1] * units.dimensionless

        unit_charge_array = np.ones(len(psi_n))

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
            omega0=omega_func,
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
                omega0=omega_func,
                rho=rho_func,
            )

        return Kinetics(kinetics_type="GACODE", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a GACODE file without processing"""
        # Try opening data file
        # If it doesn't exist or isn't netcdf, this will fail
        read_gacode = f"from pygacode import expro; expro.expro_read('{filename}', 0)"
        try:
            subprocess.run(["python", "-c", read_gacode], check=True)
        except subprocess.CalledProcessError:
            raise ValueError(f"KineticsReaderGACODE could not read {filename}")
