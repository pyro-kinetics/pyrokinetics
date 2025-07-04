import numpy as np
from path import Path

from ..constants import deuterium_mass, electron_mass
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics

test_keys = ["ne", "ni", "te", "ti"]


def read_gacode_file(filename: PathLike):
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    data_object = GACODEProfiles()

    data_object.units = {}
    current_key = None
    data_dict = {}

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                current_key = line[1:].strip()

                if "|" in current_key:
                    split_str = current_key.split("|")
                    current_key = split_str[0].strip()
                    data_object.units[current_key] = split_str[1].strip()

                setattr(data_object, current_key, [])
                data_dict[current_key] = []
            elif current_key:
                if line:
                    # Convert data to numpy array of floats or strings
                    data = []
                    for item in line.split():
                        try:
                            data.append(float(item))
                        except ValueError:
                            data.append(item)
                    # Check if data has two columns
                    if len(data) == 1 or current_key in ["mass", "name", "z"]:
                        data_dict[current_key].extend(data)
                    else:
                        data_dict[current_key].append(data[1:])

    # Check if relevant keys exist
    if len(set(test_keys).intersection(data_dict.keys())) != len(test_keys):
        raise ValueError("EquilibriumReaderGACODE could not find all relevant keys")

    for key, value in data_dict.items():
        # If data has two columns, convert to 2D array
        setattr(data_object, key, np.squeeze(np.array(value)))

    return data_object


class GACODEProfiles:
    def __init__(self):
        self.name = None
        self.nion = None
        self.z = None
        self.mass = None
        self.ni = None
        self.ti = None
        self.w0 = None
        self.ne = None
        self.te = None
        self.rho = None


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
            If ``filename`` is not a valid file.

        Returns
        -------
        Kinetics

        """

        profiles = read_gacode_file(filename)

        psi = profiles.polflux
        psi_n = psi / psi[-1] * units.dimensionless

        unit_charge_array = np.ones(len(psi_n))

        rho = profiles.rho * units.lref_minor_radius

        rho_func = UnitSpline(psi_n, rho)

        electron_temp_data = profiles.te * units.keV
        electron_temp_func = UnitSpline(psi_n, electron_temp_data)

        electron_dens_data = profiles.ne * 1e19 * units.meter**-3
        electron_dens_func = UnitSpline(psi_n, electron_dens_data)

        omega_data = profiles.w0 * units.radians / units.second

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

        ion_temp_data = profiles.ti * units.keV
        ion_dens_data = profiles.ni * 1e19 * units.meter**-3

        # TODO not always deuterium
        ion_mass_data = profiles.mass * deuterium_mass
        ion_charge_data = profiles.z
        ion_name_data = [name.strip().lower() for name in profiles.name if name]
        n_ion = int(profiles.nion)

        for i_ion in range(n_ion):
            ion_temp_func = UnitSpline(psi_n, ion_temp_data[:, i_ion])
            ion_dens_func = UnitSpline(psi_n, ion_dens_data[:, i_ion])
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
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(filename)
        try:
            profiles = read_gacode_file(filename)
            profile_keys = [hasattr(profiles, prof) for prof in test_keys]
            if not np.all(profile_keys):
                raise ValueError(
                    "EquilibriumReaderGACODE could not find all relevant keys"
                )
        except ValueError:
            raise ValueError(f"EquilibriumReaderGACODE could not find {filename}")
