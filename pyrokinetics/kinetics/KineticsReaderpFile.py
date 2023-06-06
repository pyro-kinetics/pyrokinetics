"""
Reads in an Osborne pFile: https://omfit.io/_modules/omfit_classes/omfit_osborne.html#OMFITosborneProfile


"""
from typing import Dict
from ..typing import PathLike
from .KineticsReader import KineticsReader
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass
from pyrokinetics.equilibrium.equilibrium import read_equilibrium
from ..units import ureg as units, UnitSpline

import numpy as np
import re
from textwrap import dedent
from contextlib import redirect_stdout
from freeqdsk import peqdsk

def ion_species_selector(nucleons, charge):
    """
    Returns ion species type from:

    hydrogen deuterium, tritium, helium3, helium, other impurity.

    Might need to update with more specific masses, such as 6.94 for Li-7, etc.
    """
    if nucleons == 1:
        if charge.m == 1:
            return "hydrogen"
        else:
            print(
                "You have a species with a single nucleon which is not a proton. Strange. Returning neutron for now. \n"
            )
            return "neutron"
    if nucleons == 2 and charge.m == 1:
        return "deuterium"
    elif nucleons == 4 and charge.m == 2:
        return "helium"
    elif nucleons == 3:
        if charge.m == 1:
            return "tritium"
        if charge.m == 2:
            return "helium3"
    else:
        return "impurity"


def np_to_T(n, p):
    """
    n is in m^{-3}, T is in eV, p is in Pascals.
    Returns temperature in eV.
    """
    return np.divide(p, n).to("eV")


class KineticsReaderpFile(KineticsReader):
    def read(
        self,
        filename: PathLike,
        eq_file: PathLike = None,
    ) -> Dict[str, Species]:
        """
        Reads in Osborne pFile. Your pFile should just be called, pFile.
        Also reads a geqdsk file via eq_file to obtain r/a.
        """
        # eq_file must be provided
        if eq_file is None:
            raise ValueError(
                dedent(
                    f"""\
                    {self.__class__.__name__} must be provided with a G-EQDSK file via
                    the keyword argument 'eq_file'.
                    """
                )
            )

        # Read pFile, get generic data.
        with redirect_stdout(None), open(filename) as f:
            data = peqdsk.read(f)

        profiles = data["profiles"]
        species = data["species"]

        # Interpolate on psi_n.
        te_psi_n = profiles["te"]["psinorm"] * units.dimensionless
        electron_temp_data = profiles["te"]["data"] * 1e3 * units.eV
        electron_temp_func = UnitSpline(te_psi_n, electron_temp_data)

        ne_psi_n = profiles["ne"]["psinorm"] * units.dimensionless
        electron_dens_data = profiles["ne"]["data"] * 1e20 * units.meter**-3
        electron_dens_func = UnitSpline(ne_psi_n, electron_dens_data)

        # Read geqdsk file, obtain rho_func.
        geqdsk_equilibrium = read_equilibrium(str(eq_file))
        rho_g = geqdsk_equilibrium["r_minor"].values * units.lref_minor_radius
        psi_n_g = geqdsk_equilibrium["psi_n"].values * units.dimensionless
        rho_func = UnitSpline(psi_n_g, rho_g)

        if "omeg" in profiles.keys():
            omega_psi_n = profiles["omeg"]["psinorm"] * units.dimensionless
            omega_data = profiles["omeg"]["data"] * 1e3 * units.radians / units.second
        else:
            omega_psi_n = te_psi_n * units.dimensionless
            omega_data = np.zeros(len(omega_psi_n), dtype="float") * units.radians / units.second

        omega_func = UnitSpline(omega_psi_n, omega_data)

        if "vtor1" in profiles.keys():
            rot_psi_n = profiles["vtor1"]["psinorm"] * units.dimensionless
            rotation_data = profiles["vtor1"]["data"] * 1e3 * units.meter / units.second
        else:
            rot_psi_n = te_psi_n * units.dimensionless
            rotation_data = np.zeros(len(rot_psi_n), dtype="float") * units.meter / units.second

        rotation_func = UnitSpline(rot_psi_n, rotation_data)

        electron = Species(
            species_type="electron",
            charge=-1*units.elementary_charge,
            mass=electron_mass,
            dens=electron_dens_func,
            temp=electron_temp_func,
            ang=omega_func,
            rot=rotation_func,
            rho=rho_func,
        )

        result = {"electron": electron}
        num_ions = len(species)

        # Check whether fast particles.
        try:
            if np.all(profiles["nb"]["data"] == 0.0):
                fast_particle = 0
            else:
                fast_particle = 1
        except KeyError:
            fast_particle = 0

        num_thermal_ions = num_ions - fast_particle

        # thermal ions have same temperature in pFile.
        ti_psi_n = profiles["ti"]["psinorm"] * units.dimensionless
        ion_temp_data = profiles["ti"]["data"] * 1e3 * units.eV
        ion_temp_func = UnitSpline(ti_psi_n, ion_temp_data)

        for ion_it in np.arange(num_thermal_ions):
            if ion_it == num_thermal_ions - 1:
                ni_psi_n = profiles["ni"]["psinorm"] * units.dimensionless
                ion_dens_data = profiles["ni"]["data"] * 1e20 * units.meter ** -3
                ion_dens_func = UnitSpline(ni_psi_n, ion_dens_data)

                ion_charge = species[ion_it]["Z"] * units.elementary_charge
                ion_nucleons = species[ion_it]["A"]
                ion_mass = ion_nucleons * hydrogen_mass

                species_name = ion_species_selector(ion_nucleons, ion_charge)

                result[species_name] = Species(
                    species_type=species_name,
                    charge=ion_charge,
                    mass=ion_mass,
                    dens=ion_dens_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rot=rotation_func,
                    rho=rho_func,
                )

            else:
                try:
                    nz_psi_n = profiles[f"nz{ion_it+1}"]["psinorm"] * units.dimensionless
                    impurity_dens_data = profiles[f"nz{ion_it+1}"]["data"] * 1e20 * units.meter ** -3
                except KeyError:
                    nz_psi_n = ni_psi_n
                    impurity_dens_data = ion_dens_data * 0.0

                impurity_dens_func = UnitSpline(nz_psi_n, impurity_dens_data)

                impurity_charge = species[ion_it]["Z"] * units.elementary_charge
                impurity_nucleons = species[ion_it]["A"]
                impurity_mass = impurity_nucleons * hydrogen_mass

                species_name = ion_species_selector(impurity_nucleons, impurity_charge)
                result[species_name] = Species(
                    species_type=species_name,
                    charge=impurity_charge,
                    mass=impurity_mass,
                    dens=impurity_dens_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rot=rotation_func,
                    rho=rho_func,
                )

        if fast_particle == 1:  # Adding the fast particle species.
            nb_psi_n = profiles["nb"]["psinorm"] * units.dimensionless
            fast_ion_dens_data = profiles["nb"]["data"] * 1e20 * units.meter ** -3

            pb_psi_n = profiles["pb"]["psinorm"] * units.dimensionless
            fast_ion_press_data = profiles["pb"]["data"] * 1e3 * units.pascals

            if np.all(pb_psi_n != nb_psi_n):
                fast_ion_press_func = UnitSpline(pb_psi_n, fast_ion_press_data)
                fast_ion_press_data = fast_ion_press_func(nb_psi_n)

            fast_ion_temp_data = np_to_T(fast_ion_dens_data, fast_ion_press_data)

            fast_ion_dens_func = UnitSpline(nb_psi_n, fast_ion_dens_data)
            fast_ion_temp_func = UnitSpline(nb_psi_n, fast_ion_temp_data)

            fast_ion_charge = species[-1]["Z"] * units.elementary_charge
            fast_ion_nucleons = species[-1]["A"]
            fast_ion_mass = ion_nucleons * hydrogen_mass

            fast_species = ion_species_selector(fast_ion_nucleons, fast_ion_charge) + str("_fast")

            result[fast_species] = Species(
                species_type=fast_species,
                charge=fast_ion_charge,
                mass=fast_ion_mass,
                dens=fast_ion_dens_func,
                temp=fast_ion_temp_func,
                ang=omega_func,
                rot=rotation_func,
                rho=rho_func,
            )

        return result

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a pFile file without processing"""
        # Check that the header line looks like a pFile header
        with open(filename) as f:
            header = f.readline().split()
        if not re.match(r"\d*", header[0]):
            raise ValueError("pFile header starts with an int")
        if not header[1] == "psinorm":
            raise ValueError("pFile first column name should be 'psinorm'")
