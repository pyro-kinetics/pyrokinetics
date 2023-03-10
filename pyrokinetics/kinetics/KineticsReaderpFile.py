"""
Reads in an Osborne pFile: https://omfit.io/_modules/omfit_classes/omfit_osborne.html#OMFITosborneProfile

<<<<<<< HEAD
Install OMFIT classes with:

pip install --upgrade omfit_classes

"""
=======
Uses modified pFile reader from eqtools.py.
'''
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)
from typing import Dict
from ..typing import PathLike
from .KineticsReader import KineticsReader
from ..species import Species
from ..constants import electron_mass, hydrogen_mass, deuterium_mass
from ..pfile_reader import PFileReader

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
<<<<<<< HEAD
from omfit_classes.omfit_osborne import (
    OMFITpFile,
)  # Robust pFile reader from OMFIT. See https://omfit.io/_modules/omfit_classes/omfit_osborne.html#OMFITosborneProfile.
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import copy

=======
import csv
import re
from collections import namedtuple
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

def ion_species_selector(nucleons, charge):
    """
    Returns ion species type from:

    hydrogen deuterium, tritium, helium3, helium, other impurity.

    Might need to update with more specific masses, such as 6.94 for Li-7, etc.
    """
    if nucleons == 1:
        if charge == 1:
            return "hydrogen"
        else:
            print(
                "You have a species with a single nucleon which is not a proton. Strange. Returning neutron for now. \n"
            )
            return "neutron"
    if nucleons == 2 and charge == 1:
        return "deuterium"
    elif nucleons == 4 and charge == 2:
        return "helium"
    elif nucleons == 3:
        if charge == 1:
            return "tritium"
        if charge == 2:
            return "helium3"
    else:
        return "impurity"


def np_to_T(n, p):
    """
    n is in 10^{19} m^{-3}, T is in keV, p is in Pascals.
    Returns temperature in keV.
    """
    return np.divide(p, n) / ((1.381e-23) * (1e19) * (11600) * (1000))

<<<<<<< HEAD

def remap_osborne(
    pFile_data_in, points="ne", **kw
):  #### copying from omfit, as bug in my installation.
    mapped = copy.deepcopy(pFile_data_in)
    keys = mapped.keys()

    if isinstance(points, str):
        remap_psin = pFile_data_in[points]["psinorm"]

    elif type(points) is int:  # Evenly spaced grid rebase
        remap_psin = np.linspace(0, 1, points)

    elif type(points) in [list, tuple, np.ndarray]:  # user provided grid rebase
        assert np.all(np.abs(points) <= 1), "All values must be on the interval [0, 1]"
        assert np.all(
            np.diff(points) > 0
        ), "All values must be monotonically increasing"
        remap_psin = points

    mapped["psinorm"] = remap_psin  # save the psin grid
    # iterate through all keys
    for key in keys:
        if key in ("N Z A",):  # this is not mapped to psi so skip it
            continue
        f_key = interp1d(pFile_data_in[key]["psinorm"], pFile_data_in[key]["data"])
        f_key_derivative = interp1d(
            pFile_data_in[key]["psinorm"], pFile_data_in[key]["derivative"], **kw
        )
        mapped[key]["psinorm"] = remap_psin
        mapped[key]["data"] = f_key(remap_psin)
        mapped[key]["derivative"] = f_key_derivative(remap_psin)

    return mapped

=======
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

class KineticsReaderpFile(KineticsReader):
    def read(
        self, filename: PathLike, time_index: int = -1, time: float = None
    ) -> Dict[str, Species]:
        """
        Reads in Osborne pFile. Your pFile should just be called, pFile.

        Also reads a geqdsk file, which is in the same directory as the pFile, and assumed to be called geqdsk.
        """
        # Read pFile, get generic data.

<<<<<<< HEAD
        pFile = OMFITpFile(str(filename))
        ne = remap_osborne(
            pFile, "ne"
        )  # remap to get pFile on same uniform grid for all entries.
        te = remap_osborne(pFile, "te")
        psi_n = ne["ne"]["psinorm"]

        electron_temp_data = te["te"]["data"]
        electron_dens_data = ne["ne"]["data"]
=======
        #pFile = OMFITpFile(str(filename))
        pFile = PFileReader(str(filename))
        ne = remap_osborne(pFile,'ne') # remap to get pFile on same uniform grid for all entries.
        te = remap_osborne(pFile,'te')
        
        psi_n = pFile.__getattribute__('ne').x

        electron_temp_data = pFile.__getattribute__('ne').y
        electron_dens_data = pFile.__getattribute__('te').y
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

        electron_temp_func = InterpolatedUnivariateSpline(
            psi_n, electron_temp_data
        )  # Interpolate on psi_n.
        electron_dens_func = InterpolatedUnivariateSpline(psi_n, electron_dens_data)

        geqdsk_filename = (
            str(filename)[:-5] + "geqdsk"
        )  # Important: geqdsk is in same directory as pFile, and is called geqdsk.
        # How can we get rho = r/a? For now, we read a geqdsk file to get r/a and psinorm.
        rminor_geqdsk = (
            OMFITgeqdsk(geqdsk_filename)["fluxSurfaces"]["avg"]["a"] / 2
        )  # This is the half-diameter of each flux surface.
        psinorm_geqdsk = OMFITgeqdsk(geqdsk_filename)["AuxQuantities"]["PSI_NORM"]
        # we can also get rho_t = sqrt(toroidal flux_norm): rhonorm_geqdsk = omfitgeqdsk(geqdsk_filename)['AuxQuantities']['RHO']
        rho_geqdsk = rminor_geqdsk / rminor_geqdsk[-1]

        # We next interpolate find rho_func for the pFile. This is probably overkill.
        rho_geqdsk_interp = InterpolatedUnivariateSpline(psinorm_geqdsk, psinorm_geqdsk)
        rho_pFile = rho_geqdsk_interp(psi_n)
        rho_func = InterpolatedUnivariateSpline(psi_n, rho_pFile)

<<<<<<< HEAD
        try:
            omega = remap_osborne(pFile, "omega")
            omega_data = omega["omega"]["data"]
        except Exception:
            omega_data = np.zeros(len(psi_n), dtype="float")
=======
        if 'omeg' in pFile_params:
            omega_data = pFile.__getattribute__('omeg').y
        else:
            omega_data = np.zeros(len(psi_n), dtype = 'float')
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

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

<<<<<<< HEAD
        num_ions = len(pFile["N Z A"]["Z"])

        ## Check whether fast particles.
        fast_particle = 0
        if "nb" in pFile:
=======
        num_ions = pFile.__getattribute__('ions').nions

        ## Check whether fast particles.
        fast_particle = 0
        if 'nb' in pFile_params:
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)
            fast_particle = 1
            print("Fast particles present in pFile \n.")

        num_thermal_ions = num_ions - fast_particle

        # thermal ions have same temperature in pFile.
<<<<<<< HEAD
        ti = remap_osborne(pFile, "ti")
        ion_temp_data = ti["ti"]["data"]
        ion_temp_func = InterpolatedUnivariateSpline(
            psi_n, ion_temp_data
        )  # Interpolate on psi_n.
=======
        ion_temp_data = pFile.__getattribute__('ti').y
        ion_temp_func = InterpolatedUnivariateSpline(psi_n, ion_temp_data) # Interpolate on psi_n.
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

        for ion_it in np.arange(num_thermal_ions):
            """
            --Ordering of ['N Z A'] in pFile is:

            N Z A:
            impurities
            main ion
            fast ion

            --All thermal ions have the same temperature.
            """

            if ion_it == num_thermal_ions - 1:  # Main ion.
                ni = remap_osborne(pFile, "ni")
                ion_dens_data = ni["ni"]["data"]

<<<<<<< HEAD
                charge = pFile["N Z A"]["Z"][ion_it]
                nucleons = pFile["N Z A"]["A"][ion_it]
                mass = pFile["N Z A"]["A"][ion_it] * hydrogen_mass
=======
                ion_dens_data = pFile.__getattribute__('ni').y

                charge = pFile.__getattribute__('ions').Z[ion_it]
                nucleons = pFile.__getattribute__('ions').A[ion_it]
                mass = nucleons * hydrogen_mass
               
                species_name = ion_species_selector(nucleons, charge)
                ion_dens_func = InterpolatedUnivariateSpline(psi_n, ion_dens_data)

                result[species_name] = Species(
                    species_type=species_name,
                    charge=charge,
                    mass=mass,
                    dens=ion_dens_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rho=rho_func,
                )  


            else: # Impurities.

                ion_dens_data = pFile.__getattribute__('nz{}'.format(int(ion_it+1))).y

                charge = pFile.__getattribute__('ions').Z[ion_it]
                nucleons = pFile.__getattribute__('ions').A[ion_it]
                mass = nucleons * hydrogen_mass
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

                species_name = ion_species_selector(nucleons, charge)
                ion_dens_func = InterpolatedUnivariateSpline(psi_n, ion_dens_data)

                result[species_name] = Species(
                    species_type=species_name,
                    charge=charge,
                    mass=mass,
                    dens=ion_dens_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rho=rho_func,
                )

            else:  # Impurities.
                ni = remap_osborne(pFile, "nz{}".format(int(ion_it + 1)))
                ion_dens_data = ni["nz{}".format(int(ion_it + 1))]["data"]

<<<<<<< HEAD
                charge = pFile["N Z A"]["Z"][ion_it]
                nucleons = pFile["N Z A"]["A"][ion_it]
                mass = pFile["N Z A"]["A"][ion_it] * hydrogen_mass

                species_name = ion_species_selector(nucleons, charge)
                ion_dens_func = InterpolatedUnivariateSpline(psi_n, ion_dens_data)

                result[species_name] = Species(
                    species_type=species_name,
                    charge=charge,
                    mass=mass,
                    dens=ion_dens_func,
                    temp=ion_temp_func,
                    ang=omega_func,
                    rho=rho_func,
                )

        if fast_particle:  # Adding the fast particle species.
            nb = remap_osborne(pFile, "nb")
            fast_ion_dens_data = ni["nb"]["data"]
            fast_ion_press_data = ni["pb"]["data"]
=======
            fast_ion_dens_data = pFile.__getattribute__('nb').y
            fast_ion_press_data = pFile.__getattribute__('pb').y
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

            # estimate fast particle temperature from pressure and density. Very approximate.
            fast_ion_temp_data = np_to_T(fast_ion_dens_data, fast_ion_press_data)

            fast_ion_dens_func = InterpolatedUnivariateSpline(psi_n, fast_ion_dens_data)
            fast_ion_temp_func = InterpolatedUnivariateSpline(psi_n, fast_ion_temp_data)

<<<<<<< HEAD
            charge = pFile["N Z A"]["Z"][-1]
            nucleons = pFile["N Z A"]["A"][-1]
            mass = pFile["N Z A"]["A"][-1] * hydrogen_mass
=======
            charge = pFile.__getattribute__('ions').Z[-1]
            nucleons = pFile.__getattribute__('ions').A[-1]
            mass = nucleons * hydrogen_mass
>>>>>>> de16051 (Adding in-house pFile reader adapted from eqtools)

            fast_species = ion_species_selector(nucleons, charge) + str("_fast")

            result[fast_species] = Species(
                species_type=fast_species,
                charge=charge,
                mass=mass,
                dens=fast_ion_dens_func,
                temp=fast_ion_temp_func,
                ang=omega_func,
                rho=rho_func,
            )

        print("result is {} \n".format(result))

    def verify(self, filename: PathLike) -> None:
        """Quickly verify that we're looking at a pFile file without processing"""
        # Try opening data file
        try:
            data = OMFITpFile(str(filename))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"KineticsReaderpFile could not find {filename}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"KineticsReaderpFile must be provided a pFile, was given {filename}"
            ) from e
        ## Given it is a netcdf, check it has the attribute TRANSP_version
        # try:
        #    data.TRANSP_version
        # except AttributeError:
        #    # Failing this, check for expected data_vars
        #    var_names = ["TIME3", "PLFLX", "RMNMP", "TE", "TI", "NE"]
        #    if not np.all(np.isin(var_names, list(data.variables))):
        #        raise ValueError(
        #            f"KineticsReaderTRANSP was provided an invalid NetCDF: {filename}"
        #        )
        finally:
            data.close()
