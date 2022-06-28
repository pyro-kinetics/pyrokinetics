import numpy as np
from cleverdict import CleverDict
from pathlib import Path
from ast import literal_eval
from typing import Dict, Any, Optional
from ..typing import PathLike
from ..constants import pi, electron_charge
from ..local_species import LocalSpecies
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    default_miller_inputs,
)
from ..numerics import Numerics
from ..templates import gk_templates
from .GKInput import GKInput


class GKInputCGYRO(GKInput):
    """
    Class that can read CGYRO input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "CGYRO"
    default_file_name = "input.cgyro"

    pyro_cgyro_miller = {
        "rho": "RMIN",
        "Rmaj": "RMAJ",
        "q": "Q",
        "kappa": "KAPPA",
        "s_kappa": "S_KAPPA",
        "delta": "DELTA",
        "shat": "S",
        "shift": "SHIFT",
    }

    @staticmethod
    def get_pyro_cgyro_species(iSp=1):
        return {
            "mass": f"MASS_{iSp}",
            "z": f"Z_{iSp}",
            "dens": f"DENS_{iSp}",
            "temp": f"TEMP_{iSp}",
            "a_lt": f"DLNTDR_{iSp}",
            "a_ln": f"DLNNDR_{iSp}",
        }

    cgyro_eq_types = {
        1: "SAlpha",
        2: "Miller",
        3: "Fourier",
    }

    def read(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads CGYRO input file into a dictionary
        """
        with open(filename) as f:
            self.data = self.parse_cgyro(f)
        return self.data

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads CGYRO input file given as string
        """
        self.data = self.parse_cgyro(input_string.split("\n"))
        return self.data

    @staticmethod
    def parse_cgyro(lines):
        """
        Given lines of a cgyro file or a string split by '/n', return a dict of
        CGYRO input data
        """
        results = {}
        for line in lines:
            # Get line before comments, remove trailing whitespace
            line = line.split("#")[0].strip()
            # Skip empty lines (this will also skip comment lines)
            if not line:
                continue

            # Splits by =, remove whitespace, store as (key,value) pair
            key, value = (token.strip() for token in line.split("="))

            # Use literal_eval to convert value to int/float/list etc
            # If it fails, assume value should be a string
            try:
                results[key] = literal_eval(value)
            except Exception:
                results[key] = value
        return results

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid cgyro input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a CGYRO input file,
        # but they are needed by Pyrokinetics
        expected_keys = [
            "S_DELTA",
            "BETAE_UNIT",
            "N_SPECIES",
            "NU_EE",
            "N_FIELD",
            "N_RADIAL",
            *self.pyro_cgyro_miller.values(),
        ]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as CGYRO file")

    def write(self, filename: PathLike, float_format: str = ""):
        # Create directories if they don't exist already
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Write self.data
        with open(filename, "w") as f:
            for key, value in self.data.items():
                if isinstance(value, float):
                    line = f"{key} = {value:{float_format}}\n"
                else:
                    line = f"{key} = {value}\n"
                f.write(line)

    def is_nonlinear(self) -> bool:
        return bool(self.data.get("NONLINEAR_FLAG", 0))

    def add_flags(self, flags) -> None:
        """
        Add extra flags to CGYRO input file
        """
        for key, value in flags.items():
            self.data[key] = value

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """
        eq_type = self.cgyro_eq_types[self.data["EQUILIBRIUM_MODEL"]]
        if eq_type != "Miller":
            raise NotImplementedError(
                f"LocalGeometry type {eq_type} not implemented for CGYRO"
            )

        return self.get_local_geometry_miller()

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from CGYRO file
        """
        miller_data = default_miller_inputs()

        for key, val in self.pyro_cgyro_miller.items():
            miller_data[key] = self.data[val]

        miller_data["s_delta"] = self.data["S_DELTA"] / np.sqrt(
            1 - self.data["DELTA"] ** 2
        )

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying miller after creation
        # FIXME Is this assumption general enough? Can't we get pref from local_species?
        # FIXME B0 = None can cause problems when writing
        beta = self.data["BETAE_UNIT"]
        if beta != 0:
            miller.B0 = 1 / (miller.bunit_over_b0 * beta**0.5)
        else:
            miller.B0 = None

        # Need species to set up beta_prime
        local_species = self.get_local_species()
        beta_prime_scale = self.data.get("BETA_STAR_SCALE", 1.0)

        if miller.B0 is not None:
            miller.beta_prime = -local_species.a_lp * beta_prime_scale / miller.B0**2
        else:
            miller.beta_prime = 0.0

        return miller

    def get_local_species(self):
        """
        Load LocalSpecies object from CGYRO file
        """

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(self.data["N_SPECIES"]):

            pyro_cgyro_species = self.get_pyro_cgyro_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_cgyro_species.items():
                species_data[p_key] = self.data[c_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
                species_data.nu = self.data["NU_EE"]
                te = species_data.temp
                ne = species_data.dens
                me = species_data.mass
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        for name in local_species.names:
            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

        # Get collision frequency of ion species
        nu_ee = self.data["NU_EE"]

        for ion in range(ion_count):
            key = f"ion{ion + 1}"

            nion = local_species[key]["dens"]
            tion = local_species[key]["temp"]
            mion = local_species[key]["mass"]
            # Not exact at log(Lambda) does change but pretty close...
            local_species[key]["nu"] = (
                nu_ee
                * (nion / tion**1.5 / mion**0.5)
                / (ne / te**1.5 / me**0.5)
            )

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        nfields = self.data["N_FIELD"]

        numerics_data["phi"] = nfields >= 1
        numerics_data["apar"] = nfields >= 2
        numerics_data["bpar"] = nfields >= 3

        numerics_data["delta_time"] = self.data.get("DELTA_T", 0.01)
        numerics_data["max_time"] = self.data.get("MAX_TIME", 1.0)

        numerics_data["ky"] = self.data["KY"]
        numerics_data["nky"] = self.data.get("N_TOROIDAL", 1)
        numerics_data["theta0"] = 2 * pi * self.data.get("PX0", 0.0)
        numerics_data["nkx"] = self.data.get("N_RADIAL", 1)
        numerics_data["nperiod"] = int(self.data["N_RADIAL"] / 2)

        shat = self.data[self.pyro_cgyro_miller["shat"]]
        box_size = self.data.get("BOX_SIZE", 1)
        numerics_data["kx"] = numerics_data["ky"] * 2 * pi * shat / box_size

        numerics_data["ntheta"] = self.data.get("N_THETA", 24)
        numerics_data["nenergy"] = self.data.get("N_ENERGY", 8)
        numerics_data["npitch"] = self.data.get("N_XI", 16)

        numerics_data["nonlinear"] = self.is_nonlinear()

        return Numerics(numerics_data)

    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        template_file: Optional[PathLike] = None,
        **kwargs,
    ):
        """
        Set self.data using LocalGeometry, LocalSpecies, and Numerics.
        These may be obtained via another GKInput file, or from Equilibrium/Kinetics
        objects.
        """
        # If self.data is not already populated, fill in defaults from a given
        # template file. If this is not provided by the user, fall back to the
        # default.
        if self.data is None:
            if template_file is None:
                template_file = gk_templates["CGYRO"]
            self.read(template_file)

        # Geometry data
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} not "
                "implemented for CGYRO"
            )

        # Ensure Miller settings in input file
        self.data["EQUILIBRIUM_MODEL"] = 2

        # Assign Miller values to input file
        for key, val in self.pyro_cgyro_miller.items():
            self.data[val] = local_geometry[key]

        self.data["S_DELTA"] = local_geometry.s_delta * np.sqrt(
            1 - local_geometry.delta**2
        )

        # Kinetic data
        self.data["N_SPECIES"] = local_species.nspec

        for i_sp, name in enumerate(local_species.names):
            pyro_cgyro_species = self.get_pyro_cgyro_species(i_sp + 1)

            for pyro_key, cgyro_key in pyro_cgyro_species.items():
                self.data[cgyro_key] = local_species[name][pyro_key]

        # FIXME if species aren't defined, won't this fail?
        self.data["NU_EE"] = local_species.electron.nu

        # Calculate beta and beta_prime_scale. If B0 is not defined, they take the
        # following default values.
        beta = 0.0
        beta_prime_scale = 1.0
        if local_geometry.B0 is not None:
            # If local species are defined...
            if local_species.nref is not None:

                pref = local_species.nref * local_species.tref * electron_charge

                pe = pref * local_species.electron.dens * local_species.electron.temp

                bref = local_geometry.B0 * local_geometry.bunit_over_b0
                beta = pe * 8 * pi * 1e-7 / bref**2

                # Find BETA_STAR_SCALE from beta and p_prime
                beta_prime_scale = -local_geometry.beta_prime / (
                    local_species.a_lp * beta * local_geometry.bunit_over_b0**2
                )

            # Calculate beta from existing value from input
            else:
                beta = 1.0 / (local_geometry.B0 * local_geometry.bunit_over_b0) ** 2

                # FIXME if no species are defined, how do we get a_lp?
                beta_prime_scale = -local_geometry.beta_prime / (
                    local_species.a_lp * beta * local_geometry.bunit_over_b0**2
                )

        self.data["BETAE_UNIT"] = beta
        self.data["BETA_STAR_SCALE"] = beta_prime_scale

        # Numerics
        if numerics.bpar and not numerics.apar:
            raise ValueError("Can't have bpar without apar in CGYRO")

        self.data["N_FIELD"] = 1 + int(numerics.bpar) + int(numerics.apar)

        # Set time stepping
        self.data["DELTA_T"] = numerics.delta_time
        self.data["MAX_TIME"] = numerics.max_time

        if numerics.nonlinear:
            self.data["NONLINEAR_FLAG"] = 1
            self.data["N_RADIAL"] = numerics.nkx
            self.data["BOX_SIZE"] = int(
                (numerics.ky * 2 * pi * local_geometry.shat / numerics.kx) + 0.1
            )
        else:
            self.data["NONLINEAR_FLAG"] = 0
            self.data["N_RADIAL"] = numerics.nperiod * 2
            self.data["BOX_SIZE"] = 1

        self.data["KY"] = numerics.ky
        self.data["N_TOROIDAL"] = numerics.nky

        self.data["N_THETA"] = numerics.ntheta
        self.data["THETA_PLOT"] = numerics.ntheta
        self.data["PX0"] = numerics.theta0 / (2 * pi)

        self.data["N_ENERGY"] = numerics.nenergy
        self.data["N_XI"] = numerics.npitch

        self.data["FIELD_PRINT_FLAG"] = 1
        self.data["MOMENT_PRINT_FLAG"] = 1
