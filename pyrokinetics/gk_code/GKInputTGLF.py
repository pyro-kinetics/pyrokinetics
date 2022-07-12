import numpy as np
from pathlib import Path
from cleverdict import CleverDict
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


class GKInputTGLF(GKInput):
    """Reader for TGLF input files"""

    code_name = "TGLF"
    default_file_name = "input.TGLF"
    tglf_max_ntheta = 32

    pyro_tglf_miller = {
        "rho": "rmin_loc",
        "Rmaj": "rmaj_loc",
        "q": "q_loc",
        "kappa": "kappa_loc",
        "s_kappa": "s_kappa_loc",
        "delta": "delta_loc",
        "shift": "drmajdx_loc",
    }

    @staticmethod
    def pyro_TGLF_species(iSp=1):
        return {
            "mass": f"mass_{iSp}",
            "z": f"zs_{iSp}",
            "dens": f"as_{iSp}",
            "temp": f"taus_{iSp}",
            "a_lt": f"rlts_{iSp}",
            "a_ln": f"rlns_{iSp}",
        }

    def read(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads TGLF input file into a dictionary
        """
        with open(filename) as f:
            contents = f.read()

        return self.read_str(contents)

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads TGLF input file given as string
        """
        # TGLF input files are _almost_ Fortran namelists, so if we
        # change the comments to use '!' instead of '#', and wrap it
        # in a namelist syntax, we can just use the base `read_str`
        as_namelist = f"&nml\n{input_string.replace('#', '!')}\n/"

        # We need to strip off our fake namelist wrapper when we store
        # it internally
        self.data = super().read_str(as_namelist)["nml"]
        return self.data

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid TGLF input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """

        expected_keys = ["rmin_loc", "rmaj_loc", "nky"]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as TGLF file")

    def write(self, filename: PathLike, float_format: str = ""):
        """
        Write input file for TGLF
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "w+") as new_TGLF_input:
            for key, value in self.data.items():
                if isinstance(value, float):
                    value_str = f"{value:{float_format}}"
                elif isinstance(value, bool):
                    value_str = "T" if value else "F"
                else:
                    value_str = str(value)

                new_TGLF_input.write(f"{key.upper()} = {value_str}\n")

    def is_nonlinear(self) -> bool:
        return self.data.get("use_transport_model", 1) == 1

    def add_flags(self, flags) -> None:
        """
        Add extra flags to TGLF input file
        """
        for key, value in flags.items():
            self.data[key] = value

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """

        tglf_eq_flag = self.data["geometry_flag"]
        tglf_eq_mapping = ["SAlpha", "Miller", "Fourier", "ELITE"]
        tglf_eq = tglf_eq_mapping[tglf_eq_flag]

        if tglf_eq not in ["Miller"]:
            raise NotImplementedError(
                f"TGLF equilibrium option '{tglf_eq_flag}' ('{tglf_eq}') not implemented"
            )

        return self.get_local_geometry_miller()

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from TGLF file
        """

        miller_data = default_miller_inputs()

        for pyro_key, tglf_key in self.pyro_tglf_miller.items():
            miller_data[pyro_key] = self.data[tglf_key]

        miller_data["s_delta"] = self.data["s_delta_loc"] / np.sqrt(
            1 - miller_data["delta"] ** 2
        )
        miller_data["shat"] = (
            self.data["q_prime_loc"] * (miller_data["rho"] / miller_data["q"]) ** 2
        )

        # Must construct using from_gk_data as we cannot determine
        # bunit_over_b0 here. We also need it to set B0 and
        # beta_prime, so we have to make a miller instance first
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        beta = self.data["betae"]
        miller.B0 = 1 / (beta**0.5) / miller.bunit_over_b0 if beta != 0 else None

        # FIXME: This actually needs to be scaled (or overwritten?) by
        # local_species.a_lp and self.data["BETA_STAR_SCALE"]. So we
        # need to get all the species data first?
        miller.beta_prime = (
            self.data["p_prime_loc"]
            * miller_data["rho"]
            / miller_data["q"]
            * miller.bunit_over_b0**2
            * (8 * np.pi)
        )

        return miller

    def get_local_species(self):
        """
        Load LocalSpecies object from TGLF file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(self.data["ns"]):
            pyro_TGLF_species = self.pyro_TGLF_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_TGLF_species.items():
                species_data[p_key] = self.data[c_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
                species_data.nu = self.data["xnue"]
                te = species_data.temp
                ne = species_data.dens
                me = species_data.mass
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Get collision frequency of ion species
        nu_ee = self.data.get("xnue", 0.0)

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

        local_species.normalise()
        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = True
        numerics_data["apar"] = bool(self.data.get("use_bper", False))
        numerics_data["bpar"] = bool(self.data.get("use_bpar", False))

        numerics_data["ky"] = self.data["ky"]

        numerics_data["nky"] = self.data.get("nky", 1)
        numerics_data["theta0"] = self.data.get("kx0_loc", 0.0) * 2 * pi
        numerics_data["ntheta"] = self.data.get("nxgrid", 16)
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
                template_file = gk_templates["TGLF"]
            self.read(template_file)

        # Set Miller Geometry bits
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for TGLF not supported yet"
            )

        # Geometry (Miller)
        self.data["geometry_flag"] = 1
        # Reference B field - Bunit = q/r dpsi/dr
        b_ref = None
        if local_geometry.B0 is not None:
            b_ref = local_geometry.B0 * local_geometry.bunit_over_b0

        # Assign Miller values to input file
        for key, value in self.pyro_tglf_miller.items():
            self.data[value] = local_geometry[key]

        self.data["s_delta_loc"] = local_geometry.s_delta * np.sqrt(
            1 - local_geometry.delta**2
        )
        self.data["q_prime_loc"] = (
            local_geometry.shat * (local_geometry.q / local_geometry.rho) ** 2
        )

        # Set local species bits
        self.data["ns"] = local_species.nspec
        for iSp, name in enumerate(local_species.names):
            tglf_species = self.pyro_TGLF_species(iSp + 1)

            for pyro_key, TGLF_key in tglf_species.items():
                self.data[TGLF_key] = local_species[name][pyro_key]

        self.data["xnue"] = local_species.electron.nu

        beta = 0.0

        # If species are defined calculate beta and beta_prime_scale
        if local_species.nref is not None:
            pref = local_species.nref * local_species.tref * electron_charge
            pe = pref * local_species.electron.dens * local_species.electron.temp
            beta = pe / b_ref**2 * 8 * pi * 1e-7

        elif local_geometry.B0 is not None:
            # Calculate beta from existing value from input
            beta = 1.0 / (local_geometry.B0 * local_geometry.bunit_over_b0) ** 2

        self.data["betae"] = beta

        self.data["p_prime_loc"] = (
            local_geometry.beta_prime
            * local_geometry.q
            / local_geometry.rho
            / local_geometry.bunit_over_b0**2
            / (8 * np.pi)
        )

        # Numerics
        self.data["use_bper"] = numerics.apar
        self.data["use_bpar"] = numerics.bpar

        # Set time stepping
        self.data["use_transport_model"] = numerics.nonlinear

        self.data["ky"] = numerics.ky
        self.data["nky"] = numerics.nky

        self.data["nxgrid"] = min(numerics.ntheta, self.tglf_max_ntheta)
        self.data["kx0_loc"] = numerics.theta0 / (2 * pi)

        if not numerics.nonlinear:
            self.data["write_wavefunction_flag"] = 1
