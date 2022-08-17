import numpy as np
from cleverdict import CleverDict
from copy import copy
from typing import Dict, Any, Optional
from ..typing import PathLike
from ..constants import pi, sqrt2, electron_charge
from ..local_species import LocalSpecies
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    default_miller_inputs,
)
from ..numerics import Numerics
from ..templates import gk_templates
from .GKInput import GKInput


class GKInputGS2(GKInput):
    """
    Class that can read GS2 input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GS2"
    default_file_name = "input.in"

    pyro_gs2_miller = {
        "rho": ["theta_grid_parameters", "rhoc"],
        "Rmaj": ["theta_grid_parameters", "rmaj"],
        "q": ["theta_grid_parameters", "qinp"],
        "kappa": ["theta_grid_parameters", "akappa"],
        "shat": ["theta_grid_eik_knobs", "s_hat_input"],
        "shift": ["theta_grid_parameters", "shift"],
        "beta_prime": ["theta_grid_eik_knobs", "beta_prime_input"],
    }

    pyro_gs2_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "nu": "vnewk",
        "a_lt": "tprim",
        "a_ln": "fprim",
        "a_lv": "uprim",
    }

    def read(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GS2 input file into a dictionary
        """
        result = super().read(filename)
        if self.is_nonlinear() and "wstar_units" in self.data["knobs"]:
            raise RuntimeError(
                "GKInputGS2: Cannot be nonlinear and set knobs.wstar_units"
            )
        return result

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GS2 input file given as string
        Uses default read_str, which assumes input_string is a Fortran90 namelist
        """
        result = super().read_str(input_string)
        if self.is_nonlinear() and "wstar_units" in self.data["knobs"]:
            raise RuntimeError(
                "GKInputGS2: Cannot be nonlinear and set knobs.wstar_units"
            )
        return result

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid gs2 input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a GS2 input file,
        # but they are needed by Pyrokinetics
        expected_keys = [
            "knobs",
            "parameters",
            "theta_grid_knobs",
            "theta_grid_eik_knobs",
            "theta_grid_parameters",
            "species_knobs",
            "kt_grids_knobs",
        ]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as GS2 file")

    def write(self, filename: PathLike, float_format: str = ""):
        super().write(filename, float_format=float_format)

    def is_nonlinear(self) -> bool:
        try:
            is_box = self.data["kt_grids_knobs"]["grid_option"] == "box"
            is_nonlinear = self.data["nonlinear_terms_knobs"]["nonlinear_mode"] == "on"
            return is_box and is_nonlinear
        except KeyError:
            return False

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GS2 input file
        """
        super().add_flags(flags)

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """
        gs2_eq = self.data["theta_grid_knobs"]["equilibrium_option"]

        if gs2_eq not in ["eik", "default"]:
            raise NotImplementedError(
                f"GS2 equilibrium option {gs2_eq} not implemented"
            )

        local_eq = self.data["theta_grid_eik_knobs"].get("local_eq", True)
        if not local_eq:
            raise RuntimeError("GS2 is not using local equilibrium")

        geotype = self.data["theta_grid_parameters"].get("geotype", 0)
        if geotype != 0:
            raise NotImplementedError("GS2 Fourier options are not implemented")

        return self.get_local_geometry_miller()

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GS2 file
        """
        # We require the use of Bishop mode 4, which uses a numerical equilibrium,
        # s_hat_input, and beta_prime_input to determine metric coefficients.
        # We also require 'irho' to be 2, which means rho corresponds to the ratio of
        # the midplane diameter to the Last Closed Flux Surface (LCFS) diameter
        if self.data["theta_grid_eik_knobs"]["bishop"] != 4:
            raise RuntimeError(
                "Pyrokinetics requires GS2 input files to use "
                "theta_grid_eik_knobs.bishop = 4"
            )
        if self.data["theta_grid_eik_knobs"]["irho"] != 2:
            raise RuntimeError(
                "Pyrokinetics requires GS2 input files to use "
                "theta_grid_eik_knobs.bishop = 2"
            )

        miller_data = default_miller_inputs()

        for pyro_key, (gs2_param, gs2_key) in self.pyro_gs2_miller.items():
            miller_data[pyro_key] = self.data[gs2_param][gs2_key]

        rho = miller_data["rho"]
        kappa = miller_data["kappa"]
        miller_data["delta"] = np.sin(self.data["theta_grid_parameters"]["tri"])
        miller_data["s_kappa"] = (
            self.data["theta_grid_parameters"]["akappri"] * rho / kappa
        )
        miller_data["s_delta"] = self.data["theta_grid_parameters"]["tripri"] * rho

        # Get beta and beta_prime normalised to R_major(in case R_geo != R_major)
        r_geo = self.data["theta_grid_parameters"].get("r_geo", miller_data["Rmaj"])

        beta = self.data["parameters"]["beta"] * (miller_data["Rmaj"] / r_geo) ** 2
        miller_data["beta_prime"] *= (miller_data["Rmaj"] / r_geo) ** 2

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Is this assumption general enough? Can't we get pref from local_species?
        # FIXME B0 = None can cause problems when writing
        miller_data["B0"] = np.sqrt(1.0 / beta) if beta != 0.0 else None

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from GS2 file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(self.data["species_knobs"]["nspec"]):

            species_data = CleverDict()

            gs2_key = f"species_parameters_{i_sp + 1}"

            gs2_data = self.data[gs2_key]

            for pyro_key, gs2_key in self.pyro_gs2_species.items():
                species_data[pyro_key] = gs2_data[gs2_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            # Account for sqrt(2) in vth
            species_data.nu = gs2_data["vnewk"] * sqrt2

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.normalise()
        return local_species

    def _read_single_grid(self):

        ky = self.data["kt_grids_single_parameters"]["aky"]
        shat = self.data["theta_grid_eik_knobs"]["s_hat_input"]
        theta0 = self.data["kt_grids_single_parameters"].get("theta0", 0.0)

        return {
            "nky": 1,
            "nkx": 1,
            "ky": ky / sqrt2,
            "kx": self.data["kt_grids_single_parameters"].get("akx", ky * shat * theta0)
            / sqrt2,
            "theta0": theta0,
        }

    def _read_range_grid(self):
        range_options = self.data["kt_grids_range_parameters"]
        nky = range_options.get("naky", 1)

        ky_min = range_options.get("aky_min", "0.0")
        ky_max = range_options.get("aky_max", "0.0")

        spacing_option = range_options.get("kyspacing_option", "linear")
        if spacing_option == "default":
            spacing_option = "linear"

        ky_space = np.linspace if spacing_option == "linear" else np.logspace

        ky = ky_space(ky_min, ky_max, nky) / sqrt2

        return {
            "nky": nky,
            "nkx": 1,
            "ky": ky,
            "kx": np.array([0.0]),
            "theta0": 0.0,
        }

    def _read_box_grid(self):
        box = self.data["kt_grids_box_parameters"]
        keys = box.keys()

        grid_data = {}

        # Set up ky grid
        if "ny" in keys:
            grid_data["nky"] = int((box["ny"] - 1) / 3 + 1)
        elif "n0" in keys:
            grid_data["nky"] = box["n0"]
        elif "nky" in keys:
            grid_data["nky"] = box["naky"]
        else:
            raise RuntimeError(f"ky grid details not found in {keys}")

        if "y0" in keys:
            if box["y0"] < 0.0:
                grid_data["ky"] = -box["y0"] / sqrt2
            else:
                grid_data["ky"] = 1 / box["y0"] / sqrt2
        else:
            raise RuntimeError(f"Min ky details not found in {keys}")

        if "nx" in keys:
            grid_data["nkx"] = int((2 * box["nx"] - 1) / 3 + 1)
        elif "ntheta0" in keys():
            grid_data["nkx"] = int((2 * box["ntheta0"] - 1) / 3 + 1)
        else:
            raise RuntimeError("kx grid details not found in {keys}")

        shat_params = self.pyro_gs2_miller["shat"]
        shat = self.data[shat_params[0]][shat_params[1]]
        if abs(shat) > 1e-6:
            grid_data["kx"] = grid_data["ky"] * shat * 2 * pi / box["jtwist"]
        else:
            grid_data["kx"] = 2 * pi / (box["x0"] * sqrt2)

        return grid_data

    def _read_grid(self):
        """Read the perpendicular wavenumber grid"""

        grid_option = self.data["kt_grids_knobs"].get("grid_option", "single")

        GRID_READERS = {
            "default": self._read_single_grid,
            "single": self._read_single_grid,
            "range": self._read_range_grid,
            "box": self._read_box_grid,
            "nonlinear": self._read_box_grid,
        }

        try:
            return GRID_READERS[grid_option]()
        except KeyError:
            valid_options = ", ".join(f"'{option}'" for option in GRID_READERS)
            raise ValueError(
                f"Unknown GS2 'kt_grids_knobs::grid_option', '{grid_option}'. Expected one of {valid_options}"
            )

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = self.data["knobs"].get("fphi", 0.0) > 0.0
        numerics_data["apar"] = self.data["knobs"].get("fapar", 0.0) > 0.0
        numerics_data["bpar"] = self.data["knobs"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        delta_time = self.data["knobs"].get("delt", 0.005) / sqrt2
        numerics_data["delta_time"] = delta_time
        numerics_data["max_time"] = self.data["knobs"].get("nstep", 50000) * delta_time

        numerics_data["nonlinear"] = self.is_nonlinear()

        numerics_data.update(self._read_grid())

        # Theta grid
        numerics_data["ntheta"] = self.data["theta_grid_parameters"]["ntheta"]
        numerics_data["nperiod"] = self.data["theta_grid_parameters"]["nperiod"]

        # Velocity grid
        try:
            numerics_data["nenergy"] = (
                self.data["le_grids_knobs"]["nesub"]
                + self.data["le_grids_knobs"]["nesuper"]
            )
        except KeyError:
            numerics_data["nenergy"] = self.data["le_grids_knobs"]["negrid"]

        # Currently using number of un-trapped pitch angles
        numerics_data["npitch"] = self.data["le_grids_knobs"]["ngauss"] * 2

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
                template_file = gk_templates["GS2"]
            self.read(template_file)

        # Set Miller Geometry bits
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GS2 not supported yet"
            )

        # Ensure Miller settings
        self.data["theta_grid_knobs"]["equilibrium_option"] = "eik"
        self.data["theta_grid_eik_knobs"]["iflux"] = 0
        self.data["theta_grid_eik_knobs"]["local_eq"] = True
        self.data["theta_grid_eik_knobs"]["bishop"] = 4
        self.data["theta_grid_eik_knobs"]["irho"] = 2
        self.data["theta_grid_parameters"]["geoType"] = 0

        # Assign Miller values to input file
        for key, val in self.pyro_gs2_miller.items():
            self.data[val[0]][val[1]] = local_geometry[key]

        self.data["theta_grid_parameters"]["akappri"] = (
            local_geometry.s_kappa * local_geometry.kappa / local_geometry.rho
        )
        self.data["theta_grid_parameters"]["tri"] = np.arcsin(local_geometry.delta)
        self.data["theta_grid_parameters"]["tripri"] = (
            local_geometry["s_delta"] / local_geometry.rho
        )
        self.data["theta_grid_parameters"]["r_geo"] = local_geometry.Rmaj

        # Set local species bits
        self.data["species_knobs"]["nspec"] = local_species.nspec
        for iSp, name in enumerate(local_species.names):

            # add new outer params for each species
            species_key = f"species_parameters_{iSp + 1}"

            if name == "electron":
                self.data[species_key]["type"] = "electron"
            else:
                try:
                    self.data[species_key]["type"] = "ion"
                except KeyError:
                    self.data[species_key] = copy(self.data["species_parameters_1"])
                    self.data[species_key]["type"] = "ion"

                    self.data[f"dist_fn_species_knobs_{iSp + 1}"] = self.data[
                        f"dist_fn_species_knobs_{iSp}"
                    ]

            for key, val in self.pyro_gs2_species.items():
                self.data[species_key][val] = local_species[name][key]

            # Account for sqrt(2) in vth
            self.data[species_key]["vnewk"] = local_species[name]["nu"] / sqrt2

        # If species are defined calculate beta
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge
            # FIXME local_geometry.B0 may be set to None
            bref = local_geometry.B0

            beta = pref / bref**2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if local_geometry.B0 is not None:
                beta = 1 / local_geometry.B0**2
            else:
                beta = 0.0

        self.data["parameters"]["beta"] = beta

        # Set numerics bits
        # Set no. of fields
        self.data["knobs"]["fphi"] = 1.0 if numerics.phi else 0.0
        self.data["knobs"]["fapar"] = 1.0 if numerics.apar else 0.0
        self.data["knobs"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        # Set time stepping
        self.data["knobs"]["delt"] = numerics.delta_time * sqrt2
        self.data["knobs"]["nstep"] = int(numerics.max_time / numerics.delta_time)

        if numerics.nky == 1:
            self.data["kt_grids_knobs"]["grid_option"] = "single"

            if "kt_grids_single_parameters" not in self.data.keys():
                self.data["kt_grids_single_parameters"] = {}

            self.data["kt_grids_single_parameters"]["aky"] = numerics.ky * sqrt2
            self.data["kt_grids_single_parameters"]["theta0"] = numerics.theta0
            self.data["theta_grid_parameters"]["nperiod"] = numerics.nperiod

        else:
            self.data["kt_grids_knobs"]["grid_option"] = "box"

            if "kt_grids_box_parameters" not in self.data.keys():
                self.data["kt_grids_box_parameters"] = {}

            self.data["kt_grids_box_parameters"]["nx"] = int(
                ((numerics.nkx - 1) * 3 / 2) + 1
            )
            self.data["kt_grids_box_parameters"]["ny"] = int(
                ((numerics.nky - 1) * 3) + 1
            )

            self.data["kt_grids_box_parameters"]["y0"] = -numerics.ky * sqrt2

            # Currently forces NL sims to have nperiod = 1
            self.data["theta_grid_parameters"]["nperiod"] = 1

            shat = local_geometry.shat
            if abs(shat) < 1e-6:
                self.data["kt_grids_box_parameters"]["x0"] = (
                    2 * pi / numerics.kx / sqrt2
                )
            else:
                self.data["kt_grids_box_parameters"]["jtwist"] = int(
                    (numerics.ky * shat * 2 * pi / numerics.kx) + 0.1
                )

        self.data["theta_grid_parameters"]["ntheta"] = numerics.ntheta

        self.data["le_grids_knobs"]["negrid"] = numerics.nenergy
        self.data["le_grids_knobs"]["ngauss"] = numerics.npitch // 2

        if numerics.nonlinear:
            if "nonlinear_terms_knobs" not in self.data.keys():
                self.data["nonlinear_terms_knobs"] = {}

            self.data["nonlinear_terms_knobs"]["nonlinear_mode"] = "on"
        else:
            try:
                self.data["nonlinear_terms_knobs"]["nonlinear_mode"] = "off"
            except KeyError:
                pass
