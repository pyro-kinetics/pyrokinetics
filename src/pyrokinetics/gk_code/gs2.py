from __future__ import annotations

import warnings
from copy import copy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import f90nml
import numpy as np
import pint
from cleverdict import CleverDict

from ..constants import pi
from ..file_utils import FileReader
from ..local_geometry import LocalGeometry, LocalGeometryMiller, default_miller_inputs
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict
from ..numerics import Numerics
from ..templates import gk_templates
from ..typing import PathLike
from .gk_input import GKInput
from .gk_output import (
    Coords,
    Eigenfunctions,
    Eigenvalues,
    Fields,
    Fluxes,
    GKOutput,
    Moments,
)

if TYPE_CHECKING:
    import xarray as xr


class GKInputGS2(GKInput, FileReader, file_type="GS2", reads=GKInput):
    """
    Class that can read GS2 input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GS2"
    default_file_name = "input.in"
    norm_convention = "gs2"
    _convention_dict = {}

    pyro_gs2_miller = {
        "rho": ["theta_grid_parameters", "rhoc"],
        "Rmaj": ["theta_grid_parameters", "rmaj"],
        "q": ["theta_grid_parameters", "qinp"],
        "kappa": ["theta_grid_parameters", "akappa"],
        "shat": ["theta_grid_eik_knobs", "s_hat_input"],
        "shift": ["theta_grid_parameters", "shift"],
        "beta_prime": ["theta_grid_eik_knobs", "beta_prime_input"],
    }

    pyro_gs2_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "q": 1.5,
        "kappa": 1.0,
        "shat": 0.0,
        "shift": 0.0,
        "beta_prime": 0.0,
    }

    pyro_gs2_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "nu": "vnewk",
        "inverse_lt": "tprim",
        "inverse_ln": "fprim",
    }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GS2 input file into a dictionary
        """
        result = super().read_from_file(filename)
        if self.is_nonlinear() and self.data["knobs"].get("wstar_units", False):
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
        if self.is_nonlinear() and self.data["knobs"].get("wstar_units", False):
            raise RuntimeError(
                "GKInputGS2: Cannot be nonlinear and set knobs.wstar_units"
            )
        return result

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads GS2 input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid gs2 input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a GS2 input file,
        # but they are needed by Pyrokinetics
        expected_keys = [
            "knobs",
            "theta_grid_knobs",
            "theta_grid_eik_knobs",
            "theta_grid_parameters",
            "species_knobs",
            "kt_grids_knobs",
        ]
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm: Normalisation = None,
        code_normalisation: str = None,
    ):

        if local_norm is None:
            local_norm = Normalisation("write")

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

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

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_geometry")
            convention = getattr(norms, self.norm_convention)

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

        local_geometry = self.get_local_geometry_miller()

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Basic Miller object from GS2 file
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

        for (pyro_key, (gs2_param, gs2_key)), gs2_default in zip(
            self.pyro_gs2_miller.items(), self.pyro_gs2_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data[gs2_param].get(gs2_key, gs2_default)

        rho = miller_data["rho"]
        kappa = miller_data["kappa"]
        miller_data["delta"] = np.sin(
            self.data["theta_grid_parameters"].get("tri", 0.0)
        )
        miller_data["s_kappa"] = (
            self.data["theta_grid_parameters"].get("akappri", 0.0) * rho / kappa
        )
        miller_data["s_delta"] = (
            self.data["theta_grid_parameters"].get("tripri", 0.0) * rho
        )

        beta = self._get_beta()

        # Assume pref*8pi*1e-7 = 1.0
        miller_data["B0"] = np.sqrt(1.0 / beta) if beta != 0.0 else None

        miller_data["ip_ccw"] = 1
        miller_data["bt_ccw"] = 1

        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from GS2 file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")
            convention = getattr(norms, self.norm_convention)

        # Load each species into a dictionary
        for i_sp in range(self.data["species_knobs"]["nspec"]):
            species_data = CleverDict()

            gs2_key = f"species_parameters_{i_sp + 1}"

            gs2_data = self.data[gs2_key]

            for pyro_key, gs2_key in self.pyro_gs2_species.items():
                species_data[pyro_key] = gs2_data[gs2_key]

            species_data.omega0 = self.data["dist_fn_knobs"].get("mach", 0.0)

            # Without PVG term in GS2, need to force to 0
            species_data.domega_drho = (
                self.data["dist_fn_knobs"].get("g_exb", 0.0)
                * self.data["dist_fn_knobs"].get("omprimfac", 1.0)
                * self.data["theta_grid_parameters"]["qinp"]
                / self.data["theta_grid_parameters"]["rhoc"]
            )

            if species_data.z == -1:
                name = "electron"
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # normalisations
            species_data.dens *= convention.nref
            species_data.mass *= convention.mref
            species_data.temp *= convention.tref
            species_data.nu *= convention.vref / convention.lref
            species_data.z *= convention.qref
            species_data.inverse_lt *= convention.lref**-1
            species_data.inverse_ln *= convention.lref**-1
            species_data.omega0 *= convention.vref / convention.lref
            species_data.domega_drho *= convention.vref / convention.lref**2

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.normalise()

        if "zeff" in self.data["knobs"]:
            local_species.zeff = self.data["knobs"]["zeff"] * convention.qref
        elif "parameters" in self.data.keys() and "zeff" in self.data["parameters"]:
            local_species.zeff = self.data["parameters"]["zeff"] * convention.qref
        else:
            local_species.zeff = 1.0 * convention.qref

        return local_species

    def _read_single_grid(self):
        ky = self.data["kt_grids_single_parameters"]["aky"]
        shat = self.data["theta_grid_eik_knobs"]["s_hat_input"]
        theta0 = self.data["kt_grids_single_parameters"].get("theta0", 0.0)

        return {
            "nky": 1,
            "nkx": 1,
            "ky": ky,
            "kx": self.data["kt_grids_single_parameters"].get(
                "akx", ky * shat * theta0
            ),
            "theta0": theta0,
        }

    def _read_range_grid(self):
        range_options = self.data["kt_grids_range_parameters"]
        nky = range_options.get("naky", 1)

        ky_min = range_options.get("aky_min", 0.0)
        ky_max = range_options.get("aky_max", 0.0)

        spacing_option = range_options.get("kyspacing_option", "linear")
        if spacing_option == "default":
            spacing_option = "linear"

        ky_space = np.linspace if spacing_option == "linear" else np.logspace

        ky = ky_space(ky_min, ky_max, nky)

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
                grid_data["ky"] = -box["y0"]
            else:
                grid_data["ky"] = 1 / box["y0"]
        else:
            raise RuntimeError(f"Min ky details not found in {keys}")

        if "nx" in keys:
            grid_data["nkx"] = int(2 * (box["nx"] - 1) / 3 + 1)
        elif "ntheta0" in keys():
            grid_data["nkx"] = int(2 * (box["ntheta0"] - 1) / 3 + 1)
        else:
            raise RuntimeError("kx grid details not found in {keys}")

        shat_params = self.pyro_gs2_miller["shat"]
        shat = self.data[shat_params[0]][shat_params[1]]
        if abs(shat) > 1e-6:
            jtwist_default = max(int(2 * pi * shat + 0.5), 1)
            jtwist = box.get("jtwist", jtwist_default)
            grid_data["kx"] = grid_data["ky"] * shat * 2 * pi / jtwist
        else:
            grid_data["kx"] = 2 * pi / (box["x0"])

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
            reader = GRID_READERS[grid_option]
        except KeyError:
            valid_options = ", ".join(f"'{option}'" for option in GRID_READERS)
            raise ValueError(
                f"Unknown GS2 'kt_grids_knobs::grid_option', '{grid_option}'. Expected one of {valid_options}"
            )

        return reader()

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = self.data["knobs"].get("fphi", 0.0) > 0.0
        numerics_data["apar"] = self.data["knobs"].get("fapar", 0.0) > 0.0
        numerics_data["bpar"] = self.data["knobs"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        delta_time = self.data["knobs"].get("delt", 0.005)
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
        numerics_data["npitch"] = self.data["le_grids_knobs"].get("ngauss", 5) * 2

        numerics_data["beta"] = self._get_beta()
        numerics_data["gamma_exb"] = self.data["dist_fn_knobs"].get(
            "g_exb", 0.0
        ) * self.data["dist_fn_knobs"].get("g_exbfac", 1.0)

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in reference values from input file

        """
        if "normalisations_knobs" not in self.data.keys():
            return {}

        norms = {}

        norms["tref_electron"] = (
            self.data["normalisations_knobs"]["tref"] * local_norm.units.eV
        )
        norms["nref_electron"] = (
            self.data["normalisations_knobs"]["nref"] * local_norm.units.meter**-3
        )
        norms["bref_B0"] = (
            self.data["normalisations_knobs"]["bref"] * local_norm.units.tesla
        )
        norms["lref_minor_radius"] = (
            self.data["normalisations_knobs"]["aref"] * local_norm.units.meter
        )

        return norms

    def _detect_normalisation(self):
        """
        Determines the necessary inputs and passes information to the base method _set_up_normalisation.
        The following values are needed

        default_references: dict
            Dictionary containing default reference values for the
        gk_code: str
            GK code
        electron_density: float
            Electron density from GK input
        electron_temperature: float
            Electron density from GK input
        e_mass: float
            Electron mass from GK input
        electron_index: int
            Index of electron in list of data
        found_electron: bool
            Flag on whether electron was found
        densities: ArrayLike
            List of species densities
        temperatures: ArrayLike
            List of species temperature
        reference_density_index: ArrayLike
            List of indices where the species has a density of 1.0
        reference_temperature_index: ArrayLike
            List of indices where the species has a temperature of 1.0
        major_radius: float
            Normalised major radius from GK input
        rgeo_rmaj: float
            Ratio of Geometric and flux surface major radius
        minor_radius: float
            Normalised minor radius from GK input
        """

        default_references = {
            "nref_species": "electron",
            "tref_species": "electron",
            "mref_species": "deuterium",
            "bref": "B0",
            "lref": "minor_radius",
            "ne": 1.0,
            "te": 1.0,
            "rgeo_rmaj": 1.0,
            "vref": "most_probable",
            "rhoref": "gs2",
        }

        reference_density_index = []
        reference_temperature_index = []

        densities = []
        temperatures = []
        masses = []

        found_electron = False
        e_mass = None
        electron_temperature = None
        electron_density = None
        electron_index = None

        # Load each species into a dictionary
        for i_sp in range(self.data["species_knobs"]["nspec"]):
            species_key = f"species_parameters_{i_sp + 1}"

            dens = self.data[species_key]["dens"]
            temp = self.data[species_key]["temp"]
            mass = self.data[species_key]["mass"]

            # Find all reference values
            if self.data[species_key]["z"] == -1:
                electron_density = dens
                electron_temperature = temp
                e_mass = mass
                electron_index = len(densities)
                found_electron = True

            if np.isclose(dens, 1.0):
                reference_density_index.append(len(densities))
            if np.isclose(temp, 1.0):
                reference_temperature_index.append(len(temperatures))

            densities.append(dens)
            temperatures.append(temp)
            masses.append(mass)

        rgeo_rmaj = (
            self.data["theta_grid_parameters"]["r_geo"]
            / self.data["theta_grid_parameters"]["rmaj"]
        )
        major_radius = self.data["theta_grid_parameters"]["rmaj"]

        if self.data["theta_grid_eik_knobs"]["irho"] == 2:
            minor_radius = 1.0
        else:
            minor_radius = None

        super()._set_up_normalisation(
            default_references=default_references,
            gk_code=self.code_name.lower(),
            electron_density=electron_density,
            electron_temperature=electron_temperature,
            e_mass=e_mass,
            electron_index=electron_index,
            found_electron=found_electron,
            densities=densities,
            temperatures=temperatures,
            reference_density_index=reference_density_index,
            reference_temperature_index=reference_temperature_index,
            major_radius=major_radius,
            rgeo_rmaj=rgeo_rmaj,
            minor_radius=minor_radius,
        )

    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        local_norm: Normalisation = None,
        template_file: Optional[PathLike] = None,
        code_normalisation: Optional[str] = None,
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
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

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

            if species_key not in self.data:
                self.data[species_key] = copy(self.data["species_parameters_1"])
                self.data[f"dist_fn_species_knobs_{iSp + 1}"] = self.data[
                    f"dist_fn_species_knobs_{iSp}"
                ]

            if name == "electron":
                self.data[species_key]["type"] = "electron"
            else:
                self.data[species_key]["type"] = "ion"

            for key, val in self.pyro_gs2_species.items():
                self.data[species_key][val] = local_species[name][key]

        self.data["dist_fn_knobs"]["g_exb"] = numerics.gamma_exb
        self.data["dist_fn_knobs"]["g_exbfac"] = 1.0
        if numerics.gamma_exb == 0.0:
            self.data["dist_fn_knobs"]["omprimfac"] = 1.0
        else:
            self.data["dist_fn_knobs"]["omprimfac"] = (
                local_species.electron.domega_drho
                * local_geometry.rho
                / local_geometry.q
                / numerics.gamma_exb
            )

        self.data["dist_fn_knobs"]["mach"] = local_species.electron.omega0
        self.data["knobs"]["zeff"] = local_species.zeff

        beta_ref = convention.beta if local_norm else 0.0
        self.data["knobs"]["beta"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        # Set numerics bits
        # Set no. of fields
        self.data["knobs"]["fphi"] = 1.0 if numerics.phi else 0.0
        self.data["knobs"]["fapar"] = 1.0 if numerics.apar else 0.0
        self.data["knobs"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        # Set time stepping
        self.data["knobs"]["delt"] = numerics.delta_time
        self.data["knobs"]["nstep"] = int(numerics.max_time / numerics.delta_time)

        if numerics.nky == 1:
            self.data["kt_grids_knobs"]["grid_option"] = "single"

            if "kt_grids_single_parameters" not in self.data.keys():
                self.data["kt_grids_single_parameters"] = {}

            self.data["kt_grids_single_parameters"]["aky"] = numerics.ky
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

            self.data["kt_grids_box_parameters"]["y0"] = -numerics.ky

            # Currently forces NL sims to have nperiod = 1
            self.data["theta_grid_parameters"]["nperiod"] = 1

            shat = local_geometry.shat
            if abs(shat) < 1e-6:
                self.data["kt_grids_box_parameters"]["x0"] = 2 * pi / numerics.kx
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

        if not local_norm:
            return

        try:
            (1 * convention.tref).to("keV")
            si_units = True
        except pint.errors.DimensionalityError:
            si_units = False

        if si_units:
            if "normalisations_knobs" not in self.data.keys():
                self.data["normalisations_knobs"] = f90nml.Namelist()

            self.data["normalisations_knobs"]["tref"] = (1 * convention.tref).to("eV")
            self.data["normalisations_knobs"]["nref"] = (1 * convention.nref).to(
                "meter**-3"
            )
            self.data["normalisations_knobs"]["mref"] = (1 * convention.mref).to(
                "atomic_mass_constant"
            )
            self.data["normalisations_knobs"]["bref"] = (1 * convention.bref).to(
                "tesla"
            )
            self.data["normalisations_knobs"]["aref"] = (1 * convention.lref).to(
                "meter"
            )
            self.data["normalisations_knobs"]["vref"] = (1 * convention.vref).to(
                "meter/second"
            )
            self.data["normalisations_knobs"]["qref"] = 1 * convention.qref
            self.data["normalisations_knobs"]["rhoref"] = (1 * convention.rhoref).to(
                "meter"
            )

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

    def get_ne_te_normalisation(self):
        found_electron = False
        # Load each species into a dictionary
        for i_sp in range(self.data["species_knobs"]["nspec"]):
            gs2_key = f"species_parameters_{i_sp + 1}"
            if (
                self.data[gs2_key]["z"] == -1
                and self.data[gs2_key]["type"] == "electron"
            ):
                ne = self.data[gs2_key]["dens"]
                Te = self.data[gs2_key]["temp"]
                found_electron = True
                break

        if not found_electron:
            raise TypeError(
                "Pyro currently only supports electron species with charge = -1"
            )

        return ne, Te

    def _get_beta(self):
        """
        Small helper to wrap up logic required to get beta from the input
        consistent with logic across versions of GS2.
        """
        has_parameters = "parameters" in self.data.keys()
        beta_default = 0.0
        if has_parameters:
            beta_default = self.data["parameters"].get("beta", 0.0)
        return self.data["knobs"].get("beta", beta_default)


class GKOutputReaderGS2(FileReader, file_type="GS2", reads=GKOutput):
    def read_from_file(
        self,
        filename: PathLike,
        norm: Normalisation,
        output_convention: str = "pyrokinetics",
        downsize: int = 1,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
    ) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        coords = self._get_coords(raw_data, gk_input, downsize)
        fields = self._get_fields(raw_data) if load_fields else None
        fluxes = self._get_fluxes(raw_data, gk_input, coords) if load_fluxes else None
        moments = (
            self._get_moments(raw_data, gk_input, coords) if load_moments else None
        )

        eigenvalues = None
        eigenfunctions = None
        normalise_flux_moment = True
        if not fields and coords["linear"]:
            eigenvalues = self._get_eigenvalues(raw_data, coords["time_divisor"])
            eigenfunctions = self._get_eigenfunctions(raw_data, coords)

            sum_fields = 0
            for field in coords["field"]:
                sum_fields += raw_data[f"{field}2"].data
            fluxes = (
                {k: v / sum_fields for k, v in fluxes.items()} if load_fluxes else None
            )
            moments = (
                {k: v / sum_fields for k, v in moments.items()}
                if load_moments
                else None
            )
            normalise_flux_moment = False

        # Assign units and return GKOutput
        convention = getattr(norm, gk_input.norm_convention)
        norm.default_convention = output_convention.lower()

        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = ("field", "species", "ky", "time")
        moment_dims = ("field", "species", "ky", "time")
        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["theta"],
                pitch=coords["pitch"],
                energy=coords["energy"],
                species=coords["species"],
                field=coords["field"],
            ).with_units(convention),
            norm=norm,
            fields=(
                Fields(**fields, dims=field_dims).with_units(convention)
                if fields
                else None
            ),
            fluxes=(
                Fluxes(**fluxes, dims=flux_dims).with_units(convention)
                if fluxes
                else None
            ),
            moments=(
                Moments(**moments, dims=moment_dims).with_units(convention)
                if moments
                else None
            ),
            eigenvalues=(
                Eigenvalues(**eigenvalues).with_units(convention)
                if eigenvalues
                else None
            ),
            eigenfunctions=(
                None
                if eigenfunctions is None
                else Eigenfunctions(eigenfunctions, dims=("field", "theta", "kx", "ky"))
            ),
            linear=coords["linear"],
            gk_code="GS2",
            input_file=input_str,
            normalise_flux_moment=normalise_flux_moment,
            output_convention=output_convention,
        )

    def verify_file_type(self, filename: PathLike):
        import xarray as xr

        try:
            warnings.filterwarnings("error")
            data = xr.open_dataset(filename)
        except RuntimeWarning:
            warnings.resetwarnings()
            raise RuntimeError("Error occurred reading GS2 output file")
        warnings.resetwarnings()

        if "software_name" in data.attrs:
            if data.attrs["software_name"] != "GS2":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'software_name' for a GS2 file"
                )
        elif "code_info" in data.data_vars:
            if data["code_info"].long_name != "GS2":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'code_info' for a GS2 file"
                )
        elif "gs2_help" in data.attrs.keys():
            pass
        else:
            raise RuntimeError(f"file '{filename}' missing expected GS2 attributes")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Gets path by removing ".in" and replacing it with ".out.nc"
        """
        filename = Path(filename)
        return filename.parent / (filename.stem + ".out.nc")

    @staticmethod
    def _get_raw_data(filename: PathLike) -> Tuple[xr.Dataset, GKInputGS2, str]:
        import xarray as xr

        raw_data = xr.open_dataset(filename)
        # Read input file from netcdf, store as GKInputGS2
        input_file = raw_data["input_file"]
        if input_file.shape == ():
            # New diagnostics, input file stored as bytes
            # - Stored within numpy 0D array, use [()] syntax to extract
            # - Convert bytes to str by decoding
            # - \n is represented as character literals '\' 'n'. Replace with '\n'.
            input_str = input_file.data[()].decode("utf-8").replace(r"\n", "\n")
        else:
            # Old diagnostics (and eventually the single merged diagnostics)
            # input file stored as array of bytes
            if isinstance(input_file.data[0], np.ndarray):
                input_str = "\n".join(
                    ("".join(np.char.decode(line)).strip() for line in input_file.data)
                )
            else:
                input_str = "\n".join(
                    (line.decode("utf-8") for line in input_file.data)
                )
        gk_input = GKInputGS2()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: xr.Dataset, gk_input: GKInputGS2, downsize: int
    ) -> Dict[str, Any]:
        # ky coords
        ky = raw_data["ky"].data

        # time coords
        time_divisor = 1
        try:
            if gk_input.data["knobs"]["wstar_units"]:
                time_divisor = ky[0] / 2
        except KeyError:
            pass

        time = raw_data["t"].data / time_divisor

        # kx coords
        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(raw_data["kx"].data)

        # theta coords
        theta = raw_data["theta"].data

        # energy coords
        try:
            energy = raw_data["egrid"].data
        except KeyError:
            energy = raw_data["energy"].data

        # pitch coords
        pitch = raw_data["lambda"].data

        # moment coords
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        # field coords
        # If fphi/fapar/fbpar not in 'knobs', or they equal zero, skip the field
        field_vals = {}
        for field, default in zip(["phi", "apar", "bpar"], [1.0, 0.0, -1.0]):
            try:
                field_vals[field] = gk_input.data["knobs"][f"f{field}"]
            except KeyError:
                field_vals[field] = default
        # By default, fbpar = -1, which tells gs2 to try reading faperp instead.
        # faperp is deprecated, but is treated as a synonym for fbpar
        # It has a default value of 0.0
        if field_vals["bpar"] == -1:
            try:
                field_vals["bpar"] = gk_input.data["knobs"]["faperp"]
            except KeyError:
                field_vals["bpar"] = 0.0
        fields = [field for field, val in field_vals.items() if val > 0]

        # species coords
        # TODO is there some way to get this info without looking at the input data?
        species = []
        ion_num = 0
        for idx in range(gk_input.data["species_knobs"]["nspec"]):
            if gk_input.data[f"species_parameters_{idx + 1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "linear": gk_input.is_linear(),
            "time_divisor": time_divisor,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "downsize": downsize,
        }

    @staticmethod
    def _get_fields(raw_data: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        For GS2 to print fields, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also instruct GS2 to print each field
        individually in the gs2_diagnostics_knobs using:
        - write_phi_over_time = .true.
        - write_apar_over_time = .true.
        - write_bpar_over_time = .true.
        - write_fields = .true.
        """
        field_names = ("phi", "apar", "bpar")
        results = {}

        # Loop through all fields and add field if it exists
        for field_name in field_names:
            key = f"{field_name}_t"
            if key not in raw_data:
                continue

            # raw_field has coords (t,ky,kx,theta,real/imag).
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field = raw_data[key].transpose("ri", "theta", "kx", "ky", "t").data
            field = field[0, ...] + 1j * field[1, ...]

            # Adjust fields to account for differences in defintions/normalisations
            if field_name == "apar":
                field *= 0.5

            if field_name == "bpar":
                bmag = raw_data["bmag"].data[:, np.newaxis, np.newaxis, np.newaxis]
                field *= bmag

            # Shift kx=0 to middle of axis
            field = np.fft.fftshift(field, axes=1)
            results[field_name] = field

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputGS2,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def _get_fluxes(
        raw_data: xr.Dataset,
        gk_input: GKInputGS2,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        For GS2 to print fluxes, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also set the following in
        gs2_diagnostics_knobs:
        - write_fluxes = .true. (default if nonlinear)
        - write_fluxes_by_mode = .true. (default if nonlinear)
        """
        # field names change from ["phi", "apar", "bpar"] to ["es", "apar", "bpar"]
        # Take whichever fields are present in data, relabelling "phi" to "es"
        fields = {"phi": "es", "apar": "apar", "bpar": "bpar"}
        fluxes_dict = {"particle": "part", "heat": "heat", "momentum": "mom"}

        # Get species names from input file
        species = []
        ion_num = 0
        for idx in range(gk_input.data["species_knobs"]["nspec"]):
            if gk_input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        results = {}

        coord_names = ["flux", "field", "species", "ky", "time"]
        fluxes = np.zeros([len(coords[name]) for name in coord_names])
        fields = {
            field: value for field, value in fields.items() if field in coords["field"]
        }

        for (ifield, (field, gs2_field)), (iflux, gs2_flux) in product(
            enumerate(fields.items()), enumerate(fluxes_dict.values())
        ):
            flux_key = f"{gs2_field}_{gs2_flux}_flux"
            # old diagnostics
            by_k_key = f"{gs2_field}_{gs2_flux}_by_k"
            # new diagnostics
            by_mode_key = f"{gs2_field}_{gs2_flux}_flux_by_mode"

            if by_k_key in raw_data.data_vars or by_mode_key in raw_data.data_vars:
                key = by_mode_key if by_mode_key in raw_data.data_vars else by_k_key
                flux = raw_data[key].transpose("species", "kx", "ky", "t")
                # Sum over kx
                flux = flux.sum(dim="kx")
                # Divide non-zonal components by 2 due to reality condition
                flux[:, 1:, :] *= 0.5
            elif flux_key in raw_data.data_vars:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                flux = raw_data[flux_key]
                flux = flux.expand_dims("ky").transpose("species", "ky", "t")
            else:
                continue

            fluxes[iflux, ifield, ...] = flux

        for iflux, flux in enumerate(coords["flux"]):
            if not np.all(fluxes[iflux, ...] == 0):
                results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: xr.Dataset, time_divisor: float
    ) -> Dict[str, np.ndarray]:
        # should only be called if no field data were found
        if "time" in raw_data.dims:
            time_dim = "time"
        elif "t" in raw_data.dims:
            time_dim = "t"
        mode_frequency = raw_data.omega_average.isel(ri=0).transpose(
            "kx", "ky", time_dim
        )
        growth_rate = raw_data.omega_average.isel(ri=1).transpose("kx", "ky", time_dim)
        return {
            "mode_frequency": mode_frequency.data / time_divisor,
            "growth_rate": growth_rate.data / time_divisor,
        }

    @staticmethod
    def _get_eigenfunctions(
        raw_data: xr.Dataset,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:

        raw_eig_data = [raw_data.get(f, None) for f in coords["field"]]

        ntheta = len(coords["theta"])
        nkx = len(coords["kx"])
        nky = len(coords["ky"])

        coord_names = ["field", "theta", "kx", "ky"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                eigenfunction = raw_eigenfunction.transpose("ri", "theta", "kx", "ky")

                eigenfunctions[ifield, ...] = (
                    eigenfunction[0, ...] + 1j * eigenfunction[1, ...]
                )

        square_fields = np.sum(np.abs(eigenfunctions) ** 2, axis=0)
        field_amplitude = np.sqrt(
            np.trapz(square_fields, coords["theta"], axis=0) / (2 * np.pi)
        )

        first_field = eigenfunctions[0, ...]
        theta_star = np.argmax(abs(first_field), axis=0)
        field_theta_star = first_field[theta_star, 0, 0]
        phase = np.abs(field_theta_star) / field_theta_star

        result = eigenfunctions * phase / field_amplitude

        return result
