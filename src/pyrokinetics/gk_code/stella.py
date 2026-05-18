from __future__ import annotations

import warnings
from copy import copy
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import f90nml
import numpy as np
from cleverdict import CleverDict

from ..constants import deuterium_mass, electron_mass, pi
from ..file_utils import FileReader
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    MetricTerms,
    default_miller_inputs,
)
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict, ureg
from ..numerics import Numerics
from ..templates import gk_templates
from ..typing import PathLike
from ..units import PyroContextError
from .gk_input import GKInput
from .gk_output import Coords, Eigenvalues, Fields, Fluxes, GKOutput, Moments


class StellaFormatVersion(Enum):
    """Stella input file format versions."""

    LEGACY = "legacy"  # knobs, parameters, physics_flags
    PRE_V1 = "pre_v1"  # parameters_numerical, parameters_physics
    V1 = "v1"  # restructured: many small namelists


if TYPE_CHECKING:
    import xarray as xr


class GKInputSTELLA(GKInput, FileReader, file_type="STELLA", reads=GKInput):
    """
    Class that can read STELLA input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "STELLA"
    default_file_name = "input.in"
    norm_convention = "stella"
    _format_version = StellaFormatVersion.PRE_V1
    _parameters_physics = "parameters_physics"
    _parameters_params = "parameters_physics"
    _parameters_numerical = "parameters_numerical"
    _legacy_stella = False

    # V1 routing tables: map variable name -> v1 namelist
    _V1_PHYSICS_MAP = {
        "vnew_ref": "dissipation_and_collisions_options",
        "zeff": "dissipation_and_collisions_options",
        "beta": "electromagnetic",
        "include_apar": "electromagnetic",
        "include_bpar": "electromagnetic",
        "include_electromagnetic": "gyrokinetic_terms",
        "adiabatic_option": "adiabatic_electron_response",
        "nine": "adiabatic_electron_response",
        "tite": "adiabatic_electron_response",
        "g_exb": "flow_shear",
        "g_exbfac": "flow_shear",
        "omprimfac": "flow_shear",
        "rhostar": "physics_inputs",
    }

    _V1_NUMERICAL_MAP = {
        "fphi": "scale_gyrokinetic_terms",
        "delt": "time_step",
        "delt_min": "time_step",
        "delt_max": "time_step",
        "nstep": "time_trace_options",
        "tend": "time_trace_options",
        "avail_cpu_time": "time_trace_options",
        "stream_implicit": "numerical_algorithms",
        "mirror_implicit": "numerical_algorithms",
        "mirror_semi_lagrange": "numerical_algorithms",
        "drifts_implicit": "numerical_algorithms",
        "explicit_option": "numerical_algorithms",
        "explicit_algorithm": "numerical_algorithms",
        "zed_upwind": "numerical_upwinding_for_derivatives",
        "vpa_upwind": "numerical_upwinding_for_derivatives",
        "time_upwind": "numerical_upwinding_for_derivatives",
    }

    pyro_stella_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "q": 1.5,
        "kappa": 1.0,
        "shat": 0.0,
        "shift": 0.0,
        "beta_prime": 0.0,
    }

    pyro_stella_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "inverse_lt": "tprim",
        "inverse_ln": "fprim",
    }

    # --- Format version detection and namelist name properties ---

    @property
    def _miller_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "geometry_miller"
        return "millergeo_parameters"

    @property
    def _geo_knobs_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "geometry_options"
        return "geo_knobs"

    @property
    def _species_knobs_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "species_options"
        return "species_knobs"

    @property
    def _zgrid_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "z_grid"
        return "zgrid_parameters"

    @property
    def _vpamu_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "velocity_grids"
        return "vpamu_grids_parameters"

    @property
    def _kt_grids_knobs_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "kxky_grid_option"
        return "kt_grids_knobs"

    @property
    def _kt_grids_range_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "kxky_grid_range"
        return "kt_grids_range_parameters"

    @property
    def _kt_grids_box_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "kxky_grid_box"
        return "kt_grids_box_parameters"

    @property
    def _dissipation_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "dissipation_and_collisions_options"
        return "dissipation"

    @property
    def _diagnostics_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "diagnostics"
        return "stella_diagnostics_knobs"

    @property
    def _diagnostics_fluxes_namelist(self):
        if self._format_version == StellaFormatVersion.V1:
            return "diagnostics_fluxes"
        return "stella_diagnostics_knobs"

    @property
    def pyro_stella_miller(self):
        nl = self._miller_namelist
        return {
            "rho": [nl, "rhoc"],
            "Rmaj": [nl, "rmaj"],
            "q": [nl, "qinp"],
            "kappa": [nl, "kappa"],
            "shat": [nl, "shat"],
            "shift": [nl, "shift"],
            "beta_prime": [nl, "betaprim"],
        }

    def _detect_format_version(self):
        """Detect stella input format version from namelist keys."""
        keys = set(self.data.keys())
        v1_markers = {
            "geometry_options",
            "geometry_miller",
            "gyrokinetic_terms",
            "electromagnetic",
            "kxky_grid_option",
            "z_grid",
            "velocity_grids",
            "time_step",
            "time_trace_options",
            "numerical_algorithms",
            "species_options",
            "parallelisation",
        }
        if v1_markers.intersection(keys):
            return StellaFormatVersion.V1
        legacy_markers = {"knobs", "parameters", "physics_flags"}
        if legacy_markers.intersection(keys):
            return StellaFormatVersion.LEGACY
        return StellaFormatVersion.PRE_V1

    def _set_format_version(self, version: StellaFormatVersion):
        """Set the format version and update namelist name pointers."""
        self._format_version = version
        if version == StellaFormatVersion.LEGACY:
            warnings.warn(
                "The keys 'knobs'/'parameters'/'physics_flags' were found in the input file suggesting this is a "
                "legacy input file, please update this to the latest version to suppress this warning",
                UserWarning,
                stacklevel=3,
            )
            self._legacy_stella = True
            self._parameters_numerical = "knobs"
            self._parameters_params = "parameters"
            self._parameters_physics = "physics_flags"
        elif version == StellaFormatVersion.PRE_V1:
            warnings.warn(
                "Loading a pre-v1 file. V1 is now the default.",
                UserWarning,
                stacklevel=3,
            )
            self._legacy_stella = False
            self._parameters_numerical = "parameters_numerical"
            self._parameters_params = "parameters_physics"
            self._parameters_physics = "parameters_physics"
        else:  # V1
            warnings.warn(
                "Using stella v1 format (default).", UserWarning, stacklevel=3
            )
            self._legacy_stella = False
            self._parameters_numerical = None
            self._parameters_params = None
            self._parameters_physics = None

    def _set_legacy_stella(self, flag: bool):
        """Backward-compatible shim for legacy format detection."""
        if flag:
            self._set_format_version(StellaFormatVersion.LEGACY)
        else:
            self._set_format_version(StellaFormatVersion.PRE_V1)

    def _get_physics_var(self, name, default=None):
        """Get a physics variable from the correct namelist for the current format."""
        if self._format_version == StellaFormatVersion.V1:
            namelist = self._V1_PHYSICS_MAP.get(name)
            if namelist is None:
                raise KeyError(f"Unknown v1 physics variable: {name}")
            return self.data.get(namelist, {}).get(name, default)

        return self.data.get(self._parameters_params, {}).get(name, default)

    def _set_physics_var(self, name, value):
        """Set a physics variable in the correct namelist for the current format."""
        if self._format_version == StellaFormatVersion.V1:
            namelist = self._V1_PHYSICS_MAP.get(name)
            if namelist is None:
                raise KeyError(f"Unknown v1 physics variable: {name}")
            if namelist not in self.data:
                self.data[namelist] = f90nml.Namelist()
            self.data[namelist][name] = value
        else:
            self.data[self._parameters_params][name] = value

    def _get_numerical_var(self, name, default=None):
        """Get a numerical variable from the correct namelist for the current format."""
        if self._format_version == StellaFormatVersion.V1:
            namelist = self._V1_NUMERICAL_MAP.get(name)
            if namelist is None:
                raise KeyError(f"Unknown v1 numerical variable: {name}")
            return self.data.get(namelist, {}).get(name, default)

        return self.data.get(self._parameters_numerical, {}).get(name, default)

    def _set_numerical_var(self, name, value):
        """Set a numerical variable in the correct namelist for the current format."""
        if self._format_version == StellaFormatVersion.V1:
            # Handle the explicit_option -> explicit_algorithm rename
            key = "explicit_algorithm" if name == "explicit_option" else name
            namelist = self._V1_NUMERICAL_MAP.get(name)
            if namelist is None:
                raise KeyError(f"Unknown v1 numerical variable: {name}")
            if namelist not in self.data:
                self.data[namelist] = f90nml.Namelist()
            self.data[namelist][key] = value
        else:
            self.data[self._parameters_numerical][name] = value

    def read_from_file(
        self, filename: PathLike, detect_norm: bool = True
    ) -> Dict[str, Any]:
        """
        Reads STELLA input file into a dictionary
        """
        # Read file but defer normalisation so we can detect format version first
        result = super().read_from_file(filename, detect_norm=False)
        version = self._detect_format_version()

        self._set_format_version(version)

        if detect_norm:
            self._detect_normalisation()
        return result

    def read_str(self, input_string: str, detect_norm: bool = True) -> Dict[str, Any]:
        """
        Reads STELLA input file given as string
        Uses default read_str, which assumes input_string is a Fortran90 namelist
        """
        # Read string but defer normalisation so we can detect format version first
        result = super().read_str(input_string, detect_norm=False)
        version = self._detect_format_version()

        self._set_format_version(version)

        if detect_norm:
            self._detect_normalisation()
        return result

    def read_dict(self, input_dict: dict, detect_norm: bool = True) -> Dict[str, Any]:
        """
        Reads STELLA input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        result = super().read_dict(input_dict, detect_norm=False)
        version = self._detect_format_version()
        self._set_format_version(version)

        if detect_norm:
            self._detect_normalisation()
        return result

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid stella input file, and that it contains sufficient
        info for Pyrokinetics to work with. Accepts legacy, pre-v1, and v1 formats.
        """

        # Try to read the file to check its keys
        try:
            data = f90nml.read(filename)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read '{filename}' as a Fortran namelist file"
            ) from exc

        keys = set(data.keys())

        # V1 format keys
        v1_required = {"geometry_miller", "species_options", "kxky_grid_option"}
        v1_alt_required = {"geometry_options", "species_options", "kxky_grid_option"}
        # Legacy/pre-v1 format keys
        pre_v1_required = {
            "geo_knobs",
            "millergeo_parameters",
            "species_knobs",
            "kt_grids_knobs",
        }

        if (
            v1_required.issubset(keys)
            or v1_alt_required.issubset(keys)
            or pre_v1_required.issubset(keys)
        ):
            return

        raise ValueError(
            f"Unable to verify '{filename}' as a STELLA input file. "
            f"Expected v1 keys {v1_required} or legacy/pre-v1 keys {pre_v1_required}, "
            f"but found: {keys}"
        )

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm=None,
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
            is_box = (
                self.data.get(self._kt_grids_knobs_namelist, {}).get("grid_option")
                == "box"
            )

            if self._format_version == StellaFormatVersion.V1:
                is_nonlinear = self.data.get("gyrokinetic_terms", {}).get(
                    "include_nonlinear", False
                )
            else:
                is_nonlinear = self.data.get(self._parameters_physics, {}).get(
                    "nonlinear", False
                )

            return is_box and is_nonlinear

        except (KeyError, AttributeError, TypeError):
            return False

    def add_flags(self, flags) -> None:
        """
        Add extra flags to STELLA input file
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

        geo_knobs = self.data[self._geo_knobs_namelist]
        if self._format_version == StellaFormatVersion.V1:
            stella_eq = geo_knobs.get("geometry_option", "local")
        else:
            stella_eq = geo_knobs["geo_option"]

        # Both "miller" (old) and "local" (v1) map to Miller geometry
        if stella_eq not in ["miller", "local", "default"]:
            raise NotImplementedError(
                f"stella equilibrium option {stella_eq} not implemented"
            )

        local_geometry = self.get_local_geometry_miller()

        miller_data = self.data[self._miller_namelist]
        local_geometry.B0 = miller_data["rgeo"] / miller_data["rmaj"]
        local_geometry.dpsidr *= local_geometry.B0

        local_geometry.normalise(norms=convention)

        local_geometry.Fpsi = local_geometry.get_f_psi()
        local_geometry.FF_prime = local_geometry.get_f_prime() * local_geometry.Fpsi

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Basic Miller object from stella file
        """
        miller_data = default_miller_inputs()
        ml = self._miller_namelist

        for (pyro_key, (stella_param, stella_key)), stella_default in zip(
            self.pyro_stella_miller.items(), self.pyro_stella_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data[stella_param].get(
                stella_key, stella_default
            )

        rho = miller_data["rho"]
        kappa = miller_data["kappa"]
        miller_data["delta"] = np.sin(self.data[ml].get("tri", 0.0))
        miller_data["s_kappa"] = self.data[ml].get("kapprim", 0.0) * rho / kappa
        miller_data["s_delta"] = self.data[ml].get("triprim", 0.0) * rho

        # convert from stella normalisation to pyrokinetics normalisation of beta_prime
        miller_data["beta_prime"] *= -2.0

        miller_data["ip_ccw"] = 1
        miller_data["bt_ccw"] = 1
        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from stella file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        # get the reference collision frequency from the stella data
        # ready for conversion to species-specific collision frequencies
        # in the pyrokinetics internal format
        vnew_ref = self._get_physics_var("vnew_ref", 0.0)
        # Load each species into a dictionary
        for i_sp in range(self.data[self._species_knobs_namelist]["nspec"]):
            species_data = CleverDict()

            stella_key = f"species_parameters_{i_sp + 1}"

            stella_data = self.data[stella_key]

            for pyro_key, stella_key in self.pyro_stella_species.items():
                species_data[pyro_key] = stella_data[stella_key]

            # normalisation factor to get into GS2 convention
            normfac = (
                species_data.dens
                * (species_data.z**4)
                / (np.sqrt(species_data.mass) * (species_data.temp**1.5))
            )
            species_data.nu = vnew_ref * normfac

            # assume rotation not implemented in stella
            species_data.omega0 = 0.0 * ureg.vref_most_probable / ureg.lref_minor_radius

            # assume no isolated PVG term in stella
            species_data.domega_drho = (
                0.0 * ureg.vref_most_probable / ureg.lref_minor_radius**2
            )

            if species_data.z == -1:
                name = "electron"
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # normalisations
            species_data.dens *= ureg.nref_electron / ne_norm
            species_data.mass *= ureg.mref_deuterium
            species_data.nu *= ureg.vref_most_probable / ureg.lref_minor_radius
            species_data.temp *= ureg.tref_electron / Te_norm
            species_data.z *= ureg.elementary_charge
            species_data.inverse_lt *= ureg.lref_minor_radius**-1
            species_data.inverse_ln *= ureg.lref_minor_radius**-1

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.normalise()

        zeff = self._get_physics_var("zeff", None)
        if zeff is not None:
            local_species.zeff = zeff * ureg.elementary_charge
        else:
            local_species.zeff = 1.0 * ureg.elementary_charge

        return local_species

    def _read_range_grid(self):
        range_options = self.data[self._kt_grids_range_namelist]
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
        box = self.data[self._kt_grids_box_namelist]
        keys = box.keys()

        grid_data = {}

        # Set up ky grid
        if "ny" in keys:
            grid_data["nky"] = int((box["ny"] - 1) / 3 + 1)
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
        else:
            raise RuntimeError("kx grid details not found in {keys}")

        shat_params = self.pyro_stella_miller["shat"]
        shat = self.data[shat_params[0]][shat_params[1]]
        if abs(shat) > 1e-6:
            jtwist_default = max(int(2 * pi * shat + 0.5), 1)
            jtwist = box.get("jtwist", jtwist_default)
            grid_data["kx"] = grid_data["ky"] * shat * 2 * pi / jtwist
        else:
            grid_data["kx"] = 2 * pi / box["x0"]

        return grid_data

    def _read_grid(self):
        """Read the perpendicular wavenumber grid"""

        grid_option = self.data[self._kt_grids_knobs_namelist].get(
            "grid_option", "range"
        )

        GRID_READERS = {
            "default": self._read_range_grid,
            "range": self._read_range_grid,
            "box": self._read_box_grid,
        }

        try:
            reader = GRID_READERS[grid_option]
        except KeyError:
            valid_options = ", ".join(f"'{option}'" for option in GRID_READERS)
            raise ValueError(
                f"Unknown stella 'kt_range_knobs::grid_option', '{grid_option}'. Expected one of {valid_options}"
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
        numerics_data["phi"] = self._get_numerical_var("fphi", 1.0) > 0.0
        numerics_data["apar"] = self._get_physics_var("include_apar", False)
        numerics_data["bpar"] = self._get_physics_var("include_bpar", False)

        # Set time stepping
        delta_time = self._get_numerical_var("delt", 0.005)
        numerics_data["delta_time"] = delta_time

        tend = self._get_numerical_var("tend", None)
        if tend is not None and tend > 0:
            numerics_data["max_time"] = tend
        else:
            numerics_data["max_time"] = (
                self._get_numerical_var("nstep", 50000) * delta_time
            )

        numerics_data["nonlinear"] = self.is_nonlinear()

        numerics_data.update(self._read_grid())

        # z grid
        numerics_data["ntheta"] = self.data[self._zgrid_namelist]["nzed"]
        numerics_data["nperiod"] = self.data[self._zgrid_namelist].get("nperiod", 1)

        # Velocity grid
        numerics_data["nenergy"] = self.data[self._vpamu_namelist]["nvgrid"]
        numerics_data["npitch"] = self.data[self._vpamu_namelist]["nmu"]

        numerics_data["beta"] = self._get_beta()

        numerics_data["gamma_exb"] = self._get_physics_var("g_exb", 0.0)

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in normalisation values from input file

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
            "raxis_rmaj": None,
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
        for i_sp in range(self.data[self._species_knobs_namelist]["nspec"]):
            species_key = f"species_parameters_{i_sp + 1}"

            dens = self.data[species_key]["dens"]
            temp = self.data[species_key]["temp"]
            mass = self.data[species_key]["mass"]

            # Find all reference values
            if self.data[species_key]["z"] == -1:
                electron_density = dens
                electron_temperature = temp
                e_mass = mass
                electron_index = i_sp
                found_electron = True

            if np.isclose(dens, 1.0):
                reference_density_index.append(i_sp)
            if np.isclose(temp, 1.0):
                reference_temperature_index.append(i_sp)

            densities.append(dens)
            temperatures.append(temp)
            masses.append(mass)

        adiabatic_electron_flags = ["iphi00=2", "field-line-average-term"]

        adiabatic_option = self._get_physics_var("adiabatic_option", "")
        if not found_electron and adiabatic_option in adiabatic_electron_flags:
            found_electron = True
            electron_density = 1.0 / self._get_physics_var("nine", 1.0)
            electron_temperature = 1.0 / self._get_physics_var("tite", 1.0)
            e_mass = (electron_mass / deuterium_mass).m
            n_species = self.data[self._species_knobs_namelist]["nspec"]
            electron_index = n_species + 1

            if np.isclose(electron_density, 1.0):
                reference_density_index.append(n_species + 1)
            if np.isclose(electron_temperature, 1.0):
                reference_temperature_index.append(n_species + 1)

        ml = self._miller_namelist
        rgeo_rmaj = self.data[ml]["rgeo"] / self.data[ml]["rmaj"]
        major_radius = self.data[ml]["rmaj"]

        minor_radius = 1.0

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
                template_file = gk_templates["STELLA"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

        # Set Miller Geometry bits
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for stella not supported yet"
            )

        # Ensure Miller settings
        ml = self._miller_namelist
        if self._format_version == StellaFormatVersion.V1:
            self.data[self._geo_knobs_namelist]["geometry_option"] = "local"
        else:
            self.data[self._geo_knobs_namelist]["geo_option"] = "miller"
        # Assign Miller values to input file
        for key, val in self.pyro_stella_miller.items():
            self.data[val[0]][val[1]] = local_geometry[key]

        self.data[ml]["rgeo"] = local_geometry.Rmaj
        # get stella normalised beta_prime
        self.data[ml]["betaprim"] = -0.5 * local_geometry.beta_prime

        self.data[ml]["kapprim"] = (
            local_geometry.s_kappa * local_geometry.kappa / local_geometry.rho
        )
        self.data[ml]["tri"] = np.arcsin(local_geometry.delta)
        self.data[ml]["triprim"] = local_geometry["s_delta"] / local_geometry.rho

        # Set local species bits
        n_species = local_species.nspec
        sk = self._species_knobs_namelist
        self.data[sk]["nspec"] = local_species.nspec
        self.data[sk]["species_option"] = "stella"

        stored_species = len(
            [key for key in self.data.keys() if "species_parameters_" in key]
        )
        extra_species = stored_species - n_species

        if extra_species > 0:
            for i_sp in range(extra_species):
                stella_key = f"species_parameters_{i_sp + 1 + n_species}"
                if stella_key in self.data:
                    self.data.pop(stella_key)

        for iSp, name in enumerate(local_species.names):
            # add new outer params for each species
            species_key = f"species_parameters_{iSp + 1}"
            if species_key not in self.data:
                self.data[species_key] = copy(self.data["species_parameters_1"])

            if name == "electron":
                self.data[species_key]["type"] = "electron"
            else:
                self.data[species_key]["type"] = "ion"

            for key, val in self.pyro_stella_species.items():
                self.data[species_key][val] = local_species[name][key]

        if "electron" in local_species.names:
            if local_species.electron.domega_drho.m != 0:
                warnings.warn(
                    "stella does not support PVG term so this is not included"
                )

        self._set_physics_var("zeff", local_species.zeff)

        beta_ref = convention.beta if local_norm else 0.0
        self._set_physics_var(
            "beta", numerics.beta if numerics.beta is not None else beta_ref
        )

        # set the reference collision frequency
        specref = self.data["species_parameters_1"]
        normfac = (
            (specref["z"] ** 4)
            * specref["dens"]
            / (np.sqrt(specref["mass"]) * (specref["temp"] ** 1.5))
        )
        nameref = local_species.names[0]
        vnew_ref = local_species[nameref]["nu"].to(convention)
        # convert to the reference parameter from the species parameter of species 1
        self._set_physics_var("vnew_ref", vnew_ref / normfac)

        # Set numerics bits. Only force include_collisions on when vnew_ref > 0
        # and the existing value isn't an explicit False — respect a user's
        # intentional collisionless run even when species have nu > 0.
        diss = self.data[self._dissipation_namelist]
        if vnew_ref > 0.0:
            diss.setdefault("include_collisions", True)
        else:
            diss["include_collisions"] = False

        # Set no. of fields
        self._set_numerical_var("fphi", 1.0 if numerics.phi else 0.0)
        self._set_physics_var("include_apar", numerics.apar)
        self._set_physics_var("include_bpar", numerics.bpar)
        if numerics.apar or numerics.bpar:
            self._set_physics_var("include_electromagnetic", True)

        # Set time stepping
        self._set_numerical_var("delt", numerics.delta_time)
        self._set_numerical_var("tend", int(numerics.max_time.m))

        kt_knobs = self._kt_grids_knobs_namelist
        kt_range = self._kt_grids_range_namelist
        kt_box = self._kt_grids_box_namelist
        zg = self._zgrid_namelist

        if numerics.nky == 1:
            self.data[kt_knobs]["grid_option"] = "range"

            if kt_range not in self.data.keys():
                self.data[kt_range] = {}
            try:
                ky = (
                    numerics.ky[0]
                    * (1 * convention.bref / local_norm.stella.bref).to_base_units()
                )
            except IndexError:
                ky = (
                    numerics.ky
                    * (1 * convention.bref / local_norm.stella.bref).to_base_units()
                )
            self.data[kt_range]["aky_min"] = ky
            self.data[kt_range]["aky_max"] = ky
            self.data[kt_range]["theta0_min"] = numerics.theta0
            self.data[kt_range]["theta0_max"] = numerics.theta0
            self.data[kt_range]["naky"] = 1
            self.data[kt_range]["nakx"] = 1
            self.data[zg]["nperiod"] = numerics.nperiod

        else:
            self.data[kt_knobs]["grid_option"] = "box"

            if kt_box not in self.data.keys():
                self.data[kt_box] = {}

            self.data[kt_box]["nx"] = int(((numerics.nkx - 1) * 3 / 2) + 1)
            self.data[kt_box]["ny"] = int(((numerics.nky - 1) * 3) + 1)

            # Fix y0: in v1, y0 must be positive (box size, dky = 1/y0)
            # In old format, negative y0 was used as direct ky specification
            if self._format_version == StellaFormatVersion.V1:
                self.data[kt_box]["y0"] = 1.0 / numerics.ky
            else:
                self.data[kt_box]["y0"] = -numerics.ky

            # Currently forces NL sims to have nperiod = 1
            self.data[zg]["nperiod"] = 1

            shat = local_geometry.shat
            if abs(shat) < 1e-6:
                self.data[kt_box]["x0"] = 2 * pi / numerics.kx
            else:
                if numerics.kx == 0:
                    self.data[kt_box]["jtwist"] = 1
                else:
                    self.data[kt_box]["jtwist"] = int(
                        (numerics.ky * shat * 2 * pi / numerics.kx) + 0.1
                    )

        self.data[zg]["nzed"] = numerics.ntheta

        self.data[self._vpamu_namelist]["nvgrid"] = numerics.nenergy
        self.data[self._vpamu_namelist]["nmu"] = numerics.npitch

        self._set_physics_var("g_exb", numerics.gamma_exb)

        if self._format_version == StellaFormatVersion.V1:
            if "gyrokinetic_terms" not in self.data:
                self.data["gyrokinetic_terms"] = f90nml.Namelist()
            self.data["gyrokinetic_terms"]["include_nonlinear"] = numerics.nonlinear
        else:
            self.data[self._parameters_physics]["nonlinear"] = numerics.nonlinear

        if not local_norm:
            return

        try:
            (1 * convention.tref).to("keV")
            si_units = True
        except PyroContextError:
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
        for i_sp in range(self.data[self._species_knobs_namelist]["nspec"]):
            stella_key = f"species_parameters_{i_sp + 1}"
            if (
                self.data[stella_key]["z"] == -1
                and self.data[stella_key]["type"] == "electron"
            ):
                ne = self.data[stella_key]["dens"]
                Te = self.data[stella_key]["temp"]
                found_electron = True
                break

        adiabatic_electron_flags = ["iphi00=2", "field-line-average-term"]

        adiabatic_option = self._get_physics_var("adiabatic_option", "")
        if not found_electron and adiabatic_option in adiabatic_electron_flags:
            found_electron = True
            ne = 1.0 / self._get_physics_var("nine", 1.0)
            Te = 1.0 / self._get_physics_var("tite", 1.0)

        if not found_electron:
            raise TypeError(
                "Pyro currently only supports electron species with charge = -1"
            )

        return ne, Te

    def _get_beta(self):
        return self._get_physics_var("beta", 0.0)


class GKOutputReaderSTELLA(FileReader, file_type="STELLA", reads=GKOutput):
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
        if not fields and coords["linear"]:
            eigenvalues = self._get_eigenvalues(raw_data, coords["time_divisor"])
        # Assign units and return GKOutput
        convention = getattr(norm, gk_input.norm_convention)
        norm.default_convention = output_convention.lower()
        gk_input.convention = convention

        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = ("species", "kx", "ky", "time")
        moment_dims = ("species", "kx", "ky", "time")

        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["zed"],
                energy=coords["vpa"],
                pitch=coords["mu"],
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
            linear=coords["linear"],
            gk_code="STELLA",
            input_file=input_str,
            normalise_flux_moment=True,
            output_convention=output_convention,
            input_convention=convention.name,
            jacobian=coords["jacobian"],
        )

    def verify_file_type(self, filename: PathLike):
        import xarray as xr

        try:
            # warnings.filterwarnings("error")
            data = xr.open_dataset(filename)
        except RuntimeWarning:
            warnings.resetwarnings()
            raise RuntimeError("Error occurred reading stella output file")
        warnings.resetwarnings()

        if "software_name" in data.attrs:
            if data.attrs["software_name"] != "stella":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'software_name' for a stella file"
                )
        elif "code_info" in data.data_vars:
            if data["code_info"].long_name != "stella":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'code_info' for a stella file"
                )
        elif "stella_help" in data.attrs.keys():
            pass
        else:
            raise RuntimeError(f"file '{filename}' missing expected stella attributes")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Gets path by removing ".in" and replacing it with ".out.nc"
        """
        filename = Path(filename)
        return filename.parent / (filename.stem + ".out.nc")

    @staticmethod
    def _get_raw_data(filename: PathLike) -> Tuple[xr.Dataset, GKInputSTELLA, str]:
        import xarray as xr

        raw_data = xr.open_dataset(filename)
        # Read input file from netcdf, store as GKInputSTELLA
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
        gk_input = GKInputSTELLA()
        gk_input.read_str(input_str)
        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: xr.Dataset, gk_input: GKInputSTELLA, downsize: int
    ) -> Dict[str, Any]:
        # ky coords
        ky = raw_data["ky"].data

        # time coords
        time_divisor = 1

        time = raw_data["t"].data / time_divisor

        # kx coords
        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(raw_data["kx"].data)

        # zed coords
        zed = raw_data["zed"].data

        # vpa coords
        vpa = raw_data["vpa"].data

        # mu coords
        mu = raw_data["mu"].data

        # moment coords
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "upar", "spitzer2"]

        # field coords
        # stella is hardcoded to require phi, only apar and bpar are optional
        field_vals = {"phi": True}
        for field, default in zip(["apar", "bpar"], [False, False]):
            try:
                field_vals[field] = gk_input._get_physics_var(
                    f"include_{field}", default
                )
            except KeyError:
                field_vals[field] = default

        fields = [field for field, val in field_vals.items() if val > 0]

        # species coords
        # TODO is there some way to get this info without looking at the input data?
        species = []
        ion_num = 0
        for idx in range(gk_input.data[gk_input._species_knobs_namelist]["nspec"]):
            if gk_input.data[f"species_parameters_{idx + 1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        local_geometry = gk_input.get_local_geometry()
        metric_terms = MetricTerms(local_geometry, ntheta=len(zed) * 4)
        theta_mod = np.mod(zed, 2 * np.pi)
        Jacobian = np.interp(
            theta_mod,
            metric_terms.regulartheta,
            metric_terms.Jacobian,
            period=2 * np.pi,
        )

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "zed": zed,
            "vpa": vpa,
            "mu": mu,
            "linear": gk_input.is_linear(),
            "time_divisor": time_divisor,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "downsize": downsize,
            "jacobian": Jacobian,
        }

    @staticmethod
    def _get_fields(raw_data: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        to have fields written out versus time, we must set
        &stella_diagnostics_knobs
         write_phi_vs_time = .true.
         write_apar_vs_time = .true.
         write_bpar_vs_time = .true.
        /
        at the same time, we must also set
        &physics_flags
         include_apar = .true.
         include_bpar = .true.
        /
        to include apar and bpar in the simulation
        """
        field_names = ("phi", "apar", "bpar")
        results = {}

        # Loop through all fields and add field if it exists
        for field_name in field_names:
            key = f"{field_name}_vs_t"
            if key not in raw_data:
                continue

            # raw_field has coords (t, tube, zed, kx, ky, real/imag).
            # We wish to transpose that to (real/imag,zed,kx,ky,t)
            # Selecting first index in tube
            field = raw_data[key].transpose("tube", "ri", "zed", "kx", "ky", "t").data
            field = field[0, 0, ...] + 1j * field[0, 1, ...]

            # Adjust fields to account for differences in defintions/normalisations
            # A||_stella = 0.5 * A||_gs2
            # B||_stella = B||_gs2 * B
            # infer from GS2 script that no adjustments required here

            # Shift kx=0 to middle of axis
            field = np.fft.fftshift(field, axes=1)
            results[field_name] = field

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputSTELLA,
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
        gk_input: GKInputSTELLA,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        For stella to print fluxes(t,species) to the netcdf file, at the present time we must
        be using the branch https://github.com/stellaGK/stella/tree/development/apar2plusbpar.
        Otherwise the fluxes are automatically written to the ascii text files.
        To make the fluxes as a function of ky and kx be written to the netcdf file, set
        &stella_diagnostics_knobs
         write_kspectra = .true.
        /
        Flux contributions as a function of kx ky and z are available in stella with
        &stella_diagnostics_knobs
         write_fluxes_kxkyz = .true.
        /
        These are not supported to be read here as they are a function of tubes and zed in addition
        to ky, kx.
        """
        fluxes_dict = {
            "particle": "pflux_vs_s",
            "heat": "qflux_vs_s",
            "momentum": "vflux_vs_s",
        }
        fluxes_dict_old = {
            "particle": "pflx",
            "heat": "qflx",
            "momentum": "vflx",
        }

        # Get species names from input file
        species = []
        ion_num = 0
        for idx in range(gk_input.data[gk_input._species_knobs_namelist]["nspec"]):
            if gk_input.data[f"species_parameters_{idx+1}"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        results = {}

        coord_names = ["flux", "species", "kx", "ky", "time"]
        fluxes = np.zeros([len(coords[name]) for name in coord_names])

        for iflux, (pyro_flux, stella_flux) in enumerate(fluxes_dict.items()):
            # total fluxes
            flux_key = f"{stella_flux}"
            # flux contributions by kx ky (averaged over z)
            vskxky_key = f"{stella_flux}_vs_kxky"
            # flux constributions by kx ky z
            # vskxkyz_key = f"{stella_flux}_kxky"

            # commented out as not (yet) supported
            # if vskxkyz_key in raw_data.data_vars:
            #    key = vskxkyz_key
            #    flux = raw_data[key].transpose("species", "tube", "zed", "kx", "ky", "t")
            if vskxky_key in raw_data.data_vars:
                key = vskxky_key
                flux = raw_data[key].transpose("species", "kx", "ky", "t")
            elif fluxes_dict_old[pyro_flux] in raw_data.data_vars:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                flux = raw_data[fluxes_dict_old[pyro_flux]]
                flux = flux.expand_dims("ky").transpose("species", "ky", "t")
                flux = flux.expand_dims("kx").transpose("species", "kx", "ky", "t")
            elif flux_key in raw_data.data_vars:
                # coordinates from raw are (t,species)
                # convert to (species, ky, t)
                flux = raw_data[flux_key]
                flux = flux.expand_dims("ky").transpose("species", "ky", "t")
                flux = flux.expand_dims("kx").transpose("species", "kx", "ky", "t")
            else:
                continue

            fluxes[iflux, ...] = flux.data

        if gk_input.is_linear() and gk_input.data.get(
            gk_input._diagnostics_fluxes_namelist, {}
        ).get("flux_norm", True):
            jacob = raw_data["jacob"].data
            grho = raw_data["grho"].data
            theta = raw_data["zed"].data
            theta_append = 2 * theta[-1] - theta[-2]
            dtheta = np.diff(theta, append=theta_append)
            flux_norm = np.sum(jacob * dtheta) / np.sum(jacob * dtheta * grho)
        else:
            flux_norm = 1.0

        for iflux, flux in enumerate(coords["flux"]):
            if not np.all(fluxes[iflux, ...] == 0):
                results[flux] = fluxes[iflux, ...] / flux_norm

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: xr.Dataset, time_divisor: float
    ) -> Dict[str, np.ndarray]:
        # should only be called if no field data were found
        mode_frequency = raw_data.omega.isel(ri=0).transpose("kx", "ky", "t")
        growth_rate = raw_data.omega.isel(ri=1).transpose("kx", "ky", "t")
        return {
            "mode_frequency": mode_frequency.data / time_divisor,
            "growth_rate": growth_rate.data / time_divisor,
        }
