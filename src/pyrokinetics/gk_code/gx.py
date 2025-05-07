from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import netCDF4 as nc
import numpy as np
import toml
from cleverdict import CleverDict
from scipy.integrate import cumulative_trapezoid, trapezoid
from sympy import integer_log

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


class GKInputGX(GKInput, FileReader, file_type="GX", reads=GKInput):
    """
    Class that can read GX input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GX"
    default_file_name = "input.in"
    norm_convention = "gx"
    _convention_dict = {}

    pyro_gx_miller = {
        "rho": ["Geometry", "rhoc"],
        "Rmaj": ["Geometry", "Rmaj"],
        "q": ["Geometry", "qinp"],
        "kappa": ["Geometry", "akappa"],
        "shat": ["Geometry", "shat"],
        "shift": ["Geometry", "shift"],
        "beta_prime": ["Geometry", "betaprim"],
    }

    pyro_gx_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 1.0,
        "q": 1.4,
        "kappa": 1.0,
        "shat": 0.8,
        "shift": 0.0,
        "beta_prime": 0.0,
    }

    pyro_gx_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "nu": "vnewk",
        "inverse_lt": "tprim",
        "inverse_ln": "fprim",
    }

    def _generate_three_smooth_numbers(self, n):
        """
        Generates a list of three smooth numbers (from the sequence A003586,
        see https://oeis.org/A003586) with the final element at least larger than n
        """

        def _A003586(j):
            """
            Generates the jth three smooth number
            """

            def _bisection(f, xmin=0, xmax=1):
                while f(xmax) > xmax:
                    xmax <<= 1
                while xmax - xmin > 1:
                    xmid = xmax + xmin >> 1
                    if f(xmid) <= xmid:
                        xmax = xmid
                    else:
                        xmin = xmid
                return xmax

            def f(x):
                return (
                    j
                    + x
                    - sum(
                        (x // 3**i).bit_length()
                        for i in range(integer_log(x, 3)[0] + 1)
                    )
                )

            return _bisection(f, j, j)

        three_smooth_numbers = []
        _n = 0
        while True:
            _n = _A003586(_n + 1)
            three_smooth_numbers.append(_n)
            if three_smooth_numbers[-1] > n:
                break

        return three_smooth_numbers

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GX input file into a dictionary
        """
        with open(filename, "r") as f:
            self.data = toml.load(f)

        return self.data

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GX input file given as string
        Uses default read_str, which assumes input_string is a Fortran90 namelist
        """

        cleaned_lines = []
        for line in input_string.splitlines():
            stripped_line = line.split("#")[0].strip()  # Remove comments and whitespace
            if stripped_line:  # Ignore empty lines
                cleaned_lines.append(stripped_line)

        cleaned_toml = "\n".join(cleaned_lines)
        self.data = toml.loads(cleaned_toml)

        return self.data

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads GX input file given as dict
        Uses default read_dict, which assumes input is a dict
        """

        for key, values in input_dict["species"].items():
            try:
                input_dict["species"][key] = [float(value) for value in values]
            except ValueError:
                continue
        self.data = input_dict
        return self.data

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid gx input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a GX input file,
        # but they are needed by Pyrokinetics.

        expected_keys = [
            "Dimensions",
            "Domain",
            "Physics",
            "Time",
            "Initialization",
            "Geometry",
            "species",
            "Boltzmann",
            # "Dissipation",
            # "Restart"
            "Diagnostics",
            # "Expert",
            # "Forcing"
        ]
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm: Normalisation = None,
        code_normalisation: str = None,
    ):
        # Create directories if they don't exist already
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        def drop_array(namelist):
            for key, val in namelist.items():
                if isinstance(val, np.ndarray):
                    namelist[key] = val.tolist()
            return namelist

        if local_norm is None:
            local_norm = Normalisation("write")

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        for name, namelist in self.data.items():
            if name == "debug":
                continue
            self.data[name] = convert_dict(namelist, convention)
            self.data[name] = drop_array(self.data[name])

        with open(filename, "w") as f:
            toml.dump(self.data, f)

    def is_nonlinear(self) -> bool:
        try:
            return self.data["Physics"]["nonlinear_mode"]
        except KeyError:
            return False

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GX input file
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

        gx_geo_option = self.data["Geometry"]["geo_option"]

        if gx_geo_option not in [
            "miller",
        ]:
            raise NotImplementedError(
                f"GX equilibrium option {gx_geo_option} not implemented"
            )

        local_geometry = self.get_local_geometry_miller()

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Basic Miller object from GX file
        """
        # This assumes that gx_geo_option = "miller", with all options specified
        # in the geometry group. Note that the GX parameter "tri" is actually
        # the same as pyro's "delta" parameter; compare the use
        # /gx/geometry_modules/miller/gx_geo.py to /pyrokinetics/src/local_geometry/miller.py

        miller_data = default_miller_inputs()

        for (pyro_key, (gx_param, gx_key)), gx_default in zip(
            self.pyro_gx_miller.items(), self.pyro_gx_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data[gx_param].get(gx_key, gx_default)

        rho = miller_data["rho"]
        kappa = miller_data["kappa"]
        miller_data["delta"] = self.data["Geometry"].get("tri", 0.0)
        miller_data["s_kappa"] = self.data["Geometry"].get("akappri", 0.0) * rho / kappa
        miller_data["s_delta"] = (
            self.data["Geometry"].get("tripri", 0.0)
            * rho
            / np.sqrt(1 - self.data["Geometry"].get("tri", 0.0) ** 2)
        )

        beta = self._get_beta()

        # Assume pref*8pi*1e-7 = 1.0
        miller_data["B0"] = np.sqrt(1.0 / beta) if beta != 0.0 else None

        miller_data["ip_ccw"] = 1
        miller_data["bt_ccw"] = 1

        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from GX file
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
        gx_data = self.data["species"]

        for i_sp in range(self.data["Dimensions"]["nspecies"]):
            species_data = CleverDict()

            for pyro_key, gx_key in self.pyro_gx_species.items():
                species_data[pyro_key] = gx_data[gx_key][i_sp]

            species_data.omega0 = 0.0
            species_data.domega_drho = 0.0

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

        local_species.set_zeff()

        return local_species

    def _read_grid(self, drho_dpsi):
        domain = self.data["Domain"]
        dimensions = self.data["Dimensions"]
        physics = self.data["Physics"]

        grid_data = {}

        # Set up ky grid
        if "ny" in dimensions.keys():
            grid_data["nky"] = int((dimensions["ny"] - 1) / 3 + 1)
        elif "nky" in dimensions.keys():
            grid_data["nky"] = dimensions["nky"]
        else:
            raise RuntimeError(f"ky grid details not found in {dimensions.keys()}")

        if "y0" not in domain.keys():
            raise RuntimeError(f"Min ky details not found in {domain.keys()}")

        # Treat run with nky = 2 as 1
        if grid_data["nky"] == 2 and not physics["nonlinear_mode"]:
            grid_data["nky"] = 1
        grid_data["ky"] = 1.0 / domain["y0"]

        # Set up kx grid. If nkx is specified in the gx input file, we have to
        # go via nx to compute the correct nkx for pyro.
        if "nx" in dimensions.keys():
            grid_data["nkx"] = int(2 * (dimensions["nx"] - 1) / 3 + 1)
        elif "nkx" in dimensions.keys():
            nx = int(3 * ((dimensions["nkx"] - 1) // 2) + 1)
            grid_data["nkx"] = int(1 + 2 * (nx - 1) / 3)
        else:
            raise RuntimeError("kx grid details not found in {keys}")

        # TODO this needs to be changed to use the geometry coefficients from
        # the output file if they can be found, as the following solution will
        # only work for axisymmetric equilibria (Miller)
        shat = self.data["Geometry"]["shat"]

        if "x0" in domain.keys():
            x0 = domain["x0"]
        elif abs(shat) > 1e-6:
            nperiod = dimensions["nperiod"]
            twist_shift_geo_fac = 2 * shat * (2 * nperiod - 1) * pi
            jtwist = max(int(round(twist_shift_geo_fac)), 1)
            x0 = domain["y0"] * abs(jtwist) / abs(twist_shift_geo_fac)

        else:  # Assume x0 = y0 otherwise (the case for periodic BCs)
            x0 = domain["y0"]

        grid_data["kx"] = 1 / x0 / 2

        return grid_data

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        numerics_data = {}

        # Get beta to set proper defaults of field multipliers
        numerics_data["beta"] = self._get_beta()

        # Set number of fields
        numerics_data["phi"] = self.data["Physics"].get("fphi", 1.0) > 0.0
        numerics_data["apar"] = (
            self.data["Physics"].get(
                "fapar", 1.0 if numerics_data["beta"] > 0.0 else 0.0
            )
            > 0.0
        )
        numerics_data["bpar"] = (
            self.data["Physics"].get(
                "fbpar", 1.0 if numerics_data["beta"] > 0.0 else 0.0
            )
            > 0.0
        )

        # Set time stepping
        delta_time = self.data["Time"].get("dt", 0.05)
        numerics_data["delta_time"] = delta_time
        if "nstep" in self.data["Time"]:
            numerics_data["max_time"] = (
                self.data["Time"].get("nstep", 10000) * delta_time
            )
        else:
            numerics_data["max_time"] = self.data["Time"].get("t_max", 1000.0)

        numerics_data["nonlinear"] = self.is_nonlinear()

        local_geometry = self.get_local_geometry()

        # Specifically ignore Rmaj/Rgeo so ky = n/Lref drho_pyro/dpsi_pyro [1 / rhoref]
        drho_dpsi = (
            self.data["Geometry"]["qinp"]
            / self.data["Geometry"]["rhoc"]
            / local_geometry.bunit_over_b0
        ).m

        numerics_data.update(self._read_grid(drho_dpsi))

        # Load theta0 if appropriate
        if numerics_data["nky"] == 1 and numerics_data["nkx"] == 3:
            numerics_data["theta0"] = numerics_data["kx"][-1] / (
                numerics_data["ky"] * self.data["Geometry"]["shat"]
            )

        # Theta grid
        numerics_data["ntheta"] = self.data["Dimensions"]["ntheta"]
        numerics_data["nperiod"] = self.data["Dimensions"]["nperiod"]

        # Velocity grid. Note that GX is not in energy and pitch angle coordinates
        numerics_data["nenergy"] = self.data["Dimensions"]["nlaguerre"]
        numerics_data["npitch"] = self.data["Dimensions"]["nhermite"]

        numerics_data["gamma_exb"] = self.data["Physics"].get("g_exb", 0.0)

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in reference values from input file

        """

        return {}

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
            "vref": "nrl",
            "rhoref": "gx",
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
        species_data = self.data["species"]

        for i_sp in range(self.data["Dimensions"]["nspecies"]):

            dens = species_data["dens"][i_sp]
            temp = species_data["temp"][i_sp]
            mass = species_data["mass"][i_sp]

            # Find all reference values
            if species_data["z"][i_sp] == -1:
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

        if (
            not found_electron
            and self.data["Boltzmann"]["add_Boltzmann_species"] is True
            and self.data["Boltzmann"]["Boltzmann_type"] == "electrons"
        ):
            found_electron = True

            # Set density from quasineutrality
            qn_dens = 0.0
            for i_sp in range(self.data["Dimensions"]["nspecies"]):
                qn_dens += (
                    self.data["species"]["z"][i_sp] * self.data["species"]["dens"][i_sp]
                )

            electron_density = qn_dens
            electron_temperature = 1.0 / self.data["Boltzmann"].get("tau_fac", 1.0)
            e_mass = (electron_mass / deuterium_mass).m
            n_species = self.data["Dimensions"]["nspecies"]
            electron_index = n_species + 1

            if np.isclose(electron_density, 1.0):
                reference_density_index.append(electron_index)
            if np.isclose(electron_temperature, 1.0):
                reference_temperature_index.append(electron_index)

        rgeo_rmaj = self.data["Geometry"]["R_geo"] / self.data["Geometry"]["Rmaj"]
        major_radius = self.data["Geometry"]["Rmaj"]

        # TODO May need fixing
        if major_radius != 1.0:
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
        gpu_optimised_grid: bool = False,
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
                template_file = gk_templates["GX"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

        # Set Miller Geometry bits
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GX not supported yet"
            )

        # Ensure Miller settings
        self.data["Geometry"]["geo_option"] = "miller"

        # Assign Miller values to input file
        for key, val in self.pyro_gx_miller.items():
            self.data[val[0]][val[1]] = local_geometry[key]

        self.data["Geometry"]["akappri"] = (
            local_geometry.s_kappa * local_geometry.kappa / local_geometry.rho
        )
        self.data["Geometry"]["tri"] = local_geometry.delta
        self.data["Geometry"]["tripri"] = (
            local_geometry["s_delta"]
            * np.sqrt(1 - local_geometry.delta**2)
            / local_geometry.rho
        )
        self.data["Geometry"]["R_geo"] = (
            local_geometry.Rmaj
            * (1 * local_norm.gs2.bref / convention.bref).to_base_units()
        )

        # Set local species bits
        n_species = local_species.nspec
        self.data["Dimensions"]["nspecies"] = n_species

        self.data["species"] = {
            "z": np.empty(n_species),
            "mass": np.empty(n_species),
            "dens": np.empty(n_species),
            "temp": np.empty(n_species),
            "tprim": np.empty(n_species),
            "fprim": np.empty(n_species),
            "vnewk": np.empty(n_species),
            "type": [],
        }

        local_species_units = {}
        for i_sp, name in enumerate(local_species.names):
            # add new outer params for each species
            if name == "electron":
                self.data["species"]["type"].append("electron")
            else:
                self.data["species"]["type"].append("ion")

            for key, val in self.pyro_gx_species.items():
                self.data["species"][val][i_sp] = local_species[name][key].m
                local_species_units[val] = local_species[name][key].units

        for key, units in local_species_units.items():
            self.data["species"][key] *= units

        # Handling of adiabatic species
        if n_species < 2:
            self.data["Boltzmann"]["add_Boltzmann_species"] = True
            if "electron" not in local_species.names:
                self.data["Boltzmann"]["Boltzmann_type"] = "electrons"
            else:
                self.data["Boltzmann"]["Boltzmann_type"] = "ions"

        beta_ref = convention.beta if local_norm else 0.0
        self.data["Physics"]["beta"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        # Set numerics bits
        # Set no. of fields
        self.data["Physics"]["fphi"] = 1.0 if numerics.phi else 0.0
        self.data["Physics"]["fapar"] = 1.0 if numerics.apar else 0.0
        self.data["Physics"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        if abs(numerics.gamma_exb.m) > 0.0:
            self.data["Physics"]["g_exb"] = numerics.gamma_exb

        # Set time stepping
        self.data["Time"]["dt"] = numerics.delta_time
        self.data["Time"]["t_max"] = numerics.max_time

        # Set y0 (same for linear/nonlinear)
        self.data["Domain"]["y0"] = 1.0 / (
            numerics.ky * (1 * convention.bref / local_norm.gx.bref).to_base_units()
        )

        # Set the perpendicular grid. It is reccommended to set (nx, ny) for
        # nonlinear calculations, and (nkx, nky) for linear runs.
        if numerics.nonlinear:
            ny = int(3 * ((numerics.nky - 1)) + 1)
            nx = int(3 * ((numerics.nkx - 1) / 2) + 1)

            if gpu_optimised_grid:
                ns = [nx, ny]
                three_smooth_numbers = self._generate_three_smooth_numbers(max(ns))
                ns = [
                    int(
                        min(
                            [
                                x
                                for x in three_smooth_numbers
                                if (abs(x - n) <= 2) and (x > n)
                            ],
                            default=n,
                        )
                    )
                    for n in ns
                ]
                nx, ny = ns

            self.data["Dimensions"]["ny"] = ny
            self.data["Dimensions"]["nx"] = nx
        else:
            # Since GX includes ky=0 in its definition of nky, we have to add one
            # here to match the other codes (e.g. GS2)
            self.data["Dimensions"]["nky"] = numerics.nky + 1

            if not np.isclose(numerics.theta0, 0.0):
                self.data["Dimensions"]["nkx"] = 3
                kx_min = numerics.ky * local_geometry.shat * numerics.theta0
                self.data["Domain"]["x0"] = 1.0 / (kx_min)
            else:
                self.data["Dimensions"]["nkx"] = numerics.nkx
                if numerics.nkx == 1:
                    if hasattr(self.data["Dimensions"], "x0"):
                        self.data["Dimensions"].pop("x0")

        self.data["Dimensions"]["nperiod"] = numerics.nperiod

        self.data["Dimensions"]["ntheta"] = numerics.ntheta
        self.data["Dimensions"]["nlaguerre"] = numerics.nenergy
        self.data["Dimensions"]["nhermite"] = numerics.npitch

        self.data["Physics"]["nonlinear_mode"] = numerics.nonlinear

        if not local_norm:
            return

        for name, namelist in self.data.items():
            if name == "debug":
                continue
            self.data[name] = convert_dict(namelist, convention)

    def get_ne_te_normalisation(self):  # TODO Can be removed?
        found_electron = False
        # Load each species into a dictionary
        for i_sp in range(self.data["Dimensions"]["nspecies"]):
            if (
                self.data["species"]["z"][i_sp] == -1
                and self.data["species"]["type"][i_sp] == "electron"
            ):
                ne = self.data["species"]["dens"][i_sp]
                Te = self.data["species"]["temp"][i_sp]
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
        consistent with logic across versions of GX.
        """
        has_parameters = "Physics" in self.data.keys()
        beta_default = 0.0
        if has_parameters:
            beta_default = self.data["Physics"].get("beta", 0.0)
        return self.data["Physics"].get("beta", beta_default)


class GKOutputReaderGX(FileReader, file_type="GX", reads=GKOutput):
    fields_big = ["Phi", "Apar", "Bpar"]

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
        time_indices = self._get_time_indices(raw_data)
        coords = self._get_coords(raw_data, gk_input, downsize, time_indices)
        fields = (
            self._get_fields(raw_data, time_indices, coords) if load_fields else None
        )
        fluxes = (
            self._get_fluxes(raw_data, gk_input, coords, time_indices)
            if load_fluxes
            else None
        )
        moments = (
            self._get_moments(raw_data, gk_input, coords, time_indices)
            if load_moments
            else None
        )

        if coords["linear"] and not fields:
            eigenvalues = self._get_eigenvalues(raw_data, time_indices)
            eigenfunctions = self._get_eigenfunctions(raw_data, coords)
        else:
            # Rely on gk_output to generate eigenvalues and eigenfunctions
            eigenvalues = None
            eigenfunctions = None

        # Assign units and return GKOutput
        convention = getattr(norm, gk_input.norm_convention)
        norm.default_convention = output_convention.lower()

        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = ("field", "species", "kx", "ky", "time")
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
                else Eigenfunctions(
                    eigenfunctions, dims=("field", "theta", "kx", "ky")
                ).with_units(convention)
            ),
            linear=coords["linear"],
            gk_code="GX",
            input_file=input_str,
            normalise_flux_moment=True,
            output_convention=output_convention,
        )

    def verify_file_type(self, filename: PathLike):
        import xarray as xr

        try:
            warnings.filterwarnings("error")
            data = xr.open_dataset(filename)
        except RuntimeWarning:
            warnings.resetwarnings()
            raise RuntimeError("Error occurred reading GX output file")
        warnings.resetwarnings()

        if "Title" in data.attrs:
            if data.attrs["Title"] != "GX simulation data":
                raise RuntimeError(f"file '{filename}' has wrong 'Title' for a GX file")
        else:
            raise RuntimeError(f"file '{filename}' missing expected GX attributes")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Gets path by removing ".in" and replacing it with ".out.nc"
        """
        filename = Path(filename)
        return filename.parent / (filename.stem + ".out.nc")

    @staticmethod
    def _create_in_memory_dataset(source_file):
        """
        Creates an in-memory NetCDF dataset by copying selected groups from the source file
        and adding new variables corresponding to the field data.
        """
        src = source_file

        # Create an in-memory NetCDF dataset
        dst = nc.Dataset(
            "in_memory", mode="w", format="NETCDF4", diskless=True, persist=False
        )

        # Copy global attributes
        dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

        # Copy dimensions
        for dim_name, dim in src.dimensions.items():
            dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

        # Copy selected groups
        groups_to_copy = ["Grids", "Geometry"]
        for group in groups_to_copy:
            if group in src.groups:
                src_group = src.groups[group]
                dst_group = dst.createGroup(group)
                dst_group.setncatts(
                    {attr: src_group.getncattr(attr) for attr in src_group.ncattrs()}
                )

                # Copy dimensions for the group
                for dim_name, dim in src_group.dimensions.items():
                    dst_group.createDimension(
                        dim_name, len(dim) if not dim.isunlimited() else None
                    )

                # Copy variables for the group
                for var_name, var in src_group.variables.items():
                    new_var = dst_group.createVariable(
                        var_name, var.datatype, var.dimensions
                    )
                    new_var.setncatts(
                        {attr: var.getncattr(attr) for attr in var.ncattrs()}
                    )
                    new_var[:] = var[:]

        # Create "Diagnostic" group and add fake data for fields
        diagnostics_group = dst.createGroup("Diagnostics")

        for var_name in GKOutputReaderGX.fields_big:
            var = diagnostics_group.createVariable(
                var_name, "f8", ("time", "ky", "kx", "theta", "ri")
            )
            var.setncatts({"description": f"{var_name} field"})
            var[:, :, :, :, :] = np.zeros(
                (
                    len(dst.dimensions["time"]),
                    len(dst.dimensions["ky"]),
                    len(dst.dimensions["kx"]),
                    len(dst.dimensions["theta"]),
                    2,
                )
            )

        return dst

    @staticmethod
    def _get_raw_data(
        filename: PathLike,
    ) -> Tuple[Dict[str, nc.Dataset], GKInputGX, str]:
        """
        Extracts the raw data from the output files '.out.nc' and '.big.nc'. If '.big.nc'
        does not exist, it will create trivial data to allow all other routines to proceed
        without error.
        """
        # TODO Possibly write a wrapper to import the data in xarray format?
        # However, there are no obvious issues with outputting the netcdf datasets
        # since _get_coords, _get_fluxes etc. all output dict objects,
        # which are then composed into the final xarray. The issue arises becase xr.open_dataset
        # does not support automatic loading of groups and subgroups from the netcdf file.

        # Load standard raw data
        raw_data_out = nc.Dataset(filename, mode="r")

        # Load large (field, moment) raw data if it exists. Otherwise create "fake" data
        filename_big = Path(filename.parent / (filename.stem.split(".")[0] + ".big.nc"))
        if filename_big.exists():
            raw_data_big = nc.Dataset(filename_big, mode="r")
        else:
            warn_msg = (
                f"Unable to locate {filename_big}. Any field data shown will be zero."
            )
            warnings.warn(warn_msg, UserWarning)
            raw_data_big = GKOutputReaderGX._create_in_memory_dataset(raw_data_out)

        # TODO GX does not currently store the input file as a str in the output file,
        # so we will have to pass the existing input file to GKInputGX. When this
        # is later changed to use read_str, read_str() in gk_input.py needs to be changed
        # to allow for parsing of toml file.
        input_filename = Path(filename.parent / (filename.stem.split(".")[0] + ".in"))
        with open(input_filename, "r") as f:
            input_str = f.read()

        gk_input = GKInputGX()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        return {"out": raw_data_out, "big": raw_data_big}, gk_input, input_str

    @staticmethod
    def _get_time_indices(
        raw_data: Dict[str, nc.Dataset], big_downsize_threshold: float = 4.0
    ) -> np.ndarray:
        """
        Determines whether to use the time grid corresponding to raw_data_out
        (saved with cadence 'nwrite') or raw_data_big ('nwrite_big'), and gives
        back the set of indices to be used for slicing of time data.
        """

        # Determine approximate sampling ratio. The start and end points need to
        # be excluded in this case.
        # TODO should use nwrite and nwrite_big from the output file,
        # but the latter is not currently stored.
        time_out = raw_data["out"]["Grids"]["time"][:].data
        time_big = raw_data["big"]["Grids"]["time"][:].data

        if len(time_big) > 2:
            time_len_ratio = (len(time_out) - 2) / (len(time_big) - 2)
        else:
            time_len_ratio = big_downsize_threshold + 1

        # Set time indices
        if time_len_ratio < 1.0:
            raise NotImplementedError(
                "Loading GX output data with nwrite_big < nwrite is currently not supported."
            )
        elif time_len_ratio < big_downsize_threshold:
            time_indices = [np.argmin(np.abs(time_out - val)) for val in time_big]
        else:
            time_indices = np.arange(len(time_out), dtype=int)

        # TODO currently ignores logic above to force using smaller time array
        time_indices = [np.argmin(np.abs(time_out - val)) for val in time_big]
        return time_indices

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, nc.Dataset],
        gk_input: GKInputGX,
        downsize: int,
        time_indices: np.ndarray,
    ) -> Dict[str, Any]:

        # Spatial coordinates. Note that the kx grid already has kx=0 in the middle of the array
        ky = raw_data["out"]["Grids"]["ky"][:].data
        kx = raw_data["out"]["Grids"]["kx"][:].data
        raw_theta = np.float64(raw_data["out"]["Grids"]["theta"][:].data)

        # Add final point so easier to fit
        raw_theta = np.append(raw_theta, -raw_theta[0])

        local_geometry = gk_input.get_local_geometry()
        geometric_theta = np.linspace(
            np.min(raw_theta), np.max(raw_theta), len(raw_theta) * 4
        )
        metric_terms = MetricTerms(local_geometry, theta=geometric_theta)

        # Parallel gradient
        g_tt = metric_terms.field_aligned_covariant_metric("theta", "theta")
        grho = np.sqrt(g_tt).m

        nperiod = gk_input.data["Dimensions"]["nperiod"]
        theta_range = 2 * np.pi * (2 * nperiod - 1)

        equal_arc_theta = cumulative_trapezoid(grho, geometric_theta, initial=0.0)
        equal_arc_theta *= 1.0 / equal_arc_theta[-1] * theta_range
        equal_arc_theta += -theta_range / 2

        theta = np.interp(raw_theta[:-1], equal_arc_theta, geometric_theta)

        # Time coordinates
        # TODO handle different time arrays
        time = raw_data["out"]["Grids"]["time"][time_indices].data
        # time = raw_data["big"]["Grids"]["time"][time_indices].data

        # Energy coords
        energy = np.arange(0, raw_data["out"]["nlaguerre"][:].data, 1, dtype=int)
        pitch = np.arange(0, raw_data["out"]["nhermite"][:].data, 1, dtype=int)

        # Moment coords
        fluxes = [
            "particle",
            "heat",
        ]  # GX does not currently have a momentum flux diagnostic
        moments = ["density", "temperature", "velocity"]

        # Field coords
        field_vals = {}
        beta = gk_input.data["Physics"]["beta"]
        defaults = {
            "phi": 1.0,
            "apar": 1.0 if beta > 0.0 else 0.0,
            "bpar": 1.0 if beta > 0.0 else 0.0,
        }
        for field, default in defaults.items():
            try:
                field_vals[field] = gk_input.data["Physics"][f"f{field}"]
            except KeyError:
                field_vals[field] = default
        fields = [field for field, val in field_vals.items() if val > 0]

        # Species coords
        species = []
        ion_num = 0
        for i_sp in range(gk_input.data["Dimensions"]["nspecies"]):
            if gk_input.data["species"]["z"][i_sp] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        if raw_data["out"]["nspecies"][:].data != len(species):
            raise RuntimeError(
                "GKOutputReaderGX: Different number of species in input and output."
            )

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "linear": gk_input.is_linear(),
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "downsize": downsize,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, nc.Dataset],
        time_indices: np.ndarray,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        To print fields, GX requires us to set 'fields = true' in the '[Diagnostics]'
        input file group. However, it will print all of the fields even if a given
        field multiplier is set to zero, or the simulation is electrostatic.
        We either need to interpolate the field data onto the 'out' time grid,
        or use the 'big' time grid (if the data has been downsampled).
        """

        # Need to make sure we only try and load the fields that pyro is expecting
        field_names = coords["field"]
        results = {}

        # Check if the fields have been saved
        if not raw_data["out"]["Inputs"]["Diagnostics"]["fields"][:].data:
            return results

        # Loop through all fields and add the field
        for field_name in field_names:

            # The raw field data has coordinates (time, ky, kx, theta, ri)
            field = raw_data["big"]["Diagnostics"][f"{field_name.capitalize()}"][:].data

            # # Interpolate onto 'out' time grid if needed
            # if len(time_indices) > field.shape[0]:

            #     time_out = raw_data["out"]["Grids"]["time"][:].data
            #     time_big = raw_data["big"]["Grids"]["time"][:].data

            #     indices = np.searchsorted(time_out, time_big)

            #     field_interp = np.zeros((len(time_out),) + field.shape[1:])
            #     field_interp[indices, :, :, :, :] = field

            #     field = field_interp

            #     # TODO unsure of the normalisation conventions in pyro, so have not
            #     # converted normalisations. GX normalises
            #     # to (rhostar * Tref)/qref, rhostar * (rhoref B_N), and rhostar * B_N

            # Transpose the data to have shape (ri, theta, kx, ky time)
            field = field.transpose()

            # Compose data to remove ri axis.
            field = field[0, ...] + 1j * field[1, ...]

            if field_name == "bpar":
                bmag = raw_data["out"]["Geometry"]["bmag"][:].data[
                    :, np.newaxis, np.newaxis, np.newaxis
                ]
                field *= bmag

            # Store field data
            field_name = field_name[:1].lower() + field_name[1:]
            results[field_name] = field

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, nc.Dataset],
        gk_input: GKInputGX,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """

        # TODO Import the moments that can be found in '.big.nc'.

        raise NotImplementedError

    @staticmethod
    def _get_fluxes(
        raw_data: Dict[str, nc.Dataset],
        gk_input: GKInputGX,
        coords: Dict,
        time_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        To print fluxes, GX requires us to set 'fluxes = true' in the '[Diagnostics]'
        input file group. In this case, it will output fluxes for all fields, irrespective
        of the values of fphi, fapar and fbpar.
        """
        # Setting names of variables in output file
        fields = {"phi": "ES", "apar": "Apar", "bpar": "Bpar"}
        fluxes_dict = {"particle": "ParticleFlux", "heat": "HeatFlux"}

        results = {}

        coord_names = ["flux", "field", "species", "kx", "ky", "time"]
        fluxes = np.zeros([len(coords[name]) for name in coord_names])
        fields = {
            field: value for field, value in fields.items() if field in coords["field"]
        }

        # Check if fluxes have been saved.
        if not raw_data["out"]["Inputs"]["Diagnostics"]["fluxes"][:].data:
            return results

        # Iterate over all possible fields and fluxes
        for (ifield, (field, gx_field)), (iflux, gx_flux) in product(
            enumerate(fields.items()), enumerate(fluxes_dict.values())
        ):
            flux_key = f"{gx_flux}{gx_field}_kxkyst"

            if flux_key in raw_data["out"]["Diagnostics"].variables:

                # Raw data has coordinates (time, species, ky)
                flux = raw_data["out"]["Diagnostics"][flux_key][:].data

                # Apply correct time slicing
                flux = flux[time_indices, ...]

                # Transpose to (species, kx, ky, time)
                flux = flux.transpose(1, 3, 2, 0)
            else:
                continue

            fluxes[iflux, ifield, ...] = flux.data

        if gk_input.is_linear():
            jacob = raw_data["out"]["Geometry"]["jacobian"][:].data
            grho = raw_data["out"]["Geometry"]["grho"][:].data
            theta = raw_data["out"]["Grids"]["theta"][:].data

            flux_norm = trapezoid(jacob, theta) / trapezoid(jacob * grho, theta)
        else:
            flux_norm = 1.0

        for iflux, flux in enumerate(coords["flux"]):
            if not np.all(fluxes[iflux, ...] == 0):
                results[flux] = fluxes[iflux, ...] / flux_norm

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: Dict[str, nc.Dataset], time_indices: float
    ) -> Dict[str, np.ndarray]:

        results = None

        # Check whether eigenvalue data has been saved
        if not raw_data["out"]["Inputs"]["Diagnostics"]["omega"][:].data:
            return results

        # Import eigenvalue data
        omegas = raw_data["out"]["Diagnostics"]["omega_kxkyt"][time_indices, ...].data

        mode_frequency = omegas[..., 0].transpose()
        growth_rate = omegas[..., 1].transpose()

        return {
            "mode_frequency": mode_frequency,
            "growth_rate": growth_rate,
        }

    @staticmethod
    def _get_eigenfunctions(
        raw_data: xr.Dataset,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:

        # TODO in a sense, this function is redundant, given that it relies on field data
        # However, it gives the option to accesss the eigenfunctions even when the '.big'
        # time series is very sparse.

        coord_names = ["field", "theta", "kx", "ky"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        # Check whether fields have been saved
        if not raw_data["out"]["Inputs"]["Diagnostics"]["fields"][:].data:
            return eigenfunctions

        # Import raw eigenfunction data from fields at the final time
        raw_eig_data = [
            raw_data["big"]["Diagnostics"][f"{field.capitalize()}"][-1, ...]
            for field in coords["field"]
        ]

        # Check whether any of the field data is non-trivial. It will be trivial if the user has
        # set 'load_fields = True' but has not provided the .big.nc file. In this case, it is
        # impossible to load any field/eigenfunction data
        if not np.any(raw_eig_data) > 0.0:
            return eigenfunctions

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                # Raw data has dimensions (ky, kx, theta, ri). We
                # want to output (ri, theta, kx, ky)
                eigenfunction = raw_eigenfunction.transpose()

                eigenfunctions[ifield, ...] = (
                    eigenfunction[0, ...].data + 1j * eigenfunction[1, ...].data
                )

        square_fields = np.sum(np.abs(eigenfunctions) ** 2, axis=0)
        field_amplitude = np.sqrt(
            trapezoid(square_fields, coords["theta"], axis=0) / (2 * np.pi)
        )

        field_amplitude = np.where(field_amplitude == 0, 1.0, field_amplitude)

        # FIXME I have simply copied the code from the GS2 file, as the structure
        # should be the same. However, this just gives nan's for the cases that I have
        # tried, so unsure whether this is correct?
        first_field = eigenfunctions[0, ...]
        theta_star_index = np.argmax(abs(first_field), axis=0)
        field_theta_star = np.take_along_axis(
            first_field, theta_star_index[np.newaxis, ...], axis=0
        ).squeeze(0)

        phase = np.exp(-1j * np.angle(field_theta_star))

        phase = np.nan_to_num(phase, nan=0.0)

        result = eigenfunctions * phase / field_amplitude

        return result
