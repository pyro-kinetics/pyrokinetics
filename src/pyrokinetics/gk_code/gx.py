from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import toml
from cleverdict import CleverDict
from scipy.integrate import trapezoid
from sympy import integer_log

from ..constants import deuterium_mass, electron_mass, pi
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
                while f(xmax) > xmax: xmax <<= 1
                while xmax-xmin > 1:
                    xmid = xmax+xmin>>1
                    if f(xmid) <= xmid:
                        xmax = xmid
                    else:
                        xmin = xmid
                return xmax
            
            def f(x): return j+x-sum((x//3**i).bit_length() for i in range(integer_log(x, 3)[0]+1))

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
        # TODO possibly remove, as GX does not output the input file in this way

        result = super().read_str(input_string)

        return result

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads GX input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

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

        local_species.zeff = 1.0 * convention.qref

        return local_species

    def _read_grid(self, drho_dpsi):
        domain = self.data["Domain"]
        dimensions = self.data["Dimensions"]

        grid_data = {}

        # Set up ky grid
        if "ny" in dimensions.keys():
            grid_data["nky"] = int((dimensions["ny"] - 1) / 3 + 1)
        elif "nky" in dimensions.keys():
            grid_data["nky"] = dimensions["nky"]
        else:
            raise RuntimeError(f"ky grid details not found in {dimensions.keys()}")

        if "y0" in domain.keys():
            grid_data["ky"] = (1 / domain["y0"]) * np.linspace(
                0, grid_data["nky"] - 1, grid_data["nky"]
            )
        else:
            raise RuntimeError(f"Min ky details not found in {domain.keys()}")

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

        grid_data["kx"] = np.linspace(
            -(1 / x0) * (grid_data["nkx"] // 2),
            (1 / x0) * (grid_data["nkx"] // 2),
            grid_data["nkx"],
        )

        return grid_data

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = self.data["Physics"].get("fphi", 1.0) > 0.0
        numerics_data["apar"] = self.data["Physics"].get("fapar", 0.0) > 0.0
        numerics_data["bpar"] = self.data["Physics"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        numerics_data["delta_time"] = self.data["Time"].get("dt", 0.05)
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

        # Theta grid
        numerics_data["ntheta"] = self.data["Dimensions"]["ntheta"]
        numerics_data["nperiod"] = self.data["Dimensions"]["nperiod"]

        # Velocity grid. Note that GX is not in energy and pitch angle coordinates
        numerics_data["nenergy"] = self.data["Dimensions"]["nlaguerre"]
        numerics_data["npitch"] = self.data["Dimensions"]["nhermite"]

        numerics_data["beta"] = self._get_beta()
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

        beta_ref = convention.beta if local_norm else 0.0
        self.data["Physics"]["beta"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        # Set numerics bits
        # Set no. of fields
        self.data["Physics"]["fphi"] = 1.0 if numerics.phi else 0.0
        self.data["Physics"]["fapar"] = 1.0 if numerics.apar else 0.0
        self.data["Physics"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        if numerics.gamma_exb > 0.0:
            self.data["Physics"]["g_exb"] = numerics.gamma_exb

        # Set time stepping
        self.data["Time"]["dt"] = numerics.delta_time
        self.data["Time"]["t_max"] = numerics.max_time

        if self.data["Physics"]["nonlinear_mode"] == False:
            self.data["Dimensions"]["nky"] = numerics.nky
            self.data["Dimensions"]["nkx"] = numerics.nkx
        else:
            ny = int(3 * ((numerics.nky - 1)) + 1)
            nx = int(3 * ((numerics.nkx - 1) / 2) + 1)

            # Uncomment the following if you want pyro to round the (nonlinear) grid to the nearest 
            # larger three smooth number (optimal for GPU-based FFTs). This will always ensure that 
            # the grid is larger than the original grid, and no information is lost.

            # ns = [nx, ny]
            # three_smooth_numbers = self._generate_three_smooth_numbers(max(ns))
            # ns = [int(min([x for x in three_smooth_numbers if (abs(x - n) <= 2) and (x > n)], default=n)) for n in ns]
            # nx, ny = ns

            self.data["Dimensions"]["ny"] = ny
            self.data["Dimensions"]["nx"] = nx

        self.data["Domain"]["y0"] = 1.0 / (
            numerics.ky[1] * (1 * convention.bref / local_norm.gx.bref).to_base_units()
        )

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
                else Eigenfunctions(
                    eigenfunctions, dims=("field", "theta", "kx", "ky")
                ).with_units(convention)
            ),
            linear=coords["linear"],
            gk_code="GX",
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
            raise RuntimeError("Error occurred reading GX output file")
        warnings.resetwarnings()

        if "software_name" in data.attrs:
            if data.attrs["software_name"] != "GX":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'software_name' for a GX file"
                )
        elif "code_info" in data.data_vars:
            if data["code_info"].long_name != "GX":
                raise RuntimeError(
                    f"file '{filename}' has wrong 'code_info' for a GX file"
                )
        elif "gs2_help" in data.attrs.keys():
            pass
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
    def _get_raw_data(filename: PathLike) -> Tuple[xr.Dataset, GKInputGX, str]:
        import xarray as xr

        raw_data = xr.open_dataset(filename)
        # Read input file from netcdf, store as GKInputGX
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
        gk_input = GKInputGX()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: xr.Dataset, gk_input: GKInputGX, downsize: int
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

        # species coords - Note this assumes the gs2 charge normalisation
        # is the proton charge. We could instead use the "type_of_species"
        # property instead, but would then need to maintain the mapping
        # from this integer to the actual species type (unlikely to change).
        species = []
        ion_num = 0
        for z in raw_data["charge"].data:
            if np.isclose(z, -1):
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
        For GX to print fields, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also instruct GX to print each field
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
        gk_input: GKInputGX,
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
        gk_input: GKInputGX,
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        For GX to print fluxes, we must have fphi, fapar and fbpar set to 1.0 in the
        input file under 'knobs'. We must also set the following in
        gs2_diagnostics_knobs:
        - write_fluxes = .true. (default if nonlinear)
        - write_fluxes_by_mode = .true. (default if nonlinear)
        """
        # field names change from ["phi", "apar", "bpar"] to ["es", "apar", "bpar"]
        # Take whichever fields are present in data, relabelling "phi" to "es"
        fields = {"phi": "es", "apar": "apar", "bpar": "bpar"}
        fluxes_dict = {"particle": "part", "heat": "heat", "momentum": "mom"}

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

            fluxes[iflux, ifield, ...] = flux.data

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

        coord_names = ["field", "theta", "kx", "ky"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                eigenfunction = raw_eigenfunction.transpose("ri", "theta", "kx", "ky")

                eigenfunctions[ifield, ...] = (
                    eigenfunction[0, ...].data + 1j * eigenfunction[1, ...].data
                )

        square_fields = np.sum(np.abs(eigenfunctions) ** 2, axis=0)
        field_amplitude = np.sqrt(
            trapezoid(square_fields, coords["theta"], axis=0) / (2 * np.pi)
        )

        first_field = eigenfunctions[0, ...]
        theta_star = np.argmax(abs(first_field), axis=0)
        field_theta_star = first_field[theta_star, 0, 0]
        phase = np.abs(field_theta_star) / field_theta_star

        result = eigenfunctions * phase / field_amplitude

        return result
