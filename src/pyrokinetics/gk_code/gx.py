import warnings
from copy import copy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from cleverdict import CleverDict

from ..constants import pi, sqrt2
from ..local_geometry import LocalGeometry, LocalGeometryMiller, default_miller_inputs
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict, ureg
from ..numerics import Numerics
from ..file_utils import AbstractFileReader
from ..templates import gk_templates
from ..typing import PathLike
from .gk_input import GKInput
from .gk_output import Coords, Eigenvalues, Fields, Fluxes, GKOutput, Moments


@GKInput.reader("GX")
class GKInputGX(GKInput):
    """
    Class that can read GX input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GX"
    default_file_name = "input.in"
    norm_convention = "gx"

    # gx does not calculate geometric information except for the simplest models
    # Preferred method is to read geometric coefficients from a NetCDF file
    # Secondary method is to read geometric coefficients from the GS2 eik.out file
    pyro_gx_miller = {
        "rho": ["Geometry", "rhoc"],
        "Rmaj": ["Geometry", "rmaj"],
        "q": ["Geometry", "qinp"],
        "kappa": ["Geometry", "akappa"],
        "shat": ["Geometry", "shat"],
        "shift": ["Geometry", "shift"],
        "beta_prime": ["Geometry", "beta_prime_input"],
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
        "inverse_lv": "uprim",
        "type": "type",
    }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GX input file into a dictionary
        """
        result = super().read_from_file(filename)
        return result

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GX input file given as string
        Uses default read_str, which assumes input_string has valid toml syntax
        """
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
        # but they are needed by Pyrokinetics
        expected_keys = [
            "Domain",
            "Dimensions",
            "Time",
            "Initialization",
            "Restart",
            "Dissipation",
            "Diagnostics",
            "Boltzmann",
            "Geometry",
            "Physics",
            "species",
        ]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as GX file")

    def write(self, filename: PathLike, float_format: str = "", local_norm=None):
        if local_norm is None:
            local_norm = Normalisation("write")

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, local_norm.gx)

        super().write(filename, float_format=float_format)

    def is_nonlinear(self) -> bool:
        try:
            is_nonlinear = self.data["Control"]["nonlinear_mode"] == True
            return is_nonlinear
        except KeyError:
            return False

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GX input file
        """
        super().add_flags(flags)

    def get_local_geometry(self) -> LocalGeometry:
        """
        Should return local geometry by delegating to more specific functions, but 
        instead, needs a file with pre-processed geometric data or a VMEC output file
        GX provides Python code to perform the pre-processing step
        """
        gx_eq = self.data["Geometry"]["geo_option"]

        if gx_eq not in ["none", "slab", "const-curv", "s-alpha"]:
            raise NotImplementedError(
                f"GX equilibrium option {gx_eq} not implemented"
            )

        local_eq = self.data["Geometry"].get("local_eq", False)
        if not local_eq:
            raise RuntimeError("GX is not using local equilibrium")

        geotype = self.data["Geometry"].get("geo_option", "s-alpha")
        if geotype != "s-alpha":
            raise NotImplementedError("GX option is not implemented")

        return self.get_local_geometry_miller()

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Basic Miller object from GS2 file
        """
        # We require the use of Bishop mode 4, which uses a numerical equilibrium,
        # s_hat_input, and beta_prime_input to determine metric coefficients.
        # We also require 'irho' to be 2, which means rho corresponds to the ratio of
        # the midplane diameter to the Last Closed Flux Surface (LCFS) diameter
        if self.data["Geometry"]["bishop"] != 4:
            raise RuntimeError(
                "Pyrokinetics requires GX input files to use "
                "Geometry.bishop = 4"
            )
        if self.data["Geometry"]["irho"] != 2:
            raise RuntimeError(
                "Pyrokinetics requires GX input files to use "
                "Geometry.irho = 2"
            )

        miller_data = default_miller_inputs()

        for (pyro_key, (gx_param, gx_key)), gx_default in zip(
            self.pyro_gx_miller.items(), self.pyro_gx_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data[gx_param].get(gx_key, gx_default)

        rho = miller_data["rho"]
        kappa = miller_data["kappa"]
        miller_data["delta"] = np.sin(
            self.data["Geometry"].get("tri", 0.0)
        )
        miller_data["s_kappa"] = (
            self.data["Geometry"].get("akappri", 0.0) * rho / kappa
        )
        miller_data["s_delta"] = (
            self.data["Geometry"].get("tripri", 0.0) * rho
        )

        # Get beta and beta_prime normalised to R_major(in case R_geo != R_major)
        r_geo = self.data["Geometry"].get("R_geo", miller_data["Rmaj"])

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = (
            self.data["Physics"]["beta"]
            * (miller_data["Rmaj"] / r_geo) ** 2
            * ne_norm
            * Te_norm
        )
        miller_data["beta_prime"] *= (miller_data["Rmaj"] / r_geo) ** 2

        # Assume pref*8pi*1e-7 = 1.0
        miller_data["B0"] = np.sqrt(1.0 / beta) if beta != 0.0 else None

        miller_data["ip_ccw"] = 1  # TBD
        miller_data["bt_ccw"] = 1  # TBD
        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        # Perhaps this can be aimed at the Python scripts we use for GX?
        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from GX file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        ne_norm, Te_norm = self.get_ne_te_normalisation()

        domega_drho = (
            self.data["Geometry"]["qinp"]
            / self.data["Geometry"]["rhoc"]
            * self.data["Physics"].get("g_exb", 0.0)
        )

        # Load each species into a dictionary
        for i_sp in range(self.data["Dimensions"]["nspecies"]):
            species_data = CleverDict()

            gx_key = f"species_parameters_{i_sp + 1}"

            gx_data = self.data[gx_key]

            for pyro_key, gx_key in self.pyro_gx_species.items():
                species_data[pyro_key] = gs2_data[gx_key]

            species_data.vel = 0.0 * ureg.vref_nrl
            species_data.inverse_lv = 0.0 / ureg.lref_minor_radius
            species_data.domega_drho = (
                domega_drho * ureg.vref_nrl / ureg.lref_minor_radius**2
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
            species_data.nu *= ureg.vref_nrl / ureg.lref_minor_radius
            species_data.temp *= ureg.tref_electron / Te_norm
            species_data.z *= ureg.elementary_charge
            species_data.inverse_lt *= ureg.lref_minor_radius**-1
            species_data.inverse_ln *= ureg.lref_minor_radius**-1

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.normalise()

        # Z_eff is not an input parameter for GX
        # if "zeff" in self.data["knobs"]:
        #    local_species.zeff = self.data["knobs"]["zeff"] * ureg.elementary_charge
        # elif "zeff" in self.data["parameters"]:
        #     local_species.zeff = (
        #         self.data["parameters"]["zeff"] * ureg.elementary_charge
        #     )
        # else:
        # I suppose this is the right thing to do? 
        local_species.zeff = 1.0 * ureg.elementary_charge

        return local_species

    def _read_box_grid(self):
        box = self.data["Domain"]
        keys = box.keys()

        grid_data = {}

        # Set up ky grid
        if "ny" in keys:
            grid_data["nky"] = int((box["ny"] - 1) / 3 + 1)
        elif "nky" in keys:
            grid_data["nky"] = box["nky"]
        else:
            raise RuntimeError(f"ky grid details not found in {keys}")

        if "y0" in keys:
            grid_data["ky"] = 1 / box["y0"]
        else:
            raise RuntimeError(f"Min ky details not found in {keys}")

        if "nx" in keys:
            grid_data["nkx"] = int(2 * (box["nx"] - 1) / 3 + 1)
        elif "nkx" in keys:
            grid_data["nkx"] = box["nkx"]
        else:
            raise RuntimeError("kx grid details not found in {keys}")

        if box["jtwist"] == 0 and box["boundary"] == "periodic":
            raise RuntimeError("jtwist = 0 is not permitted with periodic boundary conditions")

        if "jtwist" not in keys:
            if "x0" in keys:
                jtwist_default = min(round(2 * pi * abs(shat) * box["x0"]/box["y0"]), 1) 
            elif            
                jtwist_default = min(round(2 * pi * abs(shat)), 1)
            
        shat_params = self.pyro_gx_miller["shat"]
        shat = self.data[shat_params[0]][shat_params[1]]
        if abs(shat) > self.data["Domain"]["zero_shat_threshold"]:
            jtwist = box.get("jtwist", jtwist_default)
            grid_data["kx"] = grid_data["ky"] * abs(shat) * 2 * pi / jtwist
        elif
            set_periodic = "periodic"

        if ((set_periodic == "periodic") or (self.data["Domain"]["boundary"] == "periodic")) :
            if "x0" not in keys:
                grid_data["kx"] = grid_data["ky"]
            elif
                grid_data["kx"] = 1 / box["x0"]

        return grid_data

    def _read_grid(self):
        """Read the perpendicular wavenumber grid"""

        reader = self._read_box_grid
        return reader()

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = self.data["Physics"].get("fphi", 0.0) > 0.0
        numerics_data["apar"] = self.data["Physics"].get("fapar", 0.0) > 0.0
        numerics_data["bpar"] = self.data["Physics"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        delta_time = self.data["Time"].get("dt", 0.05)
        numerics_data["delta_time"] = delta_time
        numerics_data["max_time"] = self.data["Time"].get("t_max")

        numerics_data["nonlinear"] = self.is_nonlinear()

        numerics_data.update(self._read_grid())

        # Theta grid
        numerics_data["ntheta"] = self.data["Dimensions"]["ntheta"]
        numerics_data["nperiod"] = self.data["Dimensions"]["nperiod"]

        # Velocity information
        numerics_data["nhermite"] = self.data["Dimensions"]["nhermite"]
        numerics_data["nlaguerre"] = self.data["Dimensions"]["nlaguerre"]

        Rmaj = self.data["Geometry"]["Rmaj"]
        r_geo = self.data["Geometry"].get("R_geo", Rmaj)

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data["Physics"]["beta"] * (Rmaj / r_geo) ** 2 * ne_norm * Te_norm
        numerics_data["beta"] = beta * ureg.beta_ref_ee_B0

        numerics_data["gamma_exb"] = (
            self.data["Physics"].get("g_exb", 0.0)
            * ureg.vref_nrl / ureg.lref_minor_radius
        )

        return Numerics(**numerics_data)

    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        local_norm: Normalisation = None,
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
                template_file = gk_templates["GX"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        # Set Miller Geometry bits
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GX not supported yet"
            )

        # Ensure Miller settings
        self.data["Geometry"]["geo_option"] = "eik"
        self.data["Geometry"]["geo_file"] = "eik.out"
        self.data["Geometry"]["iflux"] = 0
        self.data["Geometry"]["local_eq"] = True
        self.data["Geometry"]["bishop"] = 4
        self.data["Geometry"]["irho"] = 2
        self.data["theta_grid_parameters"]["geoType"] = 0

        # Assign Miller values to input file
        for key, val in self.pyro_gx_miller.items():
            self.data[val[0]][val[1]] = local_geometry[key]

        self.data["Geometry"]["akappri"] = (
            local_geometry.s_kappa * local_geometry.kappa / local_geometry.rho
        )
        self.data["Geometry"]["tri"] = np.arcsin(local_geometry.delta)
        self.data["Geometry"]["tripri"] = local_geometry["s_delta"] / local_geometry.rho
        self.data["Geometry"]["R_geo"] = local_geometry.Rmaj

        # Set local species bits
        self.data["Dimensions"]["nspecies"] = local_species.nspec

        for iSp, name in enumerate(local_species.names):   # TBD: we have one species array
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
                self.data[species_key][val] = local_species[name][key].to(
                    local_norm.gs2
                )

        beta_ref = local_norm.gx.beta if local_norm else 0.0
        self.data["Physics"]["beta"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        # Set numerics bits
        # Set no. of fields
        self.data["Physics"]["fphi"] = 1.0 if numerics.phi else 0.0
        self.data["Physics"]["fapar"] = 1.0 if numerics.apar else 0.0
        self.data["Physics"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        # Set time stepping
        self.data["Time"]["dt"] = numerics.delta_time
        self.data["Time"]["nstep"] = int(numerics.max_time / numerics.delta_time)

        if numerics.nky == 1:
            self.data["Domain"]["y0"] = 1 / numerics.ky
            self.data["Dimensions"]["nperiod"] = numerics.nperiod
        else:
            self.data["Dimensions"]["nx"] = int(
                ((numerics.nkx - 1) * 3 / 2) + 1
            )
            self.data["Dimensions"]["ny"] = int(
                ((numerics.nky - 1) * 3) + 1
            )

            self.data["Dimensions"]["y0"] = 1 / numerics.ky

            # Currently forces NL sims to have nperiod = 1
            self.data["Dimensions"]["nperiod"] = 1

            periodic = ( (abs(shat) < self.data["Domain"]["zero_shat_threshold"])
                            or (self.data["Domain"]["boundary"] == "periodic")
                        )
            
            shat = local_geometry.shat
            if periodic:
                self.data["Domain"]["x0"] = (
                    2 * pi / numerics.kx
                )
            else:
                self.data["Domain"]["jtwist"] = int(
                    (numerics.ky * shat * 2 * pi / numerics.kx) + 0.1
                )

        self.data["Dimensions"]["ntheta"] = numerics.ntheta

        self.data["Dimensions"]["nhermite"] = numerics.nhermite
        self.data["Dimensions"]["nlaguerre"] = numerics.nlaguerre

        self.data["Physics"]["g_exb"] = numerics.gamma_exb

        if not local_norm:
            return

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, local_norm.gx)

    def get_ne_te_normalisation(self):
        found_electron = False
        # Load each species into a dictionary
        for i_sp in range(self.data["Dimensions"]["nspecies"]):
            gx_key = f"species[{i_sp}]"
            if (
                self.data[gx_key]["z"] == -1
                or self.data[gx_key]["type"] == "electron"
            ):
                ne = self.data[gx_key]["dens"]
                Te = self.data[gx_key]["temp"]
                found_electron = True
                break

        if not found_electron:
            raise TypeError(
                "Pyro currently only supports electron species with charge = -1"
            )

        return ne, Te


@GKOutput.reader("GX")
class GKOutputReaderGX(AbstractFileReader):
    def read_from_file(
        self,
        filename: PathLike,
        norm: Normalisation,
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

        if fields or coords["linear"]:
            # Rely on gk_output to generate eigenvalues
            eigenvalues = None
        else:
            eigenvalues = self._get_eigenvalues(raw_data, coords["time_divisor"])

        # Assign units and return GKOutput
        convention = norm.gx
        field_dims = ("z", "kx", "ky", "time")
        flux_dims = ("field", "species", "ky", "time")
        moment_dims = ("field", "species", "ky", "time")
        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["theta"],
                species=coords["species"],
                field=coords["field"],
            ).with_units(convention),
            norm=norm,
            fields=Fields(**fields, dims=field_dims).with_units(convention)
            if fields
            else None,
            fluxes=Fluxes(**fluxes, dims=flux_dims).with_units(convention)
            if fluxes
            else None,
            moments=Moments(**moments, dims=moment_dims).with_units(convention)
            if moments
            else None,
            eigenvalues=Eigenvalues(**eigenvalues).with_units(convention)
            if eigenvalues
            else None,
            linear=coords["linear"],
            gk_code="GX",
            input_file=input_str,
            normalise_flux_moment=True,
        )

    def verify_file_type(self, filename: PathLike):
        try:
            warnings.filterwarnings("error")
            data = xr.open_dataset(filename)
        except RuntimeWarning:
            warnings.resetwarnings()
            raise RuntimeError
        warnings.resetwarnings()

        if "software_name" in data.attrs:
            if data.attrs["software_name"] != "GX":
                raise RuntimeError
        elif "gx_help" in data.attrs.keys():
            pass
        else:
            raise RuntimeError

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Gets path by removing ".in" and replacing it with ".out.nc"
        """
        filename = Path(filename)
        return filename.parent / (filename.stem + ".out.nc")

    @staticmethod
    def _get_raw_data(filename: PathLike) -> Tuple[xr.Dataset, GKInputGX, str]:
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
            input_str = "\n".join((line.decode("utf-8") for line in input_file.data))
        gk_input = GKInputGX()
        gk_input.read_str(input_str)
        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: xr.Dataset, gk_input: GKInputGX, downsize: int
    ) -> Dict[str, Any]:
        # ky coords
        ky = raw_data["ky"].data

        time = raw_data["time"].data

        # kx coords
        kx = raw_data["kx"].data

        # theta coords
        theta = raw_data["theta"].data

        # moment coords
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        # field coords
        # If fphi/fapar/fbpar not in 'Physics', or they equal zero, skip the field
        field_vals = {}
        for field, default in zip(["phi", "apar", "bpar"], [1.0, 0.0, 0.0]):
            try:
                field_vals[field] = gk_input.data["Physics"][f"f{field}"]
            except KeyError:
                field_vals[field] = default

        # species coords
        # TODO is there some way to get this info without looking at the input data?
        species = []
        ion_num = 0
        for idx in range(gk_input.data["Dimensions"]["nspecies"]):
            if gk_input.data[f"species[{idx}]"]["z"] == -1:
                species.append("electron")
            else:
                ion_num += 1
                species.append(f"ion{ion_num}")

        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "linear": gk_input.is_linear(),
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
        input file under 'Physics'. We must also instruct GX to print fields in 'Diagnostics' using:
        - fields = .true.
        """
        field_names = ("phi", "apar", "bpar")
        results = {}

        # Loop through all fields and add field if it exists
        for field_name in field_names:
            key = f"{field_name}_t"
            if key not in raw_data:
                continue

            # raw_field has coords (t,ky,kx,theta,real/imag).   # What order do we use in GX? 
            # We wish to transpose that to (real/imag,theta,kx,ky,t)
            field = raw_data[key].transpose("ri", "theta", "kx", "ky", "t").data
            field = field[0, ...] + 1j * field[1, ...]

            # Adjust fields to account for differences in defintions/normalisations
            # Need to check definition of bpar in GX
            if field_name == "bpar":
                bmag = raw_data["bmag"].data[:, np.newaxis, np.newaxis, np.newaxis]
                field *= bmag

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
        For GX to print fluxes, we must have fphi, fapar and/or fbpar set to 1.0 in the
        input file under 'Physics'. We must also set the following in
        'Diagnostics':
        - fluxes = true
        """
        # field names change from ["phi", "apar", "bpar"] to ["es", "apar", "bpar"]
        # Take whichever fields are present in data, relabelling "phi" to "es"
        fields = {"phi": "es", "apar": "apar", "bpar": "bpar"}
        fluxes_dict = {"particle": "part", "heat": "heat", "momentum": "mom"}

        # Get species names from input file
        species = []
        ion_num = 0
        for idx in range(gk_input.data["Dimensions"]["nspecies"]):
            if gk_input.data[f"species[{idx}]"]["z"] == -1:
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

        for (ifield, (field, gx_field)), (iflux, gx_flux) in product(
            enumerate(fields.items()), enumerate(fluxes_dict.values())
        ):
            flux_key = f"{gx_field}_{gx_flux}_flux"
            by_k_key = f"{gx_field}_{gs2_flux}_by_k"
            # new diagnostics
            # by_mode_key = f"{gx_field}_{gx_flux}_flux_by_mode"  # what is this?

            if by_k_key in raw_data.data_vars or by_mode_key in raw_data.data_vars:
                key = by_mode_key if by_mode_key in raw_data.data_vars else by_k_key
                flux = raw_data[key].transpose("species", "kx", "ky", "t")
                # Sum over kx
                flux = flux.sum(dim="kx")
                # Divide non-zonal components by 2 due to reality condition
                # flux[:, 1:, :] *= 0.5   # This kind of thing is already handled in GX
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
        mode_frequency = raw_data.omega_kxkyt.isel(ri=0).transpose("kx", "ky", "time")
        growth_rate = raw_data.omega_kxkyt.isel(ri=1).transpose("kx", "ky", "time")
        return {
            "mode_frequency": mode_frequency.data,
            "growth_rate": growth_rate.data,
        }
