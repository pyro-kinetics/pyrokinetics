import numpy as np
from path import Path
from cleverdict import CleverDict
from typing import Optional

from ..typing import PathLike
from ..constants import electron_charge, pi, sqrt2
from ..local_species import LocalSpecies
from ..local_geometry import LocalGeometry, LocalGeometryMiller
from ..numerics import Numerics
from ..templates import template_dir
from .GKInputReader import GKInputReader


class GKInputReaderGS2(GKInputReader):
    """
    Class that can read GS2 input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects

    """

    pyro_gs2_miller = {
        "rho": ["theta_grid_parameters", "rhoc"],
        "Rmaj": ["theta_grid_parameters", "rmaj"],
        "q": ["theta_grid_parameters", "qinp"],
        "kappa": ["theta_grid_parameters", "akappa"],
        "shat": ["theta_grid_eik_knobs", "shat_input"],
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

    base_template_file = template_dir / "input.gs2"
    
    code_name = "GS2"


    def read(self, filename: PathLike):
        """
        Reads GS2 input file into a dictionary
        """
        super().read(filename)
        if self.is_nonlinear() and "wstar_units" in self.data["knobs"]:
            self.data["knobs"].pop("wstar_units")

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid gs2 input file"
        """
        data = f90nml.read(filename).todict()
        expected_keys = ["kt_grids_knobs", "theta_grid_knobs", "theta_grid_eik_knobs"]
        if not np.all(np.isin(expected_keys, list(data.keys()))):
            raise ValueError(f"Expected GS2 file, received {filename}")

    def is_nonlinear(self) -> bool:
        return self.data["kt_grids_knobs"]["grid_option"] == "box" and
            self.data["nonlinear_terms_knobs"].get("nonlinear_mode","off") == "on"

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GS2 input file

        """
        for key, parameter in flags.items():
            for param, val in parameter.items():
                self.data[key][param] = val


    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """

        gs2_eq = self.data["theta_grid_knobs"]["equilibrium_option"]
        if gs2_eq in ["eik", "default"]:
            local_eq = self.data["theta_grid_eik_knobs"].get("local_eq",True)
            iflux = self.data["theta_grid_eik_knobs"].get("iflux",0)
            if local_eq:
                if iflux == 0:
                    pyro.local_geometry = "Miller"
                    return self.get_local_geometry_miller()
                else:
                    #return self.get_fourier()
                    raise NotImplementedError(
                        "GS2 Fourier options are not implemented"
                    )
            else:
                raise RuntimeError("GS2 is not using local equilibrium")
        else:
            raise NotImplementedError(
                f"GS2 equilibrium option {gs2_eq} not implemented"
            )

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GS2 file
        """

        # Set some defaults here
        # FIXME Should not be modifying self as a side effect. What do these do?
        # If this is later needed for writing, this should be done at the write step
        self.data["theta_grid_eik_knobs"]["bishop"] = 4
        seld.data["theta_grid_eik_knobs"]["irho"] = 2

        miller_data = {}

        for key, val in self.pyro_gs2_miller.items():
            miller_data[key] = self.data[val[0]][val[1]]

        rho, kappa = miller_data["rho"], miller_data["kappa"]
        miller_data["delta"] = np.sin(self.data["theta_grid_parameters"]["tri"])
        miller_data["s_kappa"] = (
            self.data["theta_grid_parameters"]["akappri"] * rho / kappa
        )
        miller_data["s_delta"] = gs2["theta_grid_parameters"]["tripri"] * rho

        # Get beta and beta_prime normalised to R_major(in case R_geo != R_major)
        Rgeo = self.data["theta_grid_parameters"].get("Rgeo", miller_data["Rmaj"])

        beta = self.data["parameters"]["beta"] * (miller_data["Rmaj"] / Rgeo) ** 2
        miller_data["beta_prime"] *= (miller_data["Rmaj"] / Rgeo) ** 2

        # Assume pref*8pi*1e-7 = 1.0
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


    def get_numerics(self, shat: float) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = self.data["knobs"].get("fphi", 0.0) > 0.0
        numerics_data["apar"] = self.data["knobs"].get("fapar", 0.0) > 0.0
        numerics_data["bpar"] = self.data["knobs"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        delta_time = self.data["knobs"].get("delt", 0.005) / sqrt2
        numerics_data["delta_time"] = delta_time
        numerics_data["max_time"] = seld.data["knobs"].get("nstep", 50000) * numerics.delta_time

        # Fourier space grid
        # Linear simulation
        if self.is_linear():
            numerics_data["nky"] = 1
            numerics_data["nkx"] = 1
            numerics_data["ky"] = self.data["kt_grids_single_parameters"]["aky"] / sqrt2
            numerics_data["kx"] = 0.0
            numerics_data["theta0"] = self.data["kt_grids_single_parameters"].get("theta0",0.0)
        # Nonlinear/multiple modes in box
        # kt_grids_knobs.grid_option == "box"
        else: 
            box = self.data["kt_grids_box_parameters"]
            keys = box.keys()

            # Set up ky grid
            if "ny" in keys:
                numerics_data["nky"] = int(box["ny"] - 1) / 3 + 1)
            elif "n0" in keys:
                numerics_data["nky"] = box["n0"]
            elif "nky" in keys:
                numerics_data["nky"] = box["naky"]
            else:
                raise RuntimeError(f"ky grid details not found in {keys}")

            if "y0" in keys:
                if box["y0"] < 0.0:
                    numerics_data["ky"] = -box["y0"] / sqrt2
                else:
                    numerics_data["ky"] = 1 / box["y0"] / sqrt2
            else:
                raise RuntimeError(f"Min ky details not found in {keys}")

            if "nx" in keys:
                numerics_data["nkx"] = int((2 * box["nx"] - 1) / 3 + 1)
            elif "ntheta0" in keys():
                numerics_data["nkx"] = int((2 * box["ntheta0"] - 1) / 3 + 1)
            else:
                raise RuntimeError("kx grid details not found in {keys}")

            if abs(shat) > 1e-6:
                numerics_data["kx"] = numerics_data["ky"] * shat * 2 * pi / box["jtwist"]
            else:
                numerics_data["kx"] = 2 * pi / (box["x0"] * sqrt2)

        # Theta grid
        numerics_data["ntheta"] = self.data["theta_grid_parameters"]["ntheta"]
        numerics_data["nperiod"] = self.data["theta_grid_parameters"]["nperiod"]

        # Velocity grid
        try:
            numerics_data["nenergy"] = (
                self.data["le_grids_knobs"]["nesub"] + self.data["le_grids_knobs"]["nesuper"]
            )
        except KeyError:
            numerics_data["nenergy"] = self.data["le_grids_knobs"]["negrid"]

        # Currently using number of un-trapped pitch angles
        numerics_data["npitch"] = self.data["le_grids_knobs"]["ngauss"] * 2
        nl_mode = self.data["nonlinear_terms_knobs"].get("nonlinear_mode","off")

        numerics_data["nonlinear"] = (nl_mode == "on")

        return Numerics(numerics_data)

