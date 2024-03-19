import copy
import warnings
from typing import Any, Dict, Optional

import f90nml
import numpy as np
import pint
from cleverdict import CleverDict

from ..file_utils import FileReader
from ..constants import deuterium_mass, electron_mass, pi, sqrt2
from ..local_geometry import LocalGeometry, LocalGeometryMiller, default_miller_inputs
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict, ureg
from ..numerics import Numerics
from ..templates import gk_templates
from ..typing import PathLike
from .gk_input import GKInput


class GKInputGKW(GKInput, FileReader, file_type="GKW", reads=GKInput):
    """
    Class that can read GKW input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GKW"
    default_file_name = "input.dat"
    norm_convention = "gkw"

    pyro_gkw_miller = {
        "rho": ["geom", "eps"],
        "q": ["geom", "q"],
        "shat": ["geom", "shat"],
        "kappa": ["geom", "kappa"],
        "s_kappa": ["geom", "skappa"],
        "delta": ["geom", "delta"],
        "s_delta": ["geom", "sdelta"],
        "shift": ["geom", "drmil"],
    }

    pyro_gkw_miller_defaults = {
        "rho": 0.16666,
        "q": 2.0,
        "shat": 1.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shift": 0.0,
    }

    pyro_gkw_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "inverse_lt": "rlt",
        "inverse_ln": "rln",
        "inverse_lv": "uprim",
    }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GKW input file into a dictionary
        Uses default read, which assumes input is a Fortran90 namelist
        """
        return super().read_from_file(filename)

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GKW input file given as string
        Uses default read_str, which assumes input_string is a Fortran90 namelist
        """
        return super().read_str(input_string)

    def read_dict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads GKW input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid gkw input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        expected_keys = ["control", "gridsize", "mode", "geom", "spcgeneral"]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as GKW file")

    def write(self, filename: PathLike, float_format: str = "", local_norm=None):
        """
        Write self.data to a gyrokinetics input file.
        Uses default write, which writes to a Fortan90 namelist
        """
        if local_norm is None:
            local_norm = Normalisation("write")
            aspect_ratio = (
                self.data["pyrokinetics_info"]["major_r"]
                / self.data["pyrokinetics_info"]["minor_r"]
            )
            local_norm.set_ref_ratios(aspect_ratio=aspect_ratio)

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, local_norm.gs2)

        super().write(filename, float_format=float_format)

    def is_nonlinear(self) -> bool:
        is_box = self.data["mode"]["mode_box"]
        is_nonlin = self.data["control"]["non_linear"]
        return is_box and is_nonlin

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GKW input file
        """
        super().add_flags(flags)

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """
        geometry_type = self.data["geom"]["geom_type"]
        if geometry_type == "miller":
            return self.get_local_geometry_miller()
        else:
            raise NotImplementedError(
                f"LocalGeometry type {geometry_type} not implemented for GKW"
            )

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GKW file
        """
        miller_data = default_miller_inputs()

        for (pyro_key, (gkw_param, gkw_key)), gkw_default in zip(
            self.pyro_gkw_miller.items(), self.pyro_gkw_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data[gkw_param].get(gkw_key, gkw_default)

        a_minor = self.data["pyrokinetics_info"]["minor_r"]
        R_major = self.data["pyrokinetics_info"]["major_r"]
        miller_data["Rmaj"] = R_major / a_minor
        miller_data["rho"] = self.data["geom"]["eps"] * miller_data["Rmaj"]

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data["spcgeneral"]["beta_ref"] * ne_norm * Te_norm
        if beta != 0.0:
            miller_data["B0"] = np.sqrt(1.0 / beta)
        else:
            miller_data["B0"] = None

        miller_data["beta_prime"] = (
            self.data["spcgeneral"]["betaprime_ref"] / miller_data["Rmaj"]
        )

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        return LocalGeometryMiller.from_gk_data(miller_data)

    def get_local_species(self):
        """
        Load LocalSpecies object from GKW file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        ne_norm, Te_norm = self.get_ne_te_normalisation()

        # Load each species into a dictionary
        for i_sp in range(self.data["gridsize"]["number_of_species"]):
            species_data = CleverDict()

            try:
                gkw_data = self.data["species"][i_sp]
            except TypeError:
                # case when only 1 species
                gkw_data = self.data["species"]

            for pyro_key, gkw_key in self.pyro_gkw_species.items():
                species_data[pyro_key] = gkw_data[gkw_key]

            # GKW: R_major -> Pyro: a_minor normalization
            Rmaj = (
                self.data["pyrokinetics_info"]["major_r"]
                / self.data["pyrokinetics_info"]["minor_r"]
            )
            species_data["inverse_lt"] = gkw_data["rlt"] / Rmaj
            species_data["inverse_ln"] = gkw_data["rln"] / Rmaj
            species_data["inverse_lv"] = gkw_data["uprim"] / Rmaj
            species_data["vel"] = 0.0 * ureg.vref_most_probable

            if species_data.z == -1:
                name = "electron"
                species_data.nu = 0  # None
            else:
                ion_count += 1
                name = f"ion{ion_count}"
                species_data.nu = 0  # None

            species_data.name = name

            # normalisations
            species_data.dens *= ureg.nref_electron / ne_norm
            species_data.mass *= ureg.mref_deuterium
            species_data.nu *= ureg.vref_most_probable / ureg.lref_minor_radius
            species_data.temp *= ureg.tref_electron / Te_norm
            species_data.z *= ureg.elementary_charge
            species_data.inverse_lt *= ureg.lref_minor_radius**-1
            species_data.inverse_ln *= ureg.lref_minor_radius**-1
            species_data.inverse_lv *= ureg.lref_minor_radius**-1

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.zeff = (
            self.data["collisions"].get("zeff", 1.0) * ureg.elementary_charge
        )

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        local_species.normalise()

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = True
        numerics_data["apar"] = self.data["control"].get("nlapar", True)
        numerics_data["bpar"] = self.data["control"].get("nlbpar", False)

        # Set time stepping
        delta_time = self.data["control"].get("dtim", 0.005) / sqrt2  # lref = R0, not a
        naverage = self.data["control"].get("naverage", 100)
        ntime = self.data["control"].get("ntime", 100)

        numerics_data["delta_time"] = delta_time
        numerics_data["max_time"] = delta_time * naverage * ntime

        # Theta grid
        n_s_grid = self.data["gridsize"]["n_s_grid"]
        nperiod = self.data["gridsize"]["nperiod"]
        numerics_data["nperiod"] = nperiod
        numerics_data["ntheta"] = n_s_grid // (2 * nperiod - 1)

        # Mode box specifications
        numerics_data["nonlinear"] = self.is_nonlinear()
        numerics_data["nkx"] = self.data["gridsize"]["nx"]
        numerics_data["nky"] = self.data["gridsize"]["nmod"]
        numerics_data["ky"] = self.data["mode"]["kthrho"]
        numerics_data["kx"] = self.data["mode"].get("chin", 0)
        numerics_data["theta0"] = self.data["mode"].get("chin", 0.0)

        # Velocity grid
        numerics_data["nenergy"] = self.data["gridsize"].get("n_vpar_grid", 32) // 2
        numerics_data["npitch"] = self.data["gridsize"].get("n_mu_grid", 16)

        # Beta
        ne_norm, Te_norm = self.get_ne_te_normalisation()
        numerics_data["beta"] = (
            self.data["spcgeneral"]["beta_ref"]
            * ureg.beta_ref_ee_B0
            * ne_norm
            * Te_norm
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
                template_file = gk_templates["GKW"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        # Set Miller Geometry bits
        if isinstance(local_geometry, LocalGeometryMiller):
            eq_type = "Miller"
        else:
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GKW not supported yet"
            )

        # Miller settings
        self.data["geom"]["geom_type"] = "miller"
        self.data["geom"]["eps"] = local_geometry.rho / local_geometry.Rmaj
        self.data["spcgeneral"]["betaprime_type"] = "sp"
        for pyro_key, (gkw_param, gkw_key) in self.pyro_gkw_miller.items():
            self.data[gkw_param][gkw_key] = local_geometry[pyro_key]

        # species
        # FIXME check normalization from a_minor -> Rmajor
        self.data["gridsize"]["number_of_species"] = local_species.nspec

        for iSp, name in enumerate(local_species.names):
            try:
                single_species = self.data["species"][iSp]
            except IndexError:
                if f90nml.__version__ < "1.4":
                    self.data["species"].append(copy.copy(self.data["species"][0]))
                    single_species = self.data["species"][iSp]
                else:
                    # FIXME f90nml v1.4+ uses 'Cogroups' for Namelist groups sharing
                    # a common key. As of version 1.4.2, Cogroup derives from
                    # 'list', but does not implement all methods, so confusingly it
                    # allows calls to 'append', but then doesn't do anything!
                    # Currently working around this in a horribly inefficient
                    # manner, by deconstructing the entire Namelist to a dict, using
                    # secret cogroup names directly, and rebundling the Namelist.
                    # There must be a better way!
                    d = self.data.todict()
                    copied = copy.deepcopy(d["_grp_species_0"])
                    copied["name"] = None
                    d[f"_grp_species_{iSp}"] = copied
                    self.data = f90nml.Namelist(d)
                    single_species = self.data["species"][iSp]

            # TODO Currently forcing GKW to use default pyro. Should check local_norm first
            for key, val in self.pyro_gkw_species.items():
                single_species[val] = local_species[name][key].to(
                    local_norm.pyrokinetics
                )

        self.data["collisions"]["zeff"] = local_species.zeff

        # beta_ref = local_norm.gs2.beta if local_norm else 0.0
        beta_ref = 0.0
        self.data["spcgeneral"]["beta_ref"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        # Set numerics
        # fields
        self.data["control"]["nlphi"] = numerics.phi
        self.data["control"]["nlapar"] = numerics.apar
        self.data["control"]["nlbpar"] = numerics.bpar

        # time stepping
        dtim = numerics.delta_time * sqrt2
        naverage = int(1.0 / dtim)  # write every vth/R time
        self.data["control"]["dtim"] = dtim
        self.data["control"]["naverage"] = naverage
        self.data["control"]["ntime"] = (
            int(numerics.max_time / numerics.delta_time) // naverage
        )

        # mode box / single mode
        self.data["control"]["non_linear"] = numerics.nonlinear
        if numerics.nky == 1 and numerics.nky == 1:
            self.data["control"]["non_linear"] = False
            self.data["mode"]["mode_box"] = False
            self.data["mode"]["kthrho"] = numerics.ky
            self.data["mode"]["chin"] = numerics.theta0
            self.data["gridsize"]["nx"] = 1
            self.data["gridsize"]["nmod"] = 1
            self.data["gridsize"]["nperiod"] = numerics.nperiod
            self.data["gridsize"]["n_s_grid"] = (
                2 * numerics.nperiod - 1
            ) * numerics.ntheta

            warnings.warn(
                "gs2.aky = gkw.kthrho * (gkw.e_eps_zeta * 2 / gkw.kthnorm) = pyro.ky * sqrt(2) "
                "kthnorm and e_eps_zeta can be found by in geom.dat generated by GKW"
            )
        else:
            self.data["mode"]["mode_box"] = True
            self.data["gridsize"]["nx"] = numerics.nkx
            self.data["gridsize"]["nmod"] = numerics.nky
            self.data["gridsize"]["nperiod"] = 1
            self.data["gridsize"]["n_s_grid"] = numerics.ntheta

        # velocity grid
        self.data["gridsize"]["n_mu_grid"] = numerics.npitch
        self.data["gridsize"]["n_vpar_grid"] = numerics.nenergy * 2

        if not local_norm:
            return

        # for name, namelist in self.data.items():
        #    self.data[name] = convert_dict(namelist, local_norm.gs2)        # FIXME local_norm.???

    def get_ne_te_normalisation(self):
        adiabatic_electrons = True
        # Get electron temp and density to normalise input
        for i_sp in range(self.data["gridsize"]["number_of_species"]):
            if self.data["species"][i_sp]["z"] == -1:
                ne = self.data["species"][i_sp]["dens"]
                Te = self.data["species"][i_sp]["temp"]
                adiabatic_electrons = False

        if adiabatic_electrons:
            ne = 0.0
            for i_sp in range(self.data["gridsize"]["number_of_species"]):
                ne += (
                    self.data["species"][i_sp]["dens"] * self.data["species"][i_sp]["z"]
                )
            Te = self.data["species"][0]["temp"]

        return ne, Te
