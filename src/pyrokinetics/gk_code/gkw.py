import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import f90nml
import numpy as np
from cleverdict import CleverDict

from ..file_utils import FileReader
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    LocalGeometryMXH,
    MetricTerms,
    default_miller_inputs,
    default_mxh_inputs,
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
        "Z0": ["geom", "zmil"],
        "dZ0dr": ["geom", "dzmil"],
        "ip_ccw": ["geom", "signj"],
        "bt_ccw": ["geom", "signb"],
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
        "Z0": 0.0,
        "dZ0dr": 0.0,
        "ip_ccw": -1,
        "bt_ccw": -1,
    }

    pyro_gkw_mxh = {
        **pyro_gkw_miller,
        "cn": ["geom", "c"],
        "sn": ["geom", "s"],
        "dcndr": ["geom", "c_prime"],
        "dsndr": ["geom", "s_prime"],
        "n_moments": ["geom", "n_shape"],
    }

    pyro_gkw_mxh_defaults = {
        **pyro_gkw_miller_defaults,
        "cn": [0.0, 0.0, 0.0, 0.0],
        "sn": [0.0, 0.0, 0.0, 0.0],
        "dcndr": [0.0, 0.0, 0.0, 0.0],
        "dsndr": [0.0, 0.0, 0.0, 0.0],
        "n_moments": 4,
    }
    pyro_gkw_species = {
        "mass": "mass",
        "z": "z",
        "dens": "dens",
        "temp": "temp",
        "inverse_lt": "rlt",
        "inverse_ln": "rln",
        "domega_drho": "uprim",
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
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm=None,
        code_normalisation=None,
    ):
        """
        Write self.data to a gyrokinetics input file.
        Uses default write, which writes to a Fortan90 namelist
        """
        if local_norm is None:
            local_norm = Normalisation("write")

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

        # f90nml doesnt like numpy arrays...
        self.data["collisions"]["nu_ab"] = list(self.data["collisions"]["nu_ab"])

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

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_geometry")
            convention = getattr(norms, self.norm_convention)

        geometry_type = self.data["geom"]["geom_type"]

        if geometry_type == "miller":
            default_inputs = default_miller_inputs()
            pyro_gkw_local_geometry = self.pyro_gkw_miller
            pyro_gkw_local_geometry_defaults = self.pyro_gkw_miller_defaults
            local_geometry_class = LocalGeometryMiller
        elif geometry_type == "mxh":
            default_inputs = default_mxh_inputs()
            pyro_gkw_local_geometry = self.pyro_gkw_mxh
            pyro_gkw_local_geometry_defaults = self.pyro_gkw_mxh_defaults
            local_geometry_class = LocalGeometryMXH
        else:
            raise NotImplementedError(
                f"LocalGeometry type {geometry_type} not implemented for GKW"
            )

        local_geometry_data = default_inputs

        for (pyro_key, (gkw_param, gkw_key)), gkw_default in zip(
            pyro_gkw_local_geometry.items(), pyro_gkw_local_geometry_defaults.values()
        ):
            local_geometry_data[pyro_key] = self.data[gkw_param].get(
                gkw_key, gkw_default
            )

        if geometry_type == "mxh":
            for key in ["cn", "sn", "dcndr", "dsndr"]:
                local_geometry_data[key] = [float(i) for i in local_geometry_data[key]]

        for key, value in local_geometry_data.items():
            if isinstance(value, list):
                local_geometry_data[key] = np.array(value)[
                    : local_geometry_data["n_moments"]
                ]

        local_geometry_data["Rmaj"] = 1.0
        local_geometry_data["rho"] = self.data["geom"]["eps"]

        local_geometry_data["bt_ccw"] *= -1
        local_geometry_data["ip_ccw"] *= -1

        beta = self.data["spcgeneral"]["beta_ref"]
        if beta != 0.0:
            local_geometry_data["B0"] = np.sqrt(1.0 / beta)
        else:
            local_geometry_data["B0"] = None

        if self.data["spcgeneral"]["betaprime_type"] == "ref":
            local_geometry_data["beta_prime"] = self.data["spcgeneral"]["betaprime_ref"]
        elif self.data["spcgeneral"]["betaprime_type"] == "sp":
            # Need species to set up beta_prime
            local_species = self.get_local_species()
            if local_geometry_data["B0"] is not None:
                local_geometry_data["beta_prime"] = (
                    -local_species.inverse_lp.m / local_geometry_data["B0"] ** 2
                )
            else:
                local_geometry_data["beta_prime"] = 0.0
        else:
            raise ValueError(
                f"betaprime tpye {self.data['spcgeneral']['betaprime_type']} not supported for GKW"
            )

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        local_geometry = local_geometry_class.from_gk_data(local_geometry_data)

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_species(self):
        """
        Load LocalSpecies object from GKW file
        """

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        # Dictionary of local species parameters
        local_species = LocalSpecies()

        ion_count = 0

        rotation = self.data.get("rotation", {"vcor": 0.0, "shear_rate": 0.0})

        n_species = self.data["gridsize"]["number_of_species"]

        individual_coll = False
        ion_ion_coll = False
        reference_coll = False
        coll_data = self.data["collisions"]

        if coll_data.get("freq_input", False):
            individual_coll = True
            collisions = np.array(coll_data.get("nu_ab", np.zeros(n_species**2)))[
                : n_species**2
            ].reshape((n_species, n_species))
        elif coll_data.get("freq_override", False):
            ion_ion_coll = True
            collisions = coll_data.get("coll_freq", 0.0)
        else:
            reference_coll = True
            nref = coll_data.get("nref", 1.0)
            rref = coll_data.get("rref", 1.0)
            tref = coll_data.get("tref", 1.0)
            collisions = 6.5141e-5 * rref * nref / tref**2

        # Load each species into a dictionary
        for i_sp in range(n_species):
            species_data = CleverDict()

            try:
                gkw_data = self.data["species"][i_sp]
            except TypeError:
                # case when only 1 species
                gkw_data = self.data["species"]

            for pyro_key, gkw_key in self.pyro_gkw_species.items():
                species_data[pyro_key] = gkw_data[gkw_key]

            species_data["omega0"] = rotation.get("vcor", 0.0)

            if species_data.z == -1:
                name = "electron"
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            if individual_coll:
                species_data.nu = collisions[i_sp, i_sp] * np.sqrt(
                    species_data["temp"] / species_data["mass"]
                )
            elif ion_ion_coll:
                species_data.nu = (
                    collisions
                    * species_data.z**4
                    * species_data.dens
                    / species_data.temp**2
                )
            elif reference_coll:
                if name == "electron":
                    coulog = (
                        14.9
                        - 0.5 * np.log(0.1 * nref * species_data.dens)
                        + np.log(tref * species_data.temp)
                    )
                else:
                    coulog = (
                        17.3
                        - np.log(species_data.z**4 / (species_data.temp * tref))
                        - 0.5 * np.log(0.1 * nref / tref)
                        - 0.5
                        * np.log(
                            2
                            * species_data.z**2
                            * species_data.dens
                            / species_data.temp
                        )
                    )
                species_data.nu = (
                    collisions
                    * species_data.z**4
                    * species_data.dens
                    / species_data.temp**2
                    * coulog
                    * np.sqrt(species_data["temp"] / species_data["mass"])
                )

            # normalisations
            species_data.dens *= convention.nref
            species_data.mass *= convention.mref
            species_data.nu *= convention.vref / convention.lref
            species_data.temp *= convention.tref
            species_data.z *= convention.qref
            species_data.inverse_lt *= convention.lref**-1
            species_data.inverse_ln *= convention.lref**-1
            species_data.omega0 *= convention.vref / convention.lref
            species_data.domega_drho *= convention.vref / convention.lref**2

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        local_species.zeff = self.data["collisions"].get("zeff", 1.0) * convention.qref

        # Can't normalise to pyrokinetics normalisations so leave as GKW and calculate total pressure gradient
        local_species.normalise(convention)

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        numerics_data = {}

        # Set no. of fields
        numerics_data["phi"] = True
        numerics_data["apar"] = self.data["control"].get("nlapar", True)
        numerics_data["bpar"] = self.data["control"].get("nlbpar", False)

        # Set time stepping
        delta_time = self.data["control"].get("dtim", 0.005)
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
        kthrho = self.data["mode"]["kthrho"]

        if isinstance(kthrho, list):
            kthrho = kthrho[: numerics_data["nky"]]

        local_geometry = self.get_local_geometry()
        drho_dpsi = (
            local_geometry.q / local_geometry.rho / local_geometry.get_bunit_over_b0()
        )
        e_eps_zeta = drho_dpsi / (4 * np.pi)

        # Ensure odd ntheta to get  theta = 0.0 on grid
        metric_ntheta = (numerics_data["ntheta"] // 2) * 2 + 1
        metric_terms = MetricTerms(local_geometry, ntheta=metric_ntheta)
        theta_index = np.argmin(abs(metric_terms.regulartheta))
        g_aa = metric_terms.field_aligned_contravariant_metric("alpha", "alpha")[
            theta_index
        ]
        kthnorm = np.sqrt(g_aa) / (2 * np.pi)

        numerics_data["ky"] = kthrho * (e_eps_zeta * 2 / kthnorm).m
        numerics_data["kx"] = self.data["mode"].get("chin", 0)
        numerics_data["theta0"] = self.data["mode"].get("chin", 0.0)

        # Velocity grid
        numerics_data["nenergy"] = self.data["gridsize"].get("n_vpar_grid", 32) // 2
        numerics_data["npitch"] = self.data["gridsize"].get("n_mu_grid", 16)

        # Beta
        numerics_data["beta"] = self.data["spcgeneral"]["beta_ref"]

        rotation = self.data.get("rotation", {"vcor": 0.0, "shear_rate": 0.0})

        numerics_data["gamma_exb"] = rotation.get("shear_rate", 0.0)

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in normalisation values from input file

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
            "lref": "major_radius",
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
        for i_sp in range(self.data["gridsize"]["number_of_species"]):

            dens = self.data["species"][i_sp]["dens"]
            temp = self.data["species"][i_sp]["temp"]
            mass = self.data["species"][i_sp]["mass"]

            # Find all reference values
            if self.data["species"][i_sp]["z"] == -1:
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

        if not found_electron:
            ne = 0.0
            for i_sp in range(self.data["gridsize"]["number_of_species"]):
                ne += (
                    self.data["species"][i_sp]["dens"] * self.data["species"][i_sp]["z"]
                )
            electron_density = ne
            electron_temperature = self.data["species"][0]["temp"]

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
            major_radius=1.0,
            rgeo_rmaj=1.0,
            minor_radius=None,
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
                template_file = gk_templates["GKW"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

        # Set Miller Geometry bits
        if isinstance(local_geometry, LocalGeometryMiller):
            # Miller settings
            self.data["geom"]["geom_type"] = "miller"
        elif isinstance(local_geometry, LocalGeometryMXH):
            # Miller settings
            self.data["geom"]["geom_type"] = "mxh"
        else:
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GKW not supported yet"
            )

        self.data["geom"]["eps"] = local_geometry.rho

        if local_geometry.local_geometry == "Miller":
            for pyro_key, (gkw_param, gkw_key) in self.pyro_gkw_miller.items():
                self.data[gkw_param][gkw_key] = local_geometry[pyro_key]
        elif local_geometry.local_geometry == "MXH":
            for pyro_key, (gkw_param, gkw_key) in self.pyro_gkw_mxh.items():
                self.data[gkw_param][gkw_key] = local_geometry[pyro_key]

        # GENE defines whether clockwise/ pyro defines whether counter-clockwise - need to flip sign
        self.data["geom"]["signj"] = -1 * local_geometry.ip_ccw
        self.data["geom"]["signb"] = -1 * local_geometry.bt_ccw

        # Pyro forces to its own value of beta_prime
        self.data["spcgeneral"]["betaprime_type"] = "ref"
        self.data["spcgeneral"]["betaprime_ref"] = local_geometry.beta_prime

        # Kinetic data
        n_species = local_species.nspec
        self.data["gridsize"]["number_of_species"] = n_species

        stored_species = len(self.data["species"])
        extra_species = stored_species - n_species

        if extra_species > 0:
            for i in range(extra_species):
                del self.data["species"][-1]

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

            for key, val in self.pyro_gkw_species.items():
                single_species[val] = local_species[name][key]

        self.data["collisions"]["zeff"] = local_species.zeff

        # write collision information
        self.data["collisions"]["freq_override"] = False
        self.data["collisions"]["freq_input"] = True

        nu_ee = local_species["electron"].nu
        e_mass = local_species["electron"].mass
        te = local_species["electron"].temp
        ne = local_species["electron"].dens
        ze = local_species["electron"].z

        nu_ab_array = np.zeros(local_species.nspec**2)
        counter = 0
        for b in local_species.names:
            for a in local_species.names:
                dens = local_species[f"{b}"].dens
                mass = local_species[f"{a}"].mass
                temp = local_species[f"{a}"].temp
                Za = local_species[f"{a}"].z
                Zb = local_species[f"{b}"].z
                nu_ab = (
                    (
                        (dens / ne)
                        * ((Za / ze) ** 2)
                        * ((Zb / ze) ** 2)
                        / (((temp / te) ** 1.5) * (mass / e_mass) ** 0.5)
                    )
                    * nu_ee
                    / np.sqrt(temp.m / mass.m)
                )
                nu_ab_array[counter] = nu_ab.m
                counter += 1

        self.data["collisions"]["nu_ab"] = nu_ab_array * nu_ee.units

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
        dtim = numerics.delta_time
        naverage = int(1.0 / dtim.m)  # write every vth/R time
        self.data["control"]["dtim"] = dtim
        self.data["control"]["naverage"] = naverage
        self.data["control"]["ntime"] = (
            int(numerics.max_time / numerics.delta_time) // naverage
        )

        drho_dpsi = local_geometry.q / local_geometry.rho / local_geometry.bunit_over_b0
        e_eps_zeta = drho_dpsi / (4 * np.pi)

        # Ensure odd ntheta to get  theta = 0.0 on grid
        metric_ntheta = (numerics.ntheta // 2) * 2 + 1
        metric_terms = MetricTerms(local_geometry, ntheta=metric_ntheta)
        theta_index = np.argmin(abs(metric_terms.regulartheta))
        g_aa = metric_terms.field_aligned_contravariant_metric("alpha", "alpha")[
            theta_index
        ]
        kthnorm = np.sqrt(g_aa) / (2 * np.pi)
        kthrho = (
            numerics.ky
            * (1 * convention.bref / local_norm.gs2.bref).to_base_units()
            / (e_eps_zeta * 2 / kthnorm).m
        )

        # mode box / single mode
        self.data["control"]["non_linear"] = numerics.nonlinear
        if numerics.nky == 1 and numerics.nky == 1:
            self.data["control"]["non_linear"] = False
            self.data["mode"]["mode_box"] = False
            self.data["mode"]["kthrho"] = kthrho
            self.data["mode"]["chin"] = numerics.theta0
            self.data["gridsize"]["nx"] = 1
            self.data["gridsize"]["nmod"] = 1
            self.data["gridsize"]["nperiod"] = numerics.nperiod
            self.data["gridsize"]["n_s_grid"] = (
                2 * numerics.nperiod - 1
            ) * numerics.ntheta

        else:
            self.data["mode"]["mode_box"] = True
            self.data["gridsize"]["nx"] = numerics.nkx
            self.data["gridsize"]["nmod"] = numerics.nky
            self.data["gridsize"]["nperiod"] = 1
            self.data["gridsize"]["n_s_grid"] = numerics.ntheta

        # velocity grid
        self.data["gridsize"]["n_mu_grid"] = numerics.npitch
        self.data["gridsize"]["n_vpar_grid"] = numerics.nenergy * 2

        # Rotation
        self.data["rotation"]["vcor"] = local_species.electron.omega0
        self.data["rotation"]["shear_rate"] = numerics.gamma_exb

        if not local_norm:
            return

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

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


class GKWFile:
    def __init__(self, path: PathLike, required: bool, binary: bool):
        self.path = Path(path)
        self.required = required
        self.binary = binary
        self.fmt = self.path.name.split(".")[0]


def _fromfile(*args, **kwargs):
    """Replacement to ``np.fromfile`` that always promotes to 64 bit float.

    Older versions of NumPy had different rules for type promotion which
    could lead to unintentional loss of precision.
    """
    return np.asarray(np.fromfile(*args, **kwargs), dtype=float)


class GKOutputReaderGKW(FileReader, file_type="GKW", reads=GKOutput):
    fields = ["phi", "apar", "bpar"]
    moments = ["density", "temperature", "velocity"]

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
        fields = self._get_fields(raw_data, coords) if load_fields else None
        fluxes = self._get_fluxes(raw_data, coords) if load_fluxes else None
        moments = self._get_moments(raw_data, coords) if load_moments else None

        if load_fields and len(fields[coords["field"][0]].shape) == 3:
            field_dims = ("theta", "kx", "ky")
        else:
            field_dims = ("theta", "kx", "ky", "time")

        if load_moments and len(moments[coords["moment"][0]].shape) == 4:
            moment_dims = ("kx", "ky", "theta", "species")
        else:
            moment_dims = ("kx", "ky", "theta", "species", "time")

        field_normalise = gk_input.data["control"].get("normalized", True)

        if coords["linear"]:
            eigenvalues = self._get_eigenvalues(raw_data, coords)
            if "time" in field_dims:
                if field_normalise:
                    amplitude = np.exp(
                        eigenvalues["growth_rate"].flatten() * coords["time"]
                    )
                else:
                    eigenvalues = None
                    amplitude = 1

                for f in fields.keys():
                    fields[f] *= amplitude
        else:
            eigenvalues = None

        eigenfunctions = None
        eigenfunction_dims = None

        # Assign units and return GKOutput
        convention = norm.gkw
        norm.default_convention = output_convention.lower()

        flux_dims = ("field", "time", "species")
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
                else Eigenfunctions(eigenfunctions, dims=eigenfunction_dims).with_units(
                    convention
                )
            ),
            linear=coords["linear"],
            gk_code="GKW",
            input_file=input_str,
            output_convention=output_convention,
        )

    def verify_file_type(self, dirname: PathLike):
        dirname = Path(dirname)
        for f in self._required_files(dirname).values():
            if not f.path.exists():
                raise RuntimeError(f"Missing the file '{f}'")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        For GKW, simply returns dir of the path.
        """
        return Path(filename).parent

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": GKWFile(dirname / "input.dat", required=True, binary=False),
            "time": GKWFile(dirname / "time.dat", required=True, binary=False),
            "parallel": GKWFile(dirname / "parallel.dat", required=True, binary=False),
            "krho": GKWFile(dirname / "krho", required=True, binary=False),
            "kxrh": GKWFile(dirname / "kxrh", required=True, binary=False),
            "geom": GKWFile(dirname / "geom.dat", required=True, binary=False),
            "file_count": GKWFile(dirname / "file_count", required=True, binary=False),
            "flux_phi": GKWFile(dirname / "fluxes.dat", required=False, binary=False),
            "flux_apar": GKWFile(
                dirname / "fluxes_em.dat", required=False, binary=False
            ),
            "flux_bpar": GKWFile(
                dirname / "fluxes_bpar.dat", required=False, binary=False
            ),
        }

    @staticmethod
    def _get_gkw_field_files(dirname: PathLike, raw_data: dict):
        dirname = Path(dirname)
        field_names = {"phi": "Phi", "apar": "Apa", "bpar": "Bpa"}

        for pyro_field, gkw_field in field_names.items():
            raw_data[f"field_{pyro_field}"] = [
                dirname / f
                for f in sorted(os.listdir(dirname))
                if re.search(rf"^{gkw_field}_kykxs\d{{8}}_\w{{4}}", f)
            ]

    @staticmethod
    def _get_gkw_moment_files(dirname: PathLike, raw_data: dict):
        dirname = Path(dirname)
        moment_names = {
            "density": "dens",
            "temperature_par": "Tpar",
            "temperature_perp": "Tperp",
            "velocity": "vpar",
        }

        for pyro_moment, gkw_moment in moment_names.items():
            raw_data[f"moment_{pyro_moment}"] = [
                dirname / f
                for f in sorted(os.listdir(dirname))
                if re.search(rf"^{gkw_moment}_kykxs\d{{2}}_\d{{6}}_\w{{4}}", f)
            ]

    @classmethod
    def _get_raw_data(cls, dirname: PathLike) -> Tuple[Dict[str, Any], GKInputGKW, str]:
        expected_data = cls._required_files(dirname)

        # Read in files
        raw_data = {}

        for key, gkw_file in expected_data.items():
            if not gkw_file.path.exists():
                if gkw_file.required:
                    raise RuntimeError(
                        f"GKOutputReaderGKW: The file {gkw_file.path.name} is needed"
                    )
                continue
            # Read in file according to format
            if key == "input":
                with open(gkw_file.path, "r") as f:
                    raw_data[key] = f.read()
            elif key == "geom":
                with open(gkw_file.path, "r") as f:
                    raw_data[key] = f.read().split("\n")
            else:
                raw_data[key] = np.loadtxt(gkw_file.path)

        input_str = raw_data["input"]
        # Read as GKInputGKW and into plain string
        gk_input = GKInputGKW()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        cls._get_gkw_field_files(dirname, raw_data)
        cls._get_gkw_moment_files(dirname, raw_data)

        # Defer processing field and flux data until their respective functions
        # Simply return files in place of raw data
        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, Any], gk_input: GKInputGKW, downsize: int = 1
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of GKW
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing GKW output.
            gk_input (GKInputGKW): Processed GKW input file.

        Returns:
            Dict:  Dictionary with coords
        """

        # Process time data
        time = raw_data["time"][:, 0]

        if len(time) % downsize != 0:
            residual = len(time) % downsize - downsize
        else:
            residual = 0

        time = time[::downsize]

        # read geometrical terms to map from gkw.ky to pyro.ky
        geom = raw_data["geom"]
        kth_index = geom.index("kthnorm")
        eez_index = geom.index("E_eps_zeta")
        e_eps_zeta = float(geom[eez_index + 1].split(" ")[-1])
        kthnorm = float(geom[kth_index + 1])

        kx = np.array([raw_data["kxrh"]]) * 2.0 * e_eps_zeta / kthnorm
        ky = np.array([raw_data["krho"]]) * 2.0 * e_eps_zeta / kthnorm

        fields = ["phi", "apar", "bpar"]
        fields_defaults = [True, False, False]
        fields = [
            f
            for f, d in zip(fields, fields_defaults)
            if gk_input.data["control"].get(f"nl{f}", d)
        ]

        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]
        species = gk_input.get_local_species().names

        # Eigenfunctions repeated for each species
        theta_index = geom.index("poloidal_angle")
        g_eps_eps_index = geom.index("g_eps_eps")

        theta = []
        for i in range(g_eps_eps_index - theta_index - 1):
            theta.extend(
                [float(th) for th in geom[theta_index + i + 1].strip().split(" ") if th]
            )

        n_theta = len(theta)

        n_energy = gk_input.data["gridsize"]["n_vpar_grid"] // 2
        energy = np.linspace(0, n_energy - 1, n_energy)

        n_pitch = gk_input.data["gridsize"]["n_mu_grid"]
        pitch = np.linspace(0, n_pitch - 1, n_pitch)

        file_count = raw_data["file_count"]

        test_binary = _fromfile(raw_data[f"field_{fields[0]}"][0], dtype="float32")
        if len(test_binary) == n_theta:
            binary_dtype = "float32"
        elif len(test_binary) == 2 * n_theta:
            binary_dtype = "float64"
        else:
            raise ValueError("Cannot determine dtype of binary GKW output")

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "linear": gk_input.is_linear(),
            "downsize": downsize,
            "residual": residual,
            "file_count": file_count,
            "binary_dtype": binary_dtype,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """
        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])
        nfield = len(coords["field"])
        downsize = coords["downsize"]
        residual = coords["residual"]
        binary_dtype = coords["binary_dtype"]

        full_ntime = ntime * downsize + residual

        field_names = ["phi", "apar", "bpar"][:nfield]

        if len(raw_data[f"field_{field_names[0]}"]) == 0:
            raise FileNotFoundError("No field files found for GKW Output.")
        elif len(raw_data[f"field_{field_names[0]}"]) != 2 * ntime:
            full_ntime = 1

        results = {}

        # Loop through all fields and add field
        for ifield, field_name in enumerate(field_names):

            fields = np.empty((ntheta, nkx, nky, full_ntime), dtype=complex)
            raw_fields = np.empty((ntheta * nkx * nky, full_ntime), dtype=complex)

            for i_time in range(full_ntime):
                if full_ntime == 1:
                    i_time = -1
                imag_index = 2 * i_time
                real_index = imag_index + 1

                raw_fields[:, i_time] = _fromfile(
                    raw_data[f"field_{field_name}"][real_index], dtype=binary_dtype
                ) - 1j * _fromfile(
                    raw_data[f"field_{field_name}"][imag_index], dtype=binary_dtype
                )

            fields = np.reshape(raw_fields, fields.shape)
            fields = fields[:, :, :, ::downsize]

            if full_ntime == 1:
                fields = np.squeeze(fields, axis=-1)

            # Move theta to 0 axis
            results[field_name] = fields

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])
        nspecies = len(coords["species"])
        nmoment = len(coords["moment"])
        downsize = coords["downsize"]
        residual = coords["residual"]
        binary_dtype = coords["binary_dtype"]

        full_ntime = ntime * downsize + residual

        moment_names = ["density", "temperature_par", "temperature_perp", "velocity"][
            : nmoment + 1
        ]

        if len(raw_data[f"moment_{moment_names[0]}"]) == 0:
            raise FileNotFoundError("No moment files found for GKW Output")
        if len(raw_data[f"moment_{moment_names[0]}"]) != 2 * ntime * nspecies:
            full_ntime = 1

        results = {}

        # Loop through all moments and add moment
        for imoment, moment_name in enumerate(moment_names):

            moments = np.empty((nkx, nky, ntheta, nspecies, full_ntime), dtype=complex)
            raw_moments = np.empty(
                (nkx * nky * ntheta, nspecies, full_ntime), dtype=complex
            )

            for i_time in range(full_ntime):
                if full_ntime == 1:
                    i_time = -1

                for i_spec in range(nspecies):
                    i_time_species = i_time * (2 * nspecies) + i_spec * 2
                    imag_index = i_time_species
                    real_index = i_time_species + 1
                    raw_moments[:, i_spec, i_time] = _fromfile(
                        raw_data[f"moment_{moment_name}"][real_index],
                        dtype=binary_dtype,
                    ) + 1j * _fromfile(
                        raw_data[f"moment_{moment_name}"][imag_index],
                        dtype=binary_dtype,
                    )

            moments = np.reshape(raw_moments, moments.shape)
            moments = moments[:, :, :, :, ::downsize]

            if full_ntime == 1:
                moments = np.squeeze(moments, axis=-1)

            results[moment_name] = moments

        results["temperature"] = np.sqrt(
            results["temperature_par"] ** 2 + results["temperature_perp"] ** 2
        )

        del results["temperature_par"]
        del results["temperature_perp"]

        return results

    @staticmethod
    def _get_fluxes(
        raw_data: Dict[str, Any],
        coords: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Set flux data over time.
        The flux coordinates should be (species, moment, field, ky, time)
        """

        results = {}
        ntime = len(coords["time"])
        nspecies = len(coords["species"])
        nflux = len(coords["flux"])
        downsize = coords["downsize"]
        residual = coords["residual"]
        fields = coords["field"]
        nfield = len(fields)

        ntime = ntime * downsize + residual

        fluxes = np.empty((nfield, ntime, nspecies, nflux))

        for ifield, field in enumerate(fields):
            flux_key = f"flux_{field}"

            if flux_key in raw_data:
                raw_fluxes = raw_data[flux_key]
                raw_fluxes = np.reshape(raw_fluxes, (ntime, nspecies, nflux))
                fluxes[ifield, ...] = raw_fluxes

        for iflux, flux in enumerate(coords["flux"]):
            results[flux] = fluxes[:, ::downsize, :, iflux]

        return results

    @classmethod
    def _get_eigenvalues(
        self, raw_data: Dict[str, Any], coords: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(kx, ky, time)
        data['mode_frequency'] = mode_frequency(kx, ky, time)
        data['growth_rate'] = growth_rate(kx, ky, time)

        This should be called after _set_fields, and is only valid for linear runs.
        Unlike the version in the super() class, GKW may need to get extra info from
        an eigenvalue file.

        Args:
            data (xr.Dataset): The dataset to be modified.
            dirname (PathLike): Directory containing GKW output files.
        Returns:
            Dict: The modified dataset which was passed to 'data'.
        """

        ntime = len(coords["time"])
        nky = len(coords["ky"])
        nkx = len(coords["kx"])
        shape = (nkx, nky, ntime)

        growth_rate = raw_data["time"][:, 1].reshape(shape)
        mode_frequency = raw_data["time"][:, 2].reshape(shape)

        result = {
            "growth_rate": growth_rate,
            "mode_frequency": mode_frequency,
        }

        return result

    @staticmethod
    def _get_eigenfunctions(raw_data: Dict[str, Any], coords: Dict) -> np.ndarray:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(kx, ky, field, theta)

        This should be called after _set_fields, and is only valid for linear runs.
        """

        ntheta = len(coords["theta"])
        nkx = len(coords["kx"])
        nky = len(coords["ky"])

        indexes = {"phi": [1, 2], "apar": [3, 4], "bpar": [13, 14]}

        coord_names = ["field", "theta", "kx", "ky"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        parallel_data = raw_data["parallel"]
        for ifield, field in enumerate(coords["field"]):
            real_index = indexes[field][0]
            imag_index = indexes[field][1]

            eigenfunctions_data = (
                parallel_data[:ntheta, real_index]
                + 1j * parallel_data[:ntheta, imag_index]
            )
            eigenfunctions[ifield, ...] = np.reshape(
                eigenfunctions_data, (ntheta, nkx, nky)
            )

        result = eigenfunctions

        return result
