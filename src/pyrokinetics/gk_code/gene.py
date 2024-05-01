import copy
import csv
import logging
import re
import struct
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import f90nml
import h5py
import numpy as np
import pint
from cleverdict import CleverDict

from ..constants import deuterium_mass, electron_mass, pi
from ..file_utils import FileReader
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    LocalGeometryMillerTurnbull,
    default_miller_inputs,
    default_miller_turnbull_inputs,
)
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict, ureg
from ..numerics import Numerics
from ..templates import gk_templates
from ..typing import PathLike
from .gk_input import GKInput
from .gk_output import Coords, Eigenvalues, Fields, Fluxes, GKOutput, Moments


class GKInputGENE(GKInput, FileReader, file_type="GENE", reads=GKInput):
    """
    Class that can read GENE input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GENE"
    default_file_name = "input.gene"
    norm_convention = "gene"
    _convention_dict = {}

    pyro_gene_miller = {
        "q": ["geometry", "q0"],
        "kappa": ["geometry", "kappa"],
        "s_kappa": ["geometry", "s_kappa"],
        "delta": ["geometry", "delta"],
        "s_delta": ["geometry", "s_delta"],
        "shat": ["geometry", "shat"],
        "shift": ["geometry", "drr"],
        "ip_ccw": ["geometry", "sign_Ip_CW"],
        "bt_ccw": ["geometry", "sign_Bt_CW"],
    }

    pyro_gene_miller_default = {
        "q": None,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shat": 0.0,
        "shift": 0.0,
        "ip_ccw": -1,
        "bt_ccw": -1,
    }

    pyro_gene_miller_turnbull = {
        "q": ["geometry", "q0"],
        "kappa": ["geometry", "kappa"],
        "s_kappa": ["geometry", "s_kappa"],
        "delta": ["geometry", "delta"],
        "s_delta": ["geometry", "s_delta"],
        "zeta": ["geometry", "zeta"],
        "s_zeta": ["geometry", "s_zeta"],
        "shat": ["geometry", "shat"],
        "shift": ["geometry", "drr"],
        "ip_ccw": ["geometry", "sign_Ip_CW"],
        "bt_ccw": ["geometry", "sign_Bt_CW"],
    }

    pyro_gene_miller_turnbull_default = {
        "q": None,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shat": 0.0,
        "shift": 0.0,
        "ip_ccw": -1,
        "bt_ccw": -1,
    }

    pyro_gene_circular = {
        "q": ["geometry", "q0"],
        "shat": ["geometry", "shat"],
    }

    pyro_gene_circular_default = {
        "q": None,
        "shat": 0.0,
    }

    pyro_gene_species = {
        "mass": "mass",
        "z": "charge",
        "dens": "dens",
        "temp": "temp",
        "inverse_lt": "omt",
        "inverse_ln": "omn",
    }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GENE input file into a dictionary
        Uses default read, which assumes input is a Fortran90 namelist
        """
        return super().read_from_file(filename)

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GENE input file given as string
        Uses default read_str, which assumes input is a Fortran90 namelist
        """
        return super().read_str(input_string)

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads GENE input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid gene input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        expected_keys = ["general", "geometry", "box"]
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm: Normalisation = None,
        code_normalisation: str = None,
    ):
        """
        Write self.data to a gyrokinetics input file.
        Uses default write, which writes to a Fortan90 namelist
        """

        if local_norm is None:
            local_norm = Normalisation("write")
            aspect_ratio = (
                self.data["geometry"]["major_r"] / self.data["geometry"]["minor_r"]
            )
            local_norm.set_ref_ratios(aspect_ratio=aspect_ratio)

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

        super().write(filename, float_format=float_format)

    def is_nonlinear(self) -> bool:
        return bool(self.data["general"].get("nonlinear", False))

    def add_flags(self, flags) -> None:
        """
        Add extra flags to GENE input file
        Uses default, which assumes a Fortan90 namelist
        """
        super().add_flags(flags)

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """
        geometry_type = self.data["geometry"]["magn_geometry"]
        if geometry_type == "miller":
            if (
                self.data["geometry"].get("zeta", 0.0) != 0.0
                or self.data["geometry"].get("zeta", 0.0) != 0.0
            ):
                local_geometry = self.get_local_geometry_miller_turnbull()
            else:
                local_geometry = self.get_local_geometry_miller()
        elif geometry_type == "circular":
            local_geometry = self.get_local_geometry_circular()
        else:
            raise NotImplementedError(
                f"LocalGeometry type {geometry_type} not implemented for GENE"
            )

        # Need to get convention after?
        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")
            convention = getattr(norms, self.norm_convention)

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GENE file
        """

        miller_data = default_miller_inputs()

        for (pyro_key, (gene_param, gene_key)), gene_default in zip(
            self.pyro_gene_miller.items(), self.pyro_gene_miller_default.values()
        ):
            miller_data[pyro_key] = self.data[gene_param].get(gene_key, gene_default)

        minor_r = self.data["geometry"].get("minor_r", 0.0)
        major_R = self.data["geometry"].get("major_r", 1.0)

        if minor_r == 1.0:
            self.norm_convention = "pyrokinetics"
        elif major_R == 1.0:
            self.norm_convention = "gene"
        else:
            raise ValueError(
                f"Pyrokinetics can only handle GENE simulations with either minor_r=1.0 (got {minor_r}) or major_R = 1.0 (got {major_R})"
            )

        # TODO Need to handle case where minor_r not defined
        miller_data["Rmaj"] = self.data["geometry"].get("major_r", 1.0) / self.data[
            "geometry"
        ].get("minor_r", 1.0)
        miller_data["rho"] = (
            self.data["geometry"].get("trpeps", 0.0) * miller_data["Rmaj"]
        )

        # GENE defines whether clockwise - need to flip sign
        miller_data["ip_ccw"] *= -1
        miller_data["bt_ccw"] *= -1

        # Assume pref*8pi*1e-7 = 1.0
        beta = self.data["general"]["beta"]
        if beta != 0.0:
            miller_data["B0"] = np.sqrt(1.0 / beta)
        else:
            miller_data["B0"] = None

        miller_data["beta_prime"] = -self.data["geometry"].get("amhd", 0.0) / (
            miller_data["q"] ** 2 * miller_data["Rmaj"]
        )

        dpdx = self.data["geometry"].get("dpdx_pm", -2)

        if dpdx != -2 and dpdx != -miller_data["beta_prime"]:
            if dpdx == -1:
                local_species = self.get_local_species()
                beta_prime_ratio = -miller_data["beta_prime"] / (
                    local_species.inverse_lp * beta
                )
                if not np.isclose(beta_prime_ratio, 1.0):
                    warnings.warn(
                        "GENE dpdx_pm not set consistently with amhd- drifts may not behave as expected"
                    )
            else:
                warnings.warn(
                    "GENE dpdx_pm not set consistently with amhd - drifts may not behave as expected"
                )

        miller = LocalGeometryMiller.from_gk_data(miller_data)

        return miller

    def get_local_geometry_miller_turnbull(self) -> LocalGeometryMillerTurnbull:
        """
        Load Miller object from GENE file
        """
        miller_data = default_miller_turnbull_inputs()

        for (pyro_key, (gene_param, gene_key)), gene_default in zip(
            self.pyro_gene_miller_turnbull.items(),
            self.pyro_gene_miller_turnbull_default.values(),
        ):
            miller_data[pyro_key] = self.data[gene_param].get(gene_key, gene_default)

        # TODO Need to handle case where minor_r not defined
        miller_data["Rmaj"] = self.data["geometry"].get("major_r", 1.0) / self.data[
            "geometry"
        ].get("minor_r", 1.0)
        miller_data["rho"] = (
            self.data["geometry"].get("trpeps", 0.0) * miller_data["Rmaj"]
        )

        # GENE defines whether clockwise - need to flip sign
        miller_data["ip_ccw"] *= -1
        miller_data["bt_ccw"] *= -1

        # Assume pref*8pi*1e-7 = 1.0
        beta = self.data["general"]["beta"]
        if beta != 0.0:
            miller_data["B0"] = np.sqrt(1.0 / beta)
        else:
            miller_data["B0"] = None

        miller_data["beta_prime"] = -self.data["geometry"].get("amhd", 0.0) / (
            miller_data["q"] ** 2 * miller_data["Rmaj"]
        )

        miller = LocalGeometryMillerTurnbull.from_gk_data(miller_data)

        return miller

    # Treating circular as a special case of miller
    def get_local_geometry_circular(self) -> LocalGeometryMillerTurnbull:
        """
        Load Circular object from GENE file
        """
        circular_data = default_miller_turnbull_inputs()

        for pyro_key, (gene_param, gene_key) in self.pyro_gene_circular.items():
            circular_data[pyro_key] = self.data[gene_param][gene_key]
        circular_data["local_geometry"] = "Miller"

        circular_data["Rmaj"] = self.data["geometry"].get("major_r", 1.0) / self.data[
            "geometry"
        ].get("minor_r", 1.0)
        circular_data["rho"] = (
            self.data["geometry"].get("trpeps", 0.0) * circular_data["Rmaj"]
        )

        beta = self.data["general"]["beta"]
        if beta != 0.0:
            circular_data["B0"] = np.sqrt(1.0 / beta)
        else:
            circular_data["B0"] = None

        circular = LocalGeometryMillerTurnbull.from_gk_data(circular_data)

        return circular

    def get_local_species(self):
        """
        Load LocalSpecies object from GENE file
        """
        local_species = LocalSpecies()
        ion_count = 0

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")
            if self._convention_dict:
                code_normalisation = self.norm_convention
            else:
                code_normalisation = "pyrokinetics"

            convention = getattr(norms, code_normalisation)

        gene_nu_ei = self.data["general"].get("coll", 0.0)

        external_contr = self.data.get(
            "external_contr", {"ExBrate": 0.0, "Omega0_tor": 0.0, "pfsrate": 0.0}
        )

        rho = (
            self.data["geometry"].get("trpeps", 0.0)
            * self.data["geometry"].get("major_r", 1.0)
            / self.data["geometry"].get("minor_r", 1.0)
        )
        domega_drho = -self.data["geometry"]["q0"] / rho * external_contr["pfsrate"]

        # Load each species into a dictionary
        for i_sp in range(self.data["box"]["n_spec"]):
            species_data = CleverDict()

            try:
                gene_data = self.data["species"][i_sp]
            except TypeError:
                # Case when only 1 species
                gene_data = self.data["species"]

            for pyro_key, gene_key in self.pyro_gene_species.items():
                species_data[pyro_key] = gene_data[gene_key]

            # Always force to Rmaj norm and then re-normalise to pyro after
            species_data["inverse_lt"] = gene_data["omt"]
            species_data["inverse_ln"] = gene_data["omn"]
            species_data["omega0"] = (
                external_contr["Omega0_tor"]
            )
            species_data["domega_drho"] = domega_drho

            if species_data.z == -1:
                name = "electron"
                species_data.nu = (
                    gene_nu_ei * 4 * (deuterium_mass / electron_mass) ** 0.5
                ) * convention.vref / convention.lref
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # normalisations
            species_data.dens *= convention.nref
            species_data.mass *= convention.mref
            species_data.temp *= convention.tref
            species_data.z *= convention.qref
            species_data.inverse_lt *= convention.lref**-1
            species_data.inverse_ln *= convention.lref**-1
            species_data.omega0 *= convention.vref / convention.lref
            species_data.domega_drho *= convention.vref / convention.lref**2

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        nu_ee = local_species.electron.nu
        te = local_species.electron.temp
        ne = local_species.electron.dens
        me = local_species.electron.mass

        for ion in range(ion_count):
            key = f"ion{ion + 1}"

            nion = local_species[key]["dens"]
            tion = local_species[key]["temp"]
            mion = local_species[key]["mass"]
            zion = local_species[key]["z"]
            # Not exact at log(Lambda) does change but pretty close...
            local_species[key]["nu"] = (
                nu_ee
                * (zion**4 * nion / tion**1.5 / mion**0.5)
                / (ne / te**1.5 / me**0.5)
            ).m * nu_ee.units

        local_species.zeff = (
            self.data["general"].get("zeff", 1.0) * ureg.elementary_charge
        )

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        local_species.normalise()

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")
            if self._convention_dict:
                code_normalisation = self.norm_convention
            else:
                code_normalisation = "pyrokinetics"
            convention = getattr(norms, code_normalisation)

        numerics_data = dict()

        # Set number of fields
        numerics_data["phi"] = True

        numerics_data["apar"] = bool(self.data["general"].get("beta", 0))
        numerics_data["bpar"] = bool(self.data["general"].get("bpar", 0))

        numerics_data["delta_time"] = self.data["general"].get("dt_max", 0.01)
        numerics_data["max_time"] = self.data["general"].get("simtimelim", 500.0)

        try:
            numerics_data["theta0"] = -self.data["box"]["kx_center"] / (
                self.data["box"]["kymin"] * self.data["geometry"]["shat"]
            )
        except KeyError:
            numerics_data["theta0"] = 0.0

        numerics_data["nky"] = self.data["box"]["nky0"]
        numerics_data["ky"] = self.data["box"]["kymin"]

        # Set to zero if box.lx not present
        numerics_data["kx"] = 2 * pi / self.data["box"].get("lx", np.inf)

        # Velocity grid

        numerics_data["ntheta"] = self.data["box"].get("nz0", 24)
        numerics_data["nenergy"] = self.data["box"].get("nv0", 16) // 2
        numerics_data["npitch"] = self.data["box"].get("nw0", 16)

        numerics_data["nonlinear"] = bool(self.data["general"].get("nonlinear", False))

        if numerics_data["nonlinear"]:
            numerics_data["nkx"] = self.data["box"]["nx0"]
            numerics_data["nperiod"] = 1
        else:
            numerics_data["nkx"] = 1
            numerics_data["nperiod"] = self.data["box"]["nx0"] - 1

        numerics_data["beta"] = self.data["general"]["beta"]

        external_contr = self.data.get(
            "external_contr", {"ExBrate": 0.0, "Omega0_tor": 0.0, "pfsrate": 0.0}
        )

        numerics_data["gamma_exb"] = external_contr["ExBrate"]

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in reference values from input file

        """
        if "units" not in self.data.keys():
            return {}

        if not self.data["units"].keys():
            return {}

        norms = {}

        if "minor_r" in self.data["geometry"]:
            lref_scale = self.data["geometry"]["minor_r"]
            lref_key = "minor_radius"
        elif "major_R" in self.data["geometry"]:
            lref_scale = self.data["geometry"]["major_R"]
            lref_key = "major_radius"

        norms["tref_electron"] = self.data["units"]["Tref"] * local_norm.units.keV
        norms["nref_electron"] = (
            self.data["units"]["nref"] * local_norm.units.meter**-3 * 1e19
        )
        norms["bref_B0"] = self.data["units"]["Bref"] * local_norm.units.tesla
        norms[f"lref_{lref_key}"] = (
            self.data["units"]["lref"] * local_norm.units.meter / lref_scale
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
            "lref": "major_radius",
            "ne": 1.0,
            "te": 1.0,
            "rgeo_rmaj": 1.0,
            "vref": "nrl",
            "rhoref": "pyro",
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
        for i_sp in range(self.data["box"]["n_spec"]):

            dens = self.data["species"][i_sp]["dens"]
            temp = self.data["species"][i_sp]["temp"]
            mass = self.data["species"][i_sp]["mass"]

            if self.data["species"][i_sp]["charge"] == -1:
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
            for i_sp in range(self.data["box"]["n_spec"]):
                ne += densities[i_sp] * self.data["species"][i_sp]["charge"]

            electron_density = ne
            electron_temperature = self.data["species"][0]["temp"]
            e_mass = (electron_mass / deuterium_mass).m

            densities.append(dens)
            temperatures.append(temp)
            masses.append(mass)

        minor_radius = self.data["geometry"].get("minor_r", 0.0)
        major_radius = self.data["geometry"]["major_r"]

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
            rgeo_rmaj=1.0,
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
                template_file = gk_templates["GENE"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        # TODO Find way to get norm_convention = pyrokinetics if we find minor_radius as lref
        if code_normalisation is None:
            if self.data["geometry"]["minor_r"] == 1.0:
                code_normalisation = "pyrokinetics"
            elif self.data["geometry"]["major_R"] == 1.0:
                code_normalisation = "gene"

        convention = getattr(local_norm, code_normalisation)

        # Geometry data
        if isinstance(local_geometry, LocalGeometryMillerTurnbull):
            eq_type = "MillerTurnbull"
        elif isinstance(local_geometry, LocalGeometryMiller):
            eq_type = "Miller"
        else:
            raise NotImplementedError(
                f"Writing LocalGeometry type {local_geometry.__class__.__name__} "
                "for GENE not yet supported"
            )

        self.data["geometry"]["magn_geometry"] = "miller"

        if eq_type == "MillerTurnbull":
            for pyro_key, (
                gene_param,
                gene_key,
            ) in self.pyro_gene_miller_turnbull.items():
                self.data[gene_param][gene_key] = local_geometry[pyro_key]
        elif eq_type == "Miller":
            for pyro_key, (gene_param, gene_key) in self.pyro_gene_miller.items():
                self.data[gene_param][gene_key] = local_geometry[pyro_key]

        self.data["geometry"]["amhd"] = (
            -(local_geometry.q**2) * local_geometry.Rmaj * local_geometry.beta_prime
        )
        self.data["geometry"]["dpdx_pm"] = -2

        self.data["geometry"]["trpeps"] = local_geometry.rho / local_geometry.Rmaj
        self.data["geometry"]["minor_r"] = 1.0
        self.data["geometry"]["major_r"] = local_geometry.Rmaj

        # GENE defines whether clockwise/ pyro defines whether counter-clockwise - need to flip sign
        self.data["geometry"]["sign_Ip_CW"] = -1 * local_geometry.ip_ccw
        self.data["geometry"]["sign_Bt_CW"] = -1 * local_geometry.bt_ccw

        # Kinetic data
        self.data["box"]["n_spec"] = local_species.nspec

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

            if name == "electron":
                single_species["name"] = "electron"
            else:
                single_species["name"] = "ion"

            # TODO Currently forcing GENE to use default pyro. Should check local_norm first
            for key, val in self.pyro_gene_species.items():
                single_species[val] = local_species[name][key]

        if "external_contr" not in self.data.keys():
            self.data["external_contr"] = f90nml.Namelist(
                {
                    "Omega0_tor": local_species.electron.omega0,
                    "pfsrate": -local_species.electron.domega_drho
                    * local_geometry.rho
                    / self.data["geometry"]["q0"],
                }
            )
        else:
            self.data["external_contr"]["Omega0_tor"] = local_species.electron.omega0
            self.data["external_contr"]["pfsrate"] = (
                -local_species.electron.domega_drho
                * local_geometry.rho
                / self.data["geometry"]["q0"]
            )

        self.data["general"]["zeff"] = local_species.zeff

        beta_ref = convention.beta if local_norm else 0.0
        self.data["general"]["beta"] = (
            numerics.beta if numerics.beta is not None else beta_ref
        )

        self.data["general"]["coll"] = local_species.electron.nu / (
            4 * np.sqrt(deuterium_mass / electron_mass)
        )

        # Numerics
        if numerics.bpar and not numerics.apar:
            raise ValueError("Can't have bpar without apar in GENE")

        self.data["general"]["bpar"] = numerics.bpar

        # FIXME breaks a roundtrip when doing electrostatic simulations
        # FIXME can't really fix this due to GENE set up...
        if not numerics.apar:
            self.data["general"]["beta"] = 0.0

        self.data["general"]["dt_max"] = numerics.delta_time
        self.data["general"]["simtimelim"] = numerics.max_time

        if numerics["nonlinear"]:
            # TODO Currently forces NL sims to have nperiod = 1
            self.data["general"]["nonlinear"] = True
            self.data["box"]["nky0"] = numerics["nky"]
            self.data["box"]["nx0"] = numerics["nkx"]
        else:
            self.data["general"]["nonlinear"] = False

        self.data["box"]["nky0"] = numerics.nky
        self.data["box"]["kymin"] = numerics.ky

        self.data["box"]["kx_center"] = (
            -1 * numerics.theta0 * numerics.ky * local_geometry.shat
        )

        if numerics.kx != 0.0:
            self.data["box"]["lx"] = 2 * pi / numerics.kx

        self.data["box"]["nz0"] = numerics.ntheta
        self.data["box"]["nv0"] = 2 * numerics.nenergy
        self.data["box"]["nw0"] = numerics.npitch

        if "external_contr" not in self.data.keys():
            self.data["external_contr"] = f90nml.Namelist(
                {"ExBrate": numerics.gamma_exb}
            )
        else:
            self.data["external_contr"]["ExBrate"] = numerics.gamma_exb

        if not local_norm:
            return

        try:
            (1 * convention.tref).to("keV")
            si_units = True
        except pint.errors.DimensionalityError:
            si_units = False

        if si_units:
            if "units" not in self.data.keys():
                self.data["units"] = f90nml.Namelist()

            self.data["units"]["Tref"] = (1 * convention.tref).to("keV").m
            self.data["units"]["nref"] = (1e-19 * convention.nref).to("meter**-3").m
            self.data["units"]["mref"] = (1 * convention.mref).to("proton_mass").m
            self.data["units"]["Bref"] = (1 * convention.bref).to("tesla").m
            self.data["units"]["Lref"] = (1 * convention.lref).to("meter").m
            self.data["units"]["omegatorref"] = (
                local_species.electron.omega0.to(convention).to("radians/second").m
            )

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, convention)

    def get_ne_te_normalisation(self):
        adiabatic_electrons = True
        # Get electron temp and density to normalise input
        for i_sp in range(self.data["box"]["n_spec"]):
            if self.data["species"][i_sp]["charge"] == -1:
                ne = self.data["species"][i_sp]["dens"]
                Te = self.data["species"][i_sp]["temp"]
                adiabatic_electrons = False

        if adiabatic_electrons:
            ne = 0.0
            for i_sp in range(self.data["box"]["n_spec"]):
                ne += (
                    self.data["species"][i_sp]["dens"]
                    * self.data["species"][i_sp]["charge"]
                )

            Te = self.data["species"][0]["temp"]

        return ne, Te


class GKOutputReaderGENE(FileReader, file_type="GENE", reads=GKOutput):
    fields = ["phi", "apar", "bpar"]

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
        # Determine normalisation used
        nml = gk_input.data
        if nml["geometry"].get("minor_r", 0.0) == 1.0:
            convention = norm.pyrokinetics
            norm.default_convention = output_convention.lower()
        elif gk_input.data["geometry"].get("major_R", 1.0) == 1.0:
            convention = norm.gene
            norm.default_convention = "gene"
        else:
            raise NotImplementedError(
                "Pyro does not handle GENE cases where neither major_R and minor_r are 1.0"
            )

        coords = self._get_coords(raw_data, gk_input, downsize)
        fields = self._get_fields(raw_data, gk_input, coords) if load_fields else None
        fluxes = self._get_fluxes(raw_data, coords) if load_fluxes else None
        moments = (
            self._get_moments(raw_data, gk_input, coords) if load_moments else None
        )

        if coords["linear"] and not fields:
            eigenvalues = self._get_eigenvalues(raw_data, coords)
        else:
            # Rely on gk_output to generate eigenvalues
            eigenvalues = None

        # Assign units and return GKOutput
        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = ("field", "species", "time")
        moment_dims = ("theta", "kx", "species", "ky", "time")
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
            linear=coords["linear"],
            gk_code="GENE",
            input_file=input_str,
            normalise_flux_moment=True,
            output_convention=output_convention,
        )

    @staticmethod
    def _get_gene_files(filename: PathLike) -> Dict[str, Path]:
        """
        Given a directory name, looks for the files filename/parameters_0000,
        filename/field_0000 and filename/nrg_0000.
        If instead given any of the files parameters_####, field_#### or nrg_####,
        looks up the rest of the files in the same directory.
        """
        filename = Path(filename)
        prefixes = ["parameters", "field", "nrg", "omega"]
        if filename.is_dir():
            # If given a dir name, looks for dir/parameters_0000
            dirname = filename
            dat_matches = np.any(
                [Path(filename / f"{p}.dat").is_file() for p in prefixes]
            )
            if dat_matches:
                suffix = "dat"
                delimiter = "."
            else:
                suffix = "0000"
                delimiter = "_"
        else:
            # If given a file, searches for all similar GENE files in that file's dir
            dirname = filename.parent
            # Ensure provided file is a GENE file (fr"..." means raw format str)
            matches = [re.search(rf"^{p}_\d{{4}}$", filename.name) for p in prefixes]
            if not np.any(matches):
                raise RuntimeError(
                    f"GKOutputReaderGENE: The provided file {filename} is not a GENE "
                    "output file."
                )
            suffix = filename.name.split("_")[1]
            delimiter = "_"

        # Get all files in the same dir
        files = {
            prefix: dirname / f"{prefix}{delimiter}{suffix}"
            for prefix in prefixes
            if (dirname / f"{prefix}{delimiter}{suffix}").exists()
        }

        if not files:
            raise RuntimeError(
                "GKOutputReaderGENE: Could not find GENE output files in the "
                f"directory '{dirname}'."
            )
        if "parameters" not in files:
            raise RuntimeError(
                "GKOutputReaderGENE: Could not find GENE output file 'parameters_"
                f"{suffix}' when provided with the file/directory '{filename}'."
            )
        # If binary field file absent, adds .h5 field file,
        # if present, to 'files'
        if "field" not in files:
            if (dirname / f"field{delimiter}{suffix}.h5").exists():
                files.update({"field": dirname / f"field{delimiter}{suffix}.h5"})
        return files

    @staticmethod
    def _get_gene_mom_files(
        filename: PathLike, files: Dict, species_names
    ) -> Dict[str, Path]:
        """
        Given a directory name, looks for the files filename/parameters_0000,
        filename/field_0000 and filename/nrg_0000.
        If instead given any of the files parameters_####, field_#### or nrg_####,
        looks up the rest of the files in the same directory.
        """
        filename = Path(filename)
        prefixes = [f"mom_{species_name}" for species_name in species_names]
        if filename.is_dir():
            # If given a dir name, looks for dir/parameters_0000
            dirname = filename
            dat_matches = np.any(
                [Path(filename / f"{p}.dat").is_file() for p in prefixes]
            )
            if dat_matches:
                suffix = "dat"
                delimiter = "."
            else:
                suffix = "0000"
                delimiter = "_"
        else:
            # If given a file, searches for all similar GENE files in that file's dir
            dirname = filename.parent
            suffix = filename.name.split("_")[-1]
            delimiter = "_"

        # Get all files in the same dir
        for prefix in prefixes:
            if (dirname / f"{prefix}{delimiter}{suffix}").exists():
                files[prefix] = dirname / f"{prefix}{delimiter}{suffix}"

        return files

    def verify_file_type(self, filename: PathLike):
        self._get_gene_files(filename)

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        """
        # If the input file is of the form name_####, get the numbered part and
        # search for 'parameters_####' in the run directory. If not, simply return
        # the directory.
        filename = Path(filename)
        num_part_regex = re.compile(r"(\d{4})")
        num_part_match = num_part_regex.search(filename.name)

        if num_part_match is None:
            return Path(filename).parent
        else:
            return Path(filename).parent / f"parameters_{num_part_match[0]}"
        pass

    @classmethod
    def _get_raw_data(
        cls, filename: PathLike
    ) -> Tuple[Dict[str, Any], GKInputGENE, str]:
        files = cls._get_gene_files(filename)
        # Read parameters_#### as GKInputGENE and into plain string
        with open(files["parameters"], "r") as f:
            input_str = f.read()
        gk_input = GKInputGENE()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        species_names = [species["name"] for species in gk_input.data["species"]]
        files = cls._get_gene_mom_files(filename, files, species_names)
        # Defer processing field and flux data until their respective functions
        # Simply return files in place of raw data
        return files, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, Any], gk_input: GKInputGENE, downsize: int
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a GENE parameters file.

        Args:
            raw_data (Dict[str,Any]): Dict containing GENE output. Ignored.
            gk_input (GKInputGENE): Processed GENE input file.

        Returns:
            xr.Dataset: Dataset with coords and attrs set, but not data_vars
        """
        nml = gk_input.data

        # The last time step is not always written, but depends on
        # whatever condition is met first between simtimelim and timelim
        species = gk_input.get_local_species().names
        with open(raw_data["nrg"], "r") as f:
            full_data = f.readlines()
            ntime = len(full_data) // (len(species) + 1)
            lasttime = float(full_data[-(len(species) + 1)])

        if ntime * nml["in_out"]["istep_nrg"] % nml["in_out"]["istep_field"] == 0:
            add_on = 0
        else:
            add_on = 1

        ntime = (
            int(ntime * nml["in_out"]["istep_nrg"] / nml["in_out"]["istep_field"])
        ) + add_on

        ntime = ntime // downsize

        # Set time to index for now, gets overwritten by field data
        time = np.linspace(0, ntime - 1, ntime)

        nfield = nml["info"]["n_fields"]
        field = ["phi", "apar", "bpar"][:nfield]

        nky = nml["box"]["nky0"]
        nkx = nml["box"]["nx0"]
        ntheta = nml["box"]["nz0"]
        theta = np.linspace(-pi, pi, ntheta, endpoint=False)

        nenergy = nml["box"]["nv0"]
        energy = np.linspace(-1, 1, nenergy)

        npitch = nml["box"]["nw0"]
        pitch = np.linspace(-1, 1, npitch)

        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]

        if gk_input.is_linear():
            # Set up ballooning angle
            single_theta_loop = theta
            single_ntheta_loop = ntheta

            ntheta = ntheta * (nkx - 1)
            theta = np.empty(ntheta)
            start = 0
            for i in range(nkx - 1):
                pi_segment = i - nkx // 2 + 1
                theta[start : start + single_ntheta_loop] = (
                    single_theta_loop + pi_segment * 2 * pi
                )
                start += single_ntheta_loop

            ky = [nml["box"]["kymin"]]
            kx = [0.0]
            nkx = 1
            # TODO should we not also set nky=1?

        else:
            kymin = nml["box"]["kymin"]
            ky = np.linspace(0, kymin * (nky - 1), nky)
            lx = nml["box"]["lx"]
            dkx = 2 * np.pi / lx
            kx = np.empty(nkx)
            for i in range(nkx):
                if i < (nkx / 2 + 1):
                    kx[i] = i * dkx
                else:
                    kx[i] = (i - nkx) * dkx

            kx = np.roll(np.fft.fftshift(kx), -1)

        # Convert to Pyro coordinate (need magnitude to set up Dataset)

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "moment": moments,
            "flux": fluxes,
            "field": field,
            "species": species,
            "downsize": downsize,
            "linear": gk_input.is_linear(),
            "lasttime": lasttime,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any],
        gk_input: GKInputGENE,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """

        if "field" not in raw_data:
            return {}

        # Time data stored as binary (int, double, int)
        time = []
        time_data_fmt = "=idi"
        time_data_size = struct.calcsize(time_data_fmt)

        int_size = 4
        complex_size = 16

        downsize = coords["downsize"]

        nx = gk_input.data["box"]["nx0"]
        nz = gk_input.data["box"]["nz0"]

        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])
        nfield = len(coords["field"])

        field_size = nx * nz * nky * complex_size

        sliced_field = np.empty((nfield, nx, nky, nz, ntime), dtype=complex)
        fields = np.empty((nfield, nkx, nky, ntheta, ntime), dtype=complex)
        # Read binary file if present
        if ".h5" not in str(raw_data["field"]):
            with open(raw_data["field"], "rb") as file:
                for i_time in range(ntime):
                    # Read in time data (stored as int, double int)
                    time_value = float(
                        struct.unpack(time_data_fmt, file.read(time_data_size))[1]
                    )
                    time.append(time_value)
                    for i_field in range(nfield):
                        file.seek(int_size, 1)
                        binary_field = file.read(field_size)
                        raw_field = np.frombuffer(binary_field, dtype=np.complex128)
                        sliced_field[i_field, :, :, :, i_time] = raw_field.reshape(
                            (nx, nky, nz),
                            order="F",
                        )
                        file.seek(int_size, 1)
                    if i_time < ntime - 1:
                        file.seek(
                            (downsize - 1)
                            * (time_data_size + nfield * (2 * int_size + field_size)),
                            1,
                        )

        # Read .h5 file if binary file absent
        else:
            h5_field_subgroup_names = ["phi", "A_par", "B_par"]
            fields = np.empty(
                (nfield, nkx, nky, ntheta, ntime),
                dtype=complex,
            )
            with h5py.File(raw_data["field"], "r") as file:
                # Read in time data
                time.extend(list(file.get("field/time")))
                for i_field in range(nfield):
                    h5_subgroup = "field/" + h5_field_subgroup_names[i_field] + "/"
                    h5_dataset_names = list(file[h5_subgroup].keys())
                    for i_time in range(ntime):
                        h5_dataset = h5_subgroup + h5_dataset_names[i_time]
                        raw_field = np.array(file.get(h5_dataset))
                        raw_field = np.array(
                            raw_field["real"] + raw_field["imaginary"] * 1j,
                            dtype="complex128",
                        )
                        sliced_field[i_field, :, :, :, i_time] = np.swapaxes(
                            raw_field, 0, 2
                        )

        # Match pyro convention for ion/electron direction
        sliced_field = np.conjugate(sliced_field)

        if not gk_input.is_linear():
            nl_shape = (nfield, nkx, nky, ntheta, ntime)
            fields = sliced_field.reshape(nl_shape, order="F")

        # Convert from kx to ballooning space
        else:
            try:
                n0_global = gk_input.data["box"]["n0_global"]
                q0 = gk_input.data["geometry"]["q0"]
                phase_fac = -np.exp(-2 * np.pi * 1j * n0_global * q0)
            except KeyError:
                phase_fac = -1
            i_ball = 0

            for i_conn in range(-int(nx / 2) + 1, int((nx - 1) / 2) + 1):
                fields[:, 0, :, i_ball : i_ball + nz, :] = (
                    sliced_field[:, i_conn, :, :, :] * (phase_fac) ** i_conn
                )
                i_ball += nz

        # =================================================

        # Overwrite 'time' coordinate as determined in _init_dataset
        coords["time"] = time

        # Original method coords: (field, kx, ky, theta, time)
        # New coords: (field, theta, kx, ky, time)
        fields = fields.transpose(0, 3, 1, 2, 4)

        # Shift kx component to middle of array
        fields = np.roll(np.fft.fftshift(fields, axes=2), -1, axis=2)

        result = {}

        for ifield, field_name in enumerate(coords["field"]):
            result[field_name] = fields[ifield, ...]

        return result

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputGENE,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """

        if "mom_electron" not in raw_data:
            return {}

        # Time data stored as binary (int, double, int)
        time = []
        time_data_fmt = "=idi"
        time_data_size = struct.calcsize(time_data_fmt)

        int_size = 4
        complex_size = 16

        downsize = coords["downsize"]

        nx = gk_input.data["box"]["nx0"]
        nz = gk_input.data["box"]["nz0"]

        nkx = len(coords["kx"])
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntime = len(coords["time"])

        species = [species["name"] for species in gk_input.data["species"]]
        nspecies = len(species)

        nmoment_output = 6
        if len(coords["field"]) > 2:
            nmoment_output += 3

        moment_size = nx * nz * nky * complex_size

        sliced_moment = np.empty(
            (nspecies, nmoment_output, nx, nky, nz, ntime), dtype=complex
        )
        moments = np.empty(
            (nspecies, nmoment_output, nkx, nky, ntheta, ntime), dtype=complex
        )
        for i_sp, spec in enumerate(species):
            # Read binary file if present
            if ".h5" not in str(raw_data[f"mom_{spec}"]):
                with open(raw_data[f"mom_{spec}"], "rb") as file:
                    for i_time in range(ntime):
                        # Read in time data (stored as int, double int)
                        time_value = float(
                            struct.unpack(time_data_fmt, file.read(time_data_size))[1]
                        )
                        if i_sp == 0:
                            time.append(time_value)
                        for i_moment in range(nmoment_output):
                            file.seek(int_size, 1)
                            binary_moment = file.read(moment_size)
                            raw_moment = np.frombuffer(
                                binary_moment, dtype=np.complex128
                            )
                            sliced_moment[i_sp, i_moment, :, :, :, i_time] = (
                                raw_moment.reshape(
                                    (nx, nky, nz),
                                    order="F",
                                )
                            )
                            file.seek(int_size, 1)
                        if i_time < ntime - 1:
                            file.seek(
                                (downsize - 1)
                                * (
                                    time_data_size
                                    + nmoment_output * (2 * int_size + moment_size)
                                ),
                                1,
                            )

            # Read .h5 file if binary file absent
            else:
                raise NotImplementedError("Moments from HDf5 not yet supported")

            # Match pyro convention for ion/electron direction
            sliced_moment = np.conjugate(sliced_moment)

            if not gk_input.is_linear():
                nl_shape = (nspecies, nmoment_output, nkx, nky, ntheta, ntime)
                moments = sliced_moment.reshape(nl_shape, order="F")

            # Convert from kx to ballooning space
            else:
                try:
                    n0_global = gk_input.data["box"]["n0_global"]
                    q0 = gk_input.data["geometry"]["q0"]
                    phase_fac = -np.exp(-2 * np.pi * 1j * n0_global * q0)
                except KeyError:
                    phase_fac = -1
                i_ball = 0

                for i_conn in range(-int(nx / 2) + 1, int((nx - 1) / 2) + 1):
                    moments[:, 0, :, i_ball : i_ball + nz, :] = (
                        sliced_moment[:, i_conn, :, :, :] * (phase_fac) ** i_conn
                    )
                    i_ball += nz

        # =================================================

        # Overwrite 'time' coordinate as determined in _init_dataset
        coords["time"] = time

        # Original method coords: (species, moment, kx, ky, theta, time)
        # New coords: (moment, theta, kx, species, ky, time)
        moments = moments.transpose(1, 4, 2, 0, 3, 5)

        # Shift kx component to middle of array
        moments = np.roll(np.fft.fftshift(moments, axes=2), -1, axis=2)

        result = {}

        result["density"] = moments[0, ...]
        result["temperature"] = moments[1, ...] / 3 + moments[2, ...] * 2 / 3
        result["velocity"] = moments[5, ...]

        return result

    @staticmethod
    def _get_fluxes(
        raw_data: Dict[str, Any], coords: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Set flux data over time.
        The flux coordinates should  be (species, flux, field, ky, time)
        """

        # ky data not available in the nrg file so no ky coords here
        coord_names = ["species", "flux", "field", "time"]
        shape = [len(coords[coord_name]) for coord_name in coord_names]
        fluxes = np.empty(shape)

        nfield = len(coords["field"])
        nspecies = len(coords["species"])
        ntime = len(coords["time"])

        if "nrg" not in raw_data:
            logging.warning("Flux data not found, setting all fluxes to zero")
            fluxes[...] = 0
            result = {"fluxes": fluxes}
            return result

        nml = f90nml.read(raw_data["parameters"])
        flux_istep = nml["in_out"]["istep_nrg"]
        field_istep = nml["in_out"]["istep_field"]

        ntime_flux = nml["info"]["steps"][0] // flux_istep + 1
        if nml["info"]["steps"][0] % flux_istep > 0:
            ntime_flux += 1

        downsize = coords["downsize"]

        if flux_istep < field_istep:
            time_skip = int(field_istep * downsize / flux_istep) - 1
        else:
            time_skip = downsize - 1

        with open(raw_data["nrg"], "r") as csv_file:
            nrg_data = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)

            if nfield == 3:
                logging.warning(
                    "GENE combines Apar and Bpar fluxes, setting Bpar fluxes to zero"
                )
                fluxes[:, :, 2, :] = 0.0
                field_size = 2
            else:
                field_size = nfield

            for i_time in range(ntime):
                time = next(nrg_data)  # noqa
                coords["time"][i_time] = float(time[0])
                for i_species in range(nspecies):
                    nrg_line = np.array(next(nrg_data), dtype=float)

                    # Particle
                    fluxes[i_species, 0, :field_size, i_time] = nrg_line[
                        4 : 4 + field_size,
                    ]

                    # Heat
                    fluxes[i_species, 1, :field_size, i_time] = nrg_line[
                        6 : 6 + field_size,
                    ]

                    # Momentum
                    fluxes[i_species, 2, :field_size, i_time] = nrg_line[
                        8 : 8 + field_size,
                    ]

                # Skip time/data values in field print out is less
                if i_time < ntime - 1:
                    for skip_t in range(time_skip):
                        for skip_s in range(nspecies + 1):
                            next(nrg_data)

        results = {}

        fluxes = fluxes.transpose(1, 2, 0, 3)

        for iflux, flux in enumerate(coords["flux"]):
            results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: Dict[str, Any], coords: Dict
    ) -> Dict[str, np.ndarray]:
        """

        Parameters
        ----------
        raw_data
        coords

        Returns
        -------
        Dict of eigenvalues with coords (kx, ky, time)
            Only final time is output so we set that to all the times
        """

        nky = len(coords["ky"])
        nkx = len(coords["kx"])
        ntime = len(coords["time"])
        mode_frequency = np.empty((nkx, nky, ntime))
        growth_rate = np.empty((nkx, nky, ntime))

        with open(raw_data["omega"], "r") as csv_file:
            omega_data = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)
            for iky, line in enumerate(omega_data):
                ky, growth, frequency = line

                mode_frequency[:, iky, :] = float(frequency)
                growth_rate[:, iky, :] = float(growth)

        results = {"growth_rate": growth_rate, "mode_frequency": mode_frequency}

        return results
