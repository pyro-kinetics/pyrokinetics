from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from cleverdict import CleverDict

from ..file_utils import FileReader
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryFourierCGYRO,
    LocalGeometryMiller,
    LocalGeometryMXH,
    default_fourier_cgyro_inputs,
    default_miller_inputs,
    default_mxh_inputs,
)
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..normalisation import convert_dict
from ..numerics import Numerics
from ..templates import gk_templates
from ..typing import PathLike
from ..units import PyroContextError, PyroNormalisationError
from .gk_input import GKInput
from .gk_output import GKOutput


class GKInputNEO(GKInput, FileReader, file_type="NEO", reads=GKInput):
    """
    Class that can read NEO input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "NEO"
    default_file_name = "input.neo"
    norm_convention = "neo"
    _convention_dict = {}

    pyro_neo_miller = {
        "rho": "RMIN_OVER_A",
        "Rmaj": "RMAJ_OVER_A",
        "q": "Q",
        "kappa": "KAPPA",
        "s_kappa": "S_KAPPA",
        "delta": "DELTA",
        "s_delta": "S_DELTA",
        "shat": "SHEAR",
        "shift": "SHIFT",
        "Z0": "ZMAG_OVER_A",
        "dZ0dr": "S_ZMAG",
        "ip_ccw": "IPCCW",
        "bt_ccw": "BTCCW",
        "beta_prime": "BETA_STAR",
    }

    pyro_neo_mxh = {
        **pyro_neo_miller,
        "zeta": "ZETA",
        "s_zeta": "S_ZETA",
        "cn0": "SHAPE_COS0",
        "cn1": "SHAPE_COS1",
        "cn2": "SHAPE_COS2",
        "cn3": "SHAPE_COS3",
        "cn4": "SHAPE_COS4",
        "cn5": "SHAPE_COS5",
        "cn6": "SHAPE_COS6",
        "sn3": "SHAPE_SIN3",
        "sn4": "SHAPE_SIN4",
        "sn5": "SHAPE_SIN5",
        "sn6": "SHAPE_SIN6",
        "dcndr0": "SHAPE_S_COS0",
        "dcndr1": "SHAPE_S_COS1",
        "dcndr2": "SHAPE_S_COS2",
        "dcndr3": "SHAPE_S_COS3",
        "dcndr4": "SHAPE_S_COS4",
        "dcndr5": "SHAPE_S_COS5",
        "dcndr6": "SHAPE_S_COS6",
        "dsndr3": "SHAPE_S_SIN3",
        "dsndr4": "SHAPE_S_SIN4",
        "dsndr5": "SHAPE_S_SIN5",
        "dsndr6": "SHAPE_S_SIN6",
    }

    pyro_neo_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "q": 2.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shat": 1.0,
        "shift": 0.0,
        "Z0": 0.0,
        "dZ0dr": 0.0,
        "ip_ccw": -1.0,
        "bt_ccw": -1.0,
    }

    pyro_neo_mxh_defaults = {
        **pyro_neo_miller_defaults,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "cn0": 0.0,
        "cn1": 0.0,
        "cn2": 0.0,
        "cn3": 0.0,
        "cn4": 0.0,
        "cn5": 0.0,
        "cn6": 0.0,
        "sn3": 0.0,
        "sn4": 0.0,
        "sn5": 0.0,
        "sn6": 0.0,
        "dcndr0": 0.0,
        "dcndr1": 0.0,
        "dcndr2": 0.0,
        "dcndr3": 0.0,
        "dcndr4": 0.0,
        "dcndr5": 0.0,
        "dcndr6": 0.0,
        "dsndr3": 0.0,
        "dsndr4": 0.0,
        "dsndr5": 0.0,
        "dsndr6": 0.0,
    }

    pyro_neo_fourier = pyro_neo_miller

    pyro_neo_fourier_defaults = pyro_neo_miller_defaults

    @staticmethod
    def get_pyro_neo_species(iSp=1):
        return {
            "mass": f"MASS_{iSp}",
            "z": f"Z_{iSp}",
            "dens": f"DENS_{iSp}",
            "temp": f"TEMP_{iSp}",
            "inverse_lt": f"DLNTDR_{iSp}",
            "inverse_ln": f"DLNNDR_{iSp}",
        }

    neo_eq_types = {
        1: "SAlpha",
        2: "MXH",
        3: "Fourier",
    }

    def read_from_file(
        self, filename: PathLike, detect_norm: bool = True
    ) -> Dict[str, Any]:
        """
        Reads NEO input file into a dictionary
        """
        with open(filename) as f:
            data_dict = self.parse_neo(f)
        return super().read_dict(data_dict, detect_norm=detect_norm)

    def read_str(self, input_string: str, detect_norm: bool = True) -> Dict[str, Any]:
        """
        Reads NEO input file given as string
        """
        data_dict = self.parse_neo(input_string.split("\n"))
        return super().read_dict(data_dict, detect_norm=detect_norm)

    def read_dict(self, input_dict: dict, detect_norm: bool = True) -> Dict[str, Any]:
        """
        Reads GENE input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict, detect_norm=detect_norm)

    @staticmethod
    def parse_neo(lines):
        """
        Given lines of a neo file or a string split by '/n', return a dict of
        NEO input data
        """
        results = {}
        for line in lines:
            # Get line before comments, remove trailing whitespace
            line = line.split("#")[0].strip()
            # Skip empty lines (this will also skip comment lines)
            if not line:
                continue

            # Splits by =, remove whitespace, store as (key,value) pair
            key, value = (token.strip() for token in line.split("="))

            # Use literal_eval to convert value to int/float/list etc
            # If it fails, assume value should be a string
            try:
                results[key] = literal_eval(value)
            except Exception:
                results[key] = value
        return results

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid neo input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a NEO input file,
        # but they are needed by Pyrokinetics
        expected_keys = [
            "N_SPECIES",
            "NU_1",
            "N_RADIAL",
            "RMIN_OVER_A",
            "RMAJ_OVER_A",
            "Q",
            "SHEAR",
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

        if local_norm is None:
            local_norm = Normalisation("write")

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        self.data = convert_dict(self.data, convention)

        with open(filename, "w") as f:
            for key, value in self.data.items():
                if isinstance(value, float):
                    line = f"{key} = {value:{float_format}}\n"
                else:
                    line = f"{key} = {value}\n"
                f.write(line)

    def is_nonlinear(self) -> bool:
        return bool(self.data.get("NONLINEAR_FLAG", 0))

    def add_flags(self, flags) -> None:
        """
        Add extra flags to NEO input file
        """
        for key, value in flags.items():
            self.data[key] = value

    def get_local_geometry(self) -> LocalGeometry:
        """
        Returns local geometry. Delegates to more specific functions
        """

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_geometry")
            convention = getattr(norms, self.norm_convention)

        eq_type = self.neo_eq_types[self.data.get("EQUILIBRIUM_MODEL", 0)]

        is_basic_miller = self._check_basic_miller()
        if eq_type == "MXH" and is_basic_miller:
            eq_type = "Miller"

        if eq_type == "Miller":
            local_geometry = self.get_local_geometry_miller()
        elif eq_type == "MXH":
            local_geometry = self.get_local_geometry_mxh()
        elif eq_type == "Fourier":
            local_geometry = self.get_local_geometry_fourier()
        else:
            raise NotImplementedError(
                f"LocalGeometry type {eq_type} not implemented for NEO"
            )

        local_geometry.B0 = 1.0 / local_geometry.bunit_over_b0
        local_geometry.dpsidr *= local_geometry.B0

        local_geometry.normalise(norms=convention)

        local_geometry.Fpsi = local_geometry.get_f_psi()
        local_geometry.FF_prime = local_geometry.get_f_prime() * local_geometry.Fpsi

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from NEO file
        """
        miller_data = default_miller_inputs()

        for (key, val), val_default in zip(
            self.pyro_neo_miller.items(),
            self.pyro_neo_miller_defaults.values(),
        ):
            miller_data[key] = self.data.get(val, val_default)

        miller_data["s_delta"] *= 1.0 / np.sqrt(1 - miller_data["delta"] ** 2)

        miller = LocalGeometryMiller.from_gk_data(miller_data)

        return miller

    def get_local_geometry_mxh(self) -> LocalGeometryMXH:
        """
        Load MXH object from NEO file
        """
        mxh_data = default_mxh_inputs(n_moments=7)

        for (key, val), default in zip(
            self.pyro_neo_mxh.items(), self.pyro_neo_mxh_defaults.values()
        ):
            if "SHAPE" not in val:
                mxh_data[key] = self.data.get(val, default)
            else:
                index = int(key[-1])
                new_key = key[:-1]
                if "SHAPE_S" in val:
                    mxh_data[new_key][index] = (
                        self.data.get(val, default) / mxh_data["rho"]
                    )
                else:
                    mxh_data[new_key][index] = self.data.get(val, default)

        mxh_keys = ["cn", "sn", "dcndr", "dsndr"]
        for i_moment in range(6, 2, -1):
            if np.all(
                [True if mxh_data[key][i_moment] == 0.0 else False for key in mxh_keys]
            ):
                for key in mxh_keys:
                    mxh_data[key] = mxh_data[key][:-1]
            else:
                break

        # Force dsndr[0] = 0 as is definition
        mxh_data["dsndr"][0] = 0.0

        mxh_data["n_moments"] = len(mxh_data["cn"])

        mxh = LocalGeometryMXH.from_gk_data(mxh_data)

        mxh.dthetaR_dr = mxh.get_dthetaR_dr(mxh.theta, mxh.dcndr, mxh.dsndr)

        return mxh

    def get_local_geometry_fourier(self) -> LocalGeometryFourierCGYRO:
        """
        Load Fourier object from NEO file
        """
        fourier_data = default_fourier_cgyro_inputs()

        for (key, val), val_default in zip(
            self.pyro_neo_fourier.items(), self.pyro_neo_fourier_defaults.values()
        ):
            fourier_data[key] = self.data.get(val, val_default)

        fourier = LocalGeometryFourierCGYRO.from_gk_data(fourier_data)

        return fourier

    def get_local_species(self):
        """
        Load LocalSpecies object from NEO file
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
        for i_sp in range(self.data["N_SPECIES"]):
            pyro_neo_species = self.get_pyro_neo_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_neo_species.items():
                species_data[p_key] = self.data[c_key]

            species_data.omega0 = self.data.get("MACH", 0.0) / self.data["RMAJ_OVER_A"]
            species_data.domega_drho = (
                -self.data.get("GAMMA_P", 0.0) / self.data["RMAJ_OVER_A"]
            )

            if species_data.z == -1:
                name = "electron"
                species_data.nu = (
                    self.data.get("NU_1", 0.1) * convention.vref / convention.lref
                )
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

        # Handle adiabatic species if there
        if self.data.get("AE_FLAG", 0) == 1:
            nu_ee = self.data.get("NU_1", 0.1) * convention.vref / convention.lref
            te = self.data.get("TEMP_AE", 1.0) * convention.tref
            ne = self.data.get("DENS_AE", 1.0) * convention.nref
            me = self.data.get("MASS_AE", 2.724486e-4) * convention.mref
        else:
            nu_ee = local_species.electron.nu
            te = local_species.electron.temp
            ne = local_species.electron.dens
            me = local_species.electron.mass

        # Get collision frequency of ion species
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

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        local_species.normalise(convention)

        if self.data.get("Z_EFF_METHOD", 2) == 2:
            local_species.set_zeff()
        else:
            local_species.zeff = self.data.get("Z_EFF", 1.0) * convention.qref

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_numerics")
            convention = getattr(norms, self.norm_convention)

        numerics_data = {}

        numerics_data["phi"] = True
        numerics_data["apar"] = False
        numerics_data["bpar"] = False

        numerics_data["nkx"] = self.data.get("N_RADIAL", 1)
        numerics_data["nperiod"] = 1
        numerics_data["ntheta"] = self.data.get("N_THETA", 24)
        numerics_data["nenergy"] = self.data.get("N_ENERGY", 8)
        numerics_data["npitch"] = self.data.get("N_XI", 16)
        numerics_data["gamma_exb"] = self.data.get("OMEGA_ROT_DERIV", 0.0)

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
            "bref": "Bunit",
            "lref": "minor_radius",
            "ne": 1.0,
            "te": 1.0,
            "rgeo_rmaj": 1.0,
            "vref": "nrl",
            "rhoref": "unit",
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

        if self.data.get("AE_FLAG", 0) == 1:

            dens = self.data["DENS_AE"]
            temp = self.data["TEMP_AE"]
            mass = self.data["MASS_AE"]
            electron_density = dens
            electron_temperature = temp
            e_mass = mass
            electron_index = 0
            found_electron = True

            if np.isclose(dens, 1.0):
                reference_density_index.append(0)
            if np.isclose(temp, 1.0):
                reference_temperature_index.append(0)

            densities.append(dens)
            temperatures.append(temp)
            masses.append(mass)

        else:
            for i_sp in range(self.data["N_SPECIES"]):
                dens = self.data[f"DENS_{i_sp + 1}"]
                temp = self.data[f"TEMP_{i_sp + 1}"]
                mass = self.data[f"MASS_{i_sp + 1}"]

                if self.data[f"Z_{i_sp + 1}"] == -1:
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

        major_radius = self.data["RMAJ_OVER_A"]
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
            rgeo_rmaj=1.0,
            minor_radius=minor_radius,
        )

    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        local_norm: Optional[Normalisation] = None,
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
                template_file = gk_templates["NEO"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

        # Geometry data
        if isinstance(local_geometry, LocalGeometryMXH):
            eq_model = 2
            eq_type = "MXH"
        elif isinstance(local_geometry, LocalGeometryMiller):
            eq_model = 2
            eq_type = "Miller"
        elif isinstance(local_geometry, LocalGeometryFourierCGYRO):
            eq_model = 3
            eq_type = "Fourier"
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} not "
                "implemented yet for NEO"
            )
        else:
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} not "
                "implemented for NEO"
            )

        # Set equilibrium type in input file
        self.data["EQUILIBRIUM_MODEL"] = eq_model

        if eq_type == "Miller":
            # Assign Miller values to input file
            for key, val in self.pyro_neo_miller.items():
                self.data[val] = local_geometry[key]

            self.data["S_DELTA"] = local_geometry.s_delta * np.sqrt(
                1 - local_geometry.delta**2
            )

            # Need to remove any MXH keys
            for mxh_key in self.pyro_neo_mxh.keys():
                if (
                    mxh_key not in self.pyro_neo_miller.keys()
                    and mxh_key.upper() in self.data.keys()
                ):
                    self.data.pop(mxh_key.upper())

        elif eq_type == "Fourier":
            # Assign Fourier values to input file
            for key, val in self.pyro_neo_fourier.items():
                self.data[val] = local_geometry[key]

        elif eq_type == "MXH":
            # Assign MXH values to input file
            for key, val in self.pyro_neo_mxh.items():
                if "SHAPE" not in val:
                    self.data[val] = getattr(local_geometry, key)
                else:
                    index = int(key[-1])
                    new_key = key[:-1]

                    # Skip in index beyond n_moments
                    if index >= local_geometry.n_moments:
                        continue

                    if "SHAPE_S" in val:
                        self.data[val] = (
                            getattr(local_geometry, new_key)[index] * local_geometry.rho
                        )
                    else:
                        self.data[val] = getattr(local_geometry, new_key)[index]

        # Kinetic data
        n_species = local_species.nspec
        self.data["N_SPECIES"] = n_species

        stored_species = len([key for key in self.data.keys() if "DENS_" in key])
        extra_species = stored_species - n_species

        if extra_species > 0:
            for i_sp in range(extra_species):
                pyro_neo_species = self.get_pyro_neo_species(i_sp + 1 + n_species)
                for neo_key in pyro_neo_species.values():
                    if neo_key in self.data:
                        self.data.pop(neo_key)

        for i_sp, name in enumerate(local_species.names):
            pyro_neo_species = self.get_pyro_neo_species(i_sp + 1)
            for pyro_key, neo_key in pyro_neo_species.items():
                self.data[neo_key] = local_species[name][pyro_key]

        if "electron" in local_species.names:
            first_species = "electron"
            self.data["NU_1"] = local_species.electron.nu
        else:
            first_species = local_species.names[0]

            zion = local_species[first_species].z
            nion = local_species[first_species].dens
            tion = local_species[first_species].temp
            mion = local_species[first_species].mass

            te = self.data.get("TEMP_AE", 1.0) * convention.tref
            ne = self.data.get("DENS_AE", 1.0) * convention.nref
            me = self.data.get("MASS_AE", 2.724486e-4) * convention.mref

            self.data["NU_1"] = (
                local_species[first_species].nu
                / (zion**4 * nion / tion**1.5 / mion**0.5)
                * (ne / te**1.5 / me**0.5)
            )

            # Set adiabatic flags
            self.data["AE_FLAG"] = 1
            self.data["DENS_AE"] = ne
            self.data["TEMP_AE"] = te
            self.data["MASS_AE"] = me

        self.data["OMEGA_ROT"] = (
            local_species[first_species].omega0 * self.data["RMAJ_OVER_A"]
        )
        self.data["OMEGA_ROT_DERIV"] = (
            -local_species[first_species].domega_drho * self.data["RMAJ_OVER_A"]
        )

        # Set time stepping
        self.data["N_RADIAL"] = 1.0
        self.data["N_THETA"] = numerics.ntheta
        self.data["N_ENERGY"] = numerics.nenergy
        self.data["N_XI"] = numerics.npitch

        if not local_norm:
            return

        try:
            rho_star = (1 * local_norm.neo.rhoref / local_norm.neo.lref).to(
                "dimensionless"
            )
            self.data["RHO_STAR"] = rho_star
        except (PyroNormalisationError, PyroContextError):
            rho_star = self.data.get("RHO_STAR", 0.001)
            print(f"Leaving RHO_STAR unchanged as {rho_star}")

        self.data = convert_dict(self.data, convention)

    def _check_basic_miller(self):
        """
        Checks if NEO input file is a basic Miller geometry by seeing if moments that are higher than triangularity
        are 0
        Returns
        -------
        is_basic_miller: Boolean
            True if Miller, False is MXH
        """

        mxh_only_parameters = [
            "ZETA",
            "S_ZETA",
            "SHAPE_COS0",
            "SHAPE_COS1",
            "SHAPE_COS2",
            "SHAPE_COS3",
            "SHAPE_SIN3",
            "SHAPE_S_COS0",
            "SHAPE_S_COS1",
            "SHAPE_S_COS2",
            "SHAPE_S_COS3",
            "SHAPE_S_SIN3",
        ]

        is_basic_miller = True
        for param in mxh_only_parameters:
            if self.data.get(param, 0.0) != 0:
                is_basic_miller = False
                break

        return is_basic_miller

    def get_ne_te_normalisation(self):
        found_electron = False
        if self.data.get("AE_FLAG", 0) == 1:
            ne = self.data["DENS_AE"]
            Te = self.data["TEMP_AE"]
            found_electron = True
        else:
            for i_sp in range(self.data["N_SPECIES"]):
                if self.data[f"Z_{i_sp + 1}"] == -1:
                    ne = self.data[f"DENS_{i_sp + 1}"]
                    Te = self.data[f"TEMP_{i_sp + 1}"]
                    found_electron = True
                    break

        if not found_electron:
            raise TypeError(
                "Pyro currently requires an electron species in the input file"
            )

        return ne, Te


class NEOFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderNEO(FileReader, file_type="NEO", reads=GKOutput):

    def read_from_file(
        self,
        filename: PathLike,
        norm: Normalisation,
        output_convention: str = "pyrokinetics",
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
        downsample: Dict[str, Any] = {},
    ) -> GKOutput:
        raise NotImplementedError("It is not possible to load NEO output data yet")

    def verify_file_type(self, dirname: PathLike):
        raise NotImplementedError("It is not possible to load NEO output data yet")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        raise NotImplementedError("It is not possible to load NEO output data yet")
