import logging
from ast import literal_eval
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from cleverdict import CleverDict

from ..constants import deuterium_mass, electron_mass, hydrogen_mass, pi
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


class GKInputCGYRO(GKInput, FileReader, file_type="CGYRO", reads=GKInput):
    """
    Class that can read CGYRO input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "CGYRO"
    default_file_name = "input.cgyro"
    norm_convention = "cgyro"

    pyro_cgyro_miller = {
        "rho": "RMIN",
        "Rmaj": "RMAJ",
        "q": "Q",
        "kappa": "KAPPA",
        "s_kappa": "S_KAPPA",
        "delta": "DELTA",
        "shat": "S",
        "shift": "SHIFT",
        "ip_ccw": "IPCCW",
        "bt_ccw": "BTCCW",
    }

    pyro_cgyro_mxh = {
        **pyro_cgyro_miller,
        "s_delta": "S_DELTA",
        "Z0": "ZMAG",
        "dZ0dr": "DZMAG",
        "zeta": "ZETA",
        "s_zeta": "S_ZETA",
        "cn0": "SHAPE_COS0",
        "cn1": "SHAPE_COS1",
        "cn2": "SHAPE_COS2",
        "cn3": "SHAPE_COS3",
        "sn3": "SHAPE_SIN3",
        "dcndr0": "SHAPE_S_COS0",
        "dcndr1": "SHAPE_S_COS1",
        "dcndr2": "SHAPE_S_COS2",
        "dcndr3": "SHAPE_S_COS3",
        "dsndr3": "SHAPE_S_SIN3",
    }

    pyro_cgyro_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "q": 2.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "shat": 1.0,
        "shift": 0.0,
        "ip_ccw": -1.0,
        "bt_ccw": -1.0,
    }

    pyro_cgyro_mxh_defaults = {
        **pyro_cgyro_miller_defaults,
        "s_delta": 0.0,
        "Z0": 0.0,
        "dZ0dr": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "cn0": 0.0,
        "cn1": 0.0,
        "cn2": 0.0,
        "cn3": 0.0,
        "sn3": 0.0,
        "dcndr0": 0.0,
        "dcndr1": 0.0,
        "dcndr2": 0.0,
        "dcndr3": 0.0,
        "dsndr3": 0.0,
    }

    pyro_cgyro_fourier = pyro_cgyro_miller

    pyro_cgyro_fourier_defaults = pyro_cgyro_miller_defaults

    @staticmethod
    def get_pyro_cgyro_species(iSp=1):
        return {
            "mass": f"MASS_{iSp}",
            "z": f"Z_{iSp}",
            "dens": f"DENS_{iSp}",
            "temp": f"TEMP_{iSp}",
            "inverse_lt": f"DLNTDR_{iSp}",
            "inverse_ln": f"DLNNDR_{iSp}",
        }

    cgyro_eq_types = {
        1: "SAlpha",
        2: "MXH",
        3: "Fourier",
    }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads CGYRO input file into a dictionary
        """
        with open(filename) as f:
            self.data = self.parse_cgyro(f)
        return self.data

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads CGYRO input file given as string
        """
        self.data = self.parse_cgyro(input_string.split("\n"))
        return self.data

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads CGYRO input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

    @staticmethod
    def parse_cgyro(lines):
        """
        Given lines of a cgyro file or a string split by '/n', return a dict of
        CGYRO input data
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
        Ensure this file is a valid cgyro input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        # The following keys are not strictly needed for a CGYRO input file,
        # but they are needed by Pyrokinetics
        expected_keys = [
            "BETAE_UNIT",
            "N_SPECIES",
            "NU_EE",
            "N_FIELD",
            "N_RADIAL",
            "RMIN",
            "RMAJ",
            "Q",
            "S",
        ]
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm=None,
        code_normalisation=None,
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
        Add extra flags to CGYRO input file
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

        eq_type = self.cgyro_eq_types[self.data["EQUILIBRIUM_MODEL"]]

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
                f"LocalGeometry type {eq_type} not implemented for CGYRO"
            )

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from CGYRO file
        """
        miller_data = default_miller_inputs()

        for (key, val), val_default in zip(
            self.pyro_cgyro_miller.items(),
            self.pyro_cgyro_miller_defaults.values(),
        ):
            miller_data[key] = self.data.get(val, val_default)

        miller_data["s_delta"] = self.data.get("S_DELTA", 0.0) / np.sqrt(
            1 - self.data.get("DELTA", 0.0) ** 2
        )

        miller_data["Z0"] = self.data.get("ZMAG", 0.0)
        miller_data["dZ0dr"] = self.data.get("DZMAG", 0.0)

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying miller after creation
        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data.get("BETAE_UNIT", 0.0) * ne_norm * Te_norm
        if beta != 0:
            miller.B0 = 1 / beta**0.5
        else:
            miller.B0 = None

        # Need species to set up beta_prime
        local_species = self.get_local_species()
        beta_prime_scale = self.data.get("BETA_STAR_SCALE", 1.0)

        if miller.B0 is not None:
            miller.beta_prime = (
                -local_species.inverse_lp.m * beta_prime_scale / miller.B0**2
            )
        else:
            miller.beta_prime = 0.0

        return miller

    def get_local_geometry_mxh(self) -> LocalGeometryMXH:
        """
        Load MXH object from CGYRO file
        """
        mxh_data = default_mxh_inputs()

        for (key, val), default in zip(
            self.pyro_cgyro_mxh.items(), self.pyro_cgyro_mxh_defaults.values()
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

        # Force dsndr[0] = 0 as is definition
        mxh_data["dsndr"][0] = 0.0

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        mxh = LocalGeometryMXH.from_gk_data(mxh_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying mxh after creation
        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data.get("BETAE_UNIT", 0.0) * ne_norm * Te_norm
        if beta != 0:
            mxh.B0 = 1 / beta**0.5
        else:
            mxh.B0 = None

        # Need species to set up beta_prime
        local_species = self.get_local_species()
        beta_prime_scale = self.data.get("BETA_STAR_SCALE", 1.0)

        if mxh.B0 is not None:
            mxh.beta_prime = -local_species.inverse_lp.m * beta_prime_scale / mxh.B0**2
        else:
            mxh.beta_prime = 0.0

        return mxh

    def get_local_geometry_fourier(self) -> LocalGeometryFourierCGYRO:
        """
        Load Fourier object from CGYRO file
        """
        fourier_data = default_fourier_cgyro_inputs()

        for (key, val), val_default in zip(
            self.pyro_cgyro_fourier.items(), self.pyro_cgyro_fourier_defaults.values()
        ):
            fourier_data[key] = self.data.get(val, val_default)

        # Add CGYRO mappings here

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        fourier = LocalGeometryFourierCGYRO.from_gk_data(fourier_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying fourier after creation
        # FIXME Is this assumption general enough? Can't we get pref from local_species?
        # FIXME B0 = None can cause problems when writing
        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data.get("BETAE_UNIT", 0.0) * ne_norm * Te_norm
        if beta != 0:
            fourier.B0 = 1 / beta**0.5
        else:
            fourier.B0 = None

        # Need species to set up beta_prime
        local_species = self.get_local_species()
        beta_prime_scale = self.data.get("BETA_STAR_SCALE", 1.0)

        if fourier.B0 is not None:
            fourier.beta_prime = (
                -local_species.inverse_lp.m * beta_prime_scale / fourier.B0**2
            )
        else:
            fourier.beta_prime = 0.0

        return fourier

    def get_local_species(self):
        """
        Load LocalSpecies object from CGYRO file
        """

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        ion_count = 0

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")

            convention = getattr(norms, self.norm_convention)

        domega_drho = -self.data.get("GAMMA_P", 0.0) / self.data["RMAJ"]

        # Load each species into a dictionary
        for i_sp in range(self.data["N_SPECIES"]):
            pyro_cgyro_species = self.get_pyro_cgyro_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_cgyro_species.items():
                species_data[p_key] = self.data[c_key]

            species_data.omega0 = (
                self.data.get("MACH", 0.0)
                * convention.vref
                / convention.lref
                / self.data["RMAJ"]
            )
            species_data.domega_drho = (
                domega_drho * convention.vref / convention.lref**2
            )

            if species_data.z == -1:
                name = "electron"
                species_data.nu = (
                    self.data.get("NU_EE", 0.1) * convention.vref / convention.lref
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

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

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
        local_species.normalise()

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

        nfields = self.data.get("N_FIELD", 1)

        numerics_data["phi"] = nfields >= 1
        numerics_data["apar"] = nfields >= 2
        numerics_data["bpar"] = nfields >= 3

        numerics_data["delta_time"] = self.data.get("DELTA_T", 0.01)
        numerics_data["max_time"] = self.data.get("MAX_TIME", 1.0)

        numerics_data["ky"] = self.data["KY"]
        numerics_data["nky"] = self.data.get("N_TOROIDAL", 1)
        numerics_data["theta0"] = 2 * pi * self.data.get("PX0", 0.0)
        numerics_data["nkx"] = self.data.get("N_RADIAL", 1)
        numerics_data["nperiod"] = int(self.data["N_RADIAL"] / 2)

        shat = self.data[self.pyro_cgyro_miller["shat"]]
        box_size = self.data.get("BOX_SIZE", 1)
        if numerics_data["nky"] == 1:
            numerics_data["kx"] = numerics_data["ky"] * shat * numerics_data["theta0"]
        else:
            numerics_data["kx"] = numerics_data["ky"] * 2 * pi * shat / box_size

        numerics_data["ntheta"] = self.data.get("N_THETA", 24)
        numerics_data["nenergy"] = self.data.get("N_ENERGY", 8)
        numerics_data["npitch"] = self.data.get("N_XI", 16)

        numerics_data["nonlinear"] = self.is_nonlinear()

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        numerics_data["beta"] = (
            self.data.get("BETAE_UNIT", 0.0) * convention.beta_ref * ne_norm * Te_norm
        )

        numerics_data["gamma_exb"] = (
            self.data.get("GAMMA_E", 0.0) * convention.vref / convention.lref
        )

        return Numerics(**numerics_data)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in reference values from input file

        """
        return {}

    def _get_normalisation(self):
        """
        Automatically detects the normalisation from the input file and
        returns a dictionary of the different reference species. If the
        references used match the default references then an empty dict
        is returned

        Returns
        -------
        references : dict
            Dictionary of reference species for the density, temperature
            and mass along with reference magnetic field and length. The
            electron temp, density and ratio of R_geometric/R_major is
            included where R_geometric corresponds to the R where Bref is.
            B0 means magnetic field at the centre of the local flux surface
            and Bgeo is the magnetic field at the centre of the last closed
            flux surface.
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
        }

        references = copy(default_references)

        dens_index = []
        temp_index = []

        found_electron = False
        if self.data.get("AE_FLAG", 0) == 1:
            references["ne"] = self.data["DENS_AE"]
            references["te"] = self.data["TEMP_AE"]
            e_mass = self.data["MASS_AE"]
            electron_index = "AE"
            found_electron = True

            if np.isclose(self.data["DENS_AE"], 1.0):
                dens_index.append("AE")
            if np.isclose(self.data["TEMP_AE"], 1.0):
                temp_index.append("AE")
        else:
            for i_sp in range(self.data["N_SPECIES"]):
                if self.data[f"Z_{i_sp+1}"] == -1:
                    references["ne"] = self.data[f"DENS_{i_sp+1}"]
                    references["te"] = self.data[f"TEMP_{i_sp+1}"]
                    e_mass = self.data[f"MASS_{i_sp+1}"]
                    electron_index = i_sp + 1
                    found_electron = True

                if np.isclose(self.data[f"DENS_{i_sp+1}"], 1.0):
                    dens_index.append(i_sp + 1)
                if np.isclose(self.data[f"TEMP_{i_sp+1}"], 1.0):
                    temp_index.append(i_sp + 1)

        if not found_electron:
            raise TypeError(
                "Pyro currently requires an electron species in the input file"
            )

        if len(temp_index) == 0 or len(dens_index) == 0:
            raise ValueError("Cannot find any reference temperature/density species")

        if not found_electron:
            raise TypeError(
                "Pyro currently only supports electron species with charge = -1"
            )

        me_md = (electron_mass / deuterium_mass).m
        me_mh = (electron_mass / hydrogen_mass).m

        if np.isclose(e_mass, 1.0):
            references["mref_species"] = "electron"
        elif np.isclose(e_mass, me_md, rtol=0.1):
            references["mref_species"] = "deuterium"
        elif np.isclose(e_mass, me_mh, rtol=0.1):
            references["mref_species"] = "hydrogen"
        else:
            raise ValueError("Cannot determine reference mass")

        if electron_index in dens_index:
            references["nref_species"] = "electron"
        else:
            for i_sp in dens_index:
                if np.isclose(self.data[f"MASS{i_sp}"], 1.0):
                    references["nref_species"] = references["mref_species"]

        if references["nref_species"] is None:
            raise ValueError("Cannot determine reference density species")

        if electron_index in temp_index:
            references["tref_species"] = "electron"
        else:
            for i_sp in temp_index:
                if np.isclose(self.data[f"TEMP_{i_sp}"], 1.0):
                    references["tref_species"] = references["mref_species"]

        if references["nref_species"] is None:
            raise ValueError("Cannot determine reference density species")

        rmaj = self.data["RMAJ"]

        if rmaj == 1:
            references["lref"] = "major_radius"
        else:
            references["lref"] = "minor_radius"

        if references == default_references:
            return {}
        else:
            self.norm_convention = f"{self.code_name.lower()}_bespoke"
            return references

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
                template_file = gk_templates["CGYRO"]
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
                "implemented yet for CGYRO"
            )
        else:
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} not "
                "implemented for CGYRO"
            )

        # Set equilibrium type in input file
        self.data["EQUILIBRIUM_MODEL"] = eq_model

        if eq_type == "Miller":
            # Assign Miller values to input file
            for key, val in self.pyro_cgyro_miller.items():
                self.data[val] = local_geometry[key]

            self.data["S_DELTA"] = local_geometry.s_delta * np.sqrt(
                1 - local_geometry.delta**2
            )
            self.data["ZMAG"] = local_geometry.Z0
            self.data["DZMAG"] = local_geometry.dZ0dr

        elif eq_type == "Fourier":
            # Assign Fourier values to input file
            for key, val in self.pyro_cgyro_fourier.items():
                self.data[val] = local_geometry[key]

        elif eq_type == "MXH":
            # Assign MXH values to input file
            for key, val in self.pyro_cgyro_mxh.items():
                if "SHAPE" not in val:
                    self.data[val] = getattr(local_geometry, key)
                else:
                    index = int(key[-1])
                    new_key = key[:-1]
                    if "SHAPE_S" in val:
                        self.data[val] = (
                            getattr(local_geometry, new_key)[index] * local_geometry.rho
                        )
                    else:
                        self.data[val] = getattr(local_geometry, new_key)[index]

        # Kinetic data
        self.data["N_SPECIES"] = local_species.nspec

        for i_sp, name in enumerate(local_species.names):
            pyro_cgyro_species = self.get_pyro_cgyro_species(i_sp + 1)
            for pyro_key, cgyro_key in pyro_cgyro_species.items():
                self.data[cgyro_key] = local_species[name][pyro_key]
        self.data["MACH"] = local_species.electron.omega0 * self.data["RMAJ"]
        self.data["GAMMA_P"] = (
            -local_species.electron.domega_drho * self.data["RMAJ"] * convention.lref
        )
        self.data["Z_EFF_METHOD"] = 1
        self.data["Z_EFF"] = local_species.zeff

        # FIXME if species aren't defined, won't this fail?
        self.data["NU_EE"] = local_species.electron.nu

        beta_ref = convention.beta if local_norm else 0.0
        beta = numerics.beta if numerics.beta is not None else beta_ref

        # Calculate beta_prime_scale
        if beta != 0.0:
            beta_prime_scale = -local_geometry.beta_prime / (
                local_species.inverse_lp.m * beta
            )
        else:
            beta_prime_scale = 1.0

        self.data["BETAE_UNIT"] = beta
        self.data["BETA_STAR_SCALE"] = beta_prime_scale

        # Numerics
        if numerics.bpar and not numerics.apar:
            raise ValueError("Can't have bpar without apar in CGYRO")

        self.data["N_FIELD"] = 1 + int(numerics.bpar) + int(numerics.apar)

        # Set time stepping
        self.data["DELTA_T"] = numerics.delta_time
        self.data["MAX_TIME"] = numerics.max_time

        if numerics.nonlinear:
            self.data["NONLINEAR_FLAG"] = 1
            self.data["N_RADIAL"] = numerics.nkx
            self.data["BOX_SIZE"] = int(
                (numerics.ky * 2 * pi * local_geometry.shat / numerics.kx) + 0.1
            )
        else:
            self.data["NONLINEAR_FLAG"] = 0
            self.data["N_RADIAL"] = numerics.nperiod * 2
            self.data["BOX_SIZE"] = 1

        self.data["KY"] = numerics.ky
        self.data["N_TOROIDAL"] = numerics.nky

        self.data["N_THETA"] = numerics.ntheta
        self.data["THETA_PLOT"] = numerics.ntheta
        self.data["PX0"] = numerics.theta0 / (2 * pi)

        self.data["GAMMA_E"] = numerics.gamma_exb

        self.data["N_ENERGY"] = numerics.nenergy
        self.data["N_XI"] = numerics.npitch

        self.data["FIELD_PRINT_FLAG"] = 1
        self.data["MOMENT_PRINT_FLAG"] = 1

        if not local_norm:
            return

        self.data = convert_dict(self.data, convention)

    def _check_basic_miller(self):
        """
        Checks if CGYRO input file is a basic Miller geometry by seeing if moments that are higher than triangularity
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
                if self.data[f"Z_{i_sp+1}"] == -1:
                    ne = self.data[f"DENS_{i_sp+1}"]
                    Te = self.data[f"TEMP_{i_sp+1}"]
                    found_electron = True
                    break

        if not found_electron:
            raise TypeError(
                "Pyro currently requires an electron species in the input file"
            )

        return ne, Te


class CGYROFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderCGYRO(FileReader, file_type="CGYRO", reads=GKOutput):
    fields = ["phi", "apar", "bpar"]
    moments = ["n", "e", "v"]

    def read_from_file(
        self,
        filename: PathLike,
        norm: Normalisation,
        downsize: int = 1,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
    ) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(
            filename, load_fields, load_moments
        )
        coords = self._get_coords(raw_data, gk_input, downsize)
        fields = self._get_fields(raw_data, gk_input, coords) if load_fields else None
        fluxes = self._get_fluxes(raw_data, coords) if load_fluxes else None
        moments = (
            self._get_moments(raw_data, gk_input, coords) if load_moments else None
        )

        if coords["linear"] and (
            coords["ntheta_plot"] != coords["ntheta_grid"] or not fields
        ):
            eigenvalues = self._get_eigenvalues(raw_data, coords, gk_input)
            eigenfunctions = self._get_eigenfunctions(raw_data, coords)
        else:
            # Rely on gk_output to generate eigenvalues
            eigenvalues = None
            eigenfunctions = None

        # Assign units and return GKOutput
        convention = getattr(norm, gk_input.norm_convention)
        field_dims = ("theta", "kx", "ky", "time")
        flux_dims = ("field", "species", "ky", "time")
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
            eigenfunctions=(
                None if eigenfunctions is None else Eigenfunctions(eigenfunctions)
            ),
            linear=coords["linear"],
            gk_code="CGYRO",
            input_file=input_str,
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
        For CGYRO, simply returns dir of the path.
        """
        return Path(filename).parent

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": CGYROFile(dirname / "input.cgyro", required=True),
            "time": CGYROFile(dirname / "out.cgyro.time", required=True),
            "grids": CGYROFile(dirname / "out.cgyro.grids", required=True),
            "equilibrium": CGYROFile(dirname / "out.cgyro.equilibrium", required=True),
        }

    @classmethod
    def _get_raw_data(
        cls, dirname: PathLike, load_fields, load_moments
    ) -> Tuple[Dict[str, Any], GKInputCGYRO, str]:
        dirname = Path(dirname)
        if not dirname.exists():
            raise RuntimeError(
                f"GKOutputReaderCGYRO: Provided path {dirname} does not exist. "
                "Please supply the name of a directory containing CGYRO output files."
            )
        if not dirname.is_dir():
            raise RuntimeError(
                f"GKOutputReaderCGYRO: Provided path {dirname} is not a directory. "
                "Please supply the name of a directory containing CGYRO output files."
            )

        # The following list of CGYRO files may exist
        expected_files = {
            **cls._required_files(dirname),
            "flux": CGYROFile(dirname / "bin.cgyro.ky_flux", required=False),
            "cflux": CGYROFile(dirname / "bin.cgyro.ky_cflux", required=False),
            "eigenvalues_bin": CGYROFile(dirname / "bin.cgyro.freq", required=False),
            "eigenvalues_out": CGYROFile(dirname / "out.cgyro.freq", required=False),
            **{
                f"field_{f}": CGYROFile(dirname / f"bin.cgyro.kxky_{f}", required=False)
                for f in cls.fields
                if load_fields
            },
            **{
                f"moment_{m}": CGYROFile(
                    dirname / f"bin.cgyro.kxky_{m}", required=False
                )
                for m in cls.moments
                if load_moments
            },
            **{
                f"eigenfunctions_{f}": CGYROFile(
                    dirname / f"bin.cgyro.{f}b", required=False
                )
                for f in cls.fields
            },
        }
        # Read in files
        raw_data = {}
        for key, cgyro_file in expected_files.items():
            if not cgyro_file.path.exists():
                if cgyro_file.required:
                    raise RuntimeError(
                        f"GKOutputReaderCGYRO: The file {cgyro_file.path.name} is needed"
                    )
                continue
            # Read in file according to format
            if cgyro_file.fmt == "input":
                with open(cgyro_file.path, "r") as f:
                    raw_data[key] = f.read()
            if cgyro_file.fmt == "out":
                raw_data[key] = np.loadtxt(cgyro_file.path)
            if cgyro_file.fmt == "bin":
                raw_data[key] = np.fromfile(cgyro_file.path, dtype="float32")
        input_str = raw_data["input"]
        gk_input = GKInputCGYRO()
        gk_input.read_str(input_str)
        gk_input._get_normalisation()

        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(
        raw_data: Dict[str, Any], gk_input: GKInputCGYRO, downsize: int = 1
    ) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of CGYRO
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing CGYRO output.
            gk_input (GKInputCGYRO): Processed CGYRO input file.

        Returns:
            Dict:  Dictionary with coords
        """
        bunit_over_b0 = gk_input.get_local_geometry().bunit_over_b0.m

        # Process time data
        time = raw_data["time"][:, 0]

        if len(time) % downsize != 0:
            residual = len(time) % downsize - downsize
        else:
            residual = 0

        time = time[::downsize]

        # Process grid data
        grid_data = raw_data["grids"]
        nky = int(grid_data[0])
        nspecies = int(grid_data[1])
        nfield = int(grid_data[2])
        nkx = int(grid_data[3])
        ntheta_grid = int(grid_data[4])
        nenergy = int(grid_data[5])
        npitch = int(grid_data[6])
        box_size = int(grid_data[7])
        length_x = grid_data[8]
        ntheta_plot = int(grid_data[10])

        # Iterate through grid_data in chunks, starting after kx
        pos = 11 + nkx

        theta_grid = grid_data[pos : pos + ntheta_grid]
        pos += ntheta_grid

        energy = grid_data[pos : pos + nenergy]
        pos += nenergy

        pitch = grid_data[pos : pos + npitch]
        pos += npitch

        ntheta_ballooning = ntheta_grid * int(nkx / box_size)
        theta_ballooning = grid_data[pos : pos + ntheta_ballooning]
        pos += ntheta_ballooning

        ky = grid_data[pos : pos + nky] / bunit_over_b0

        if gk_input.is_linear():
            # Convert to ballooning co-ordinate so only 1 kx
            theta = theta_ballooning
            ntheta = ntheta_ballooning
            kx = [0.0]
            nkx = 1
        else:
            # Output data actually given on theta_plot grid
            ntheta = ntheta_plot
            theta = [0.0] if ntheta == 1 else theta_grid[:: ntheta_grid // ntheta]
            kx = (
                2
                * pi
                * np.linspace(-int(nkx / 2), int((nkx + 1) / 2) - 1, nkx)
                / length_x
            ) / bunit_over_b0

        # Get rho_star from equilibrium file
        if len(raw_data["equilibrium"]) == 54 + 7 * nspecies:
            rho_star = raw_data["equilibrium"][35]
        else:
            rho_star = raw_data["equilibrium"][23]

        fields = ["phi", "apar", "bpar"][:nfield]
        fluxes = ["particle", "heat", "momentum"]
        moments = ["density", "temperature", "velocity"]
        species = gk_input.get_local_species().names
        if nspecies != len(species):
            raise RuntimeError(
                "GKOutputReaderCGYRO: Different number of species in input and output."
            )

        # Store grid data as xarray DataSet
        return {
            "time": time,
            "kx": kx,
            "ky": ky,
            "theta": theta,
            "energy": energy,
            "pitch": pitch,
            "ntheta_plot": ntheta_plot,
            "ntheta_grid": ntheta_grid,
            "nradial": int(gk_input.data["N_RADIAL"]),
            "rho_star": rho_star,
            "field": fields,
            "moment": moments,
            "flux": fluxes,
            "species": species,
            "linear": gk_input.is_linear(),
            "downsize": downsize,
            "residual": residual,
        }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any],
        gk_input: GKInputCGYRO,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D fields over time.
        The field coordinates should be (field, theta, kx, ky, time)
        """
        nkx = len(coords["kx"])
        nradial = coords["nradial"]
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntheta_plot = coords["ntheta_plot"]
        ntheta_grid = coords["ntheta_grid"]
        ntime = len(coords["time"])
        nfield = len(coords["field"])
        downsize = coords["downsize"]
        residual = coords["residual"]

        full_ntime = ntime * downsize + residual
        field_names = ["phi", "apar", "bpar"][:nfield]

        raw_field_data = {f: raw_data.get(f"field_{f}", None) for f in field_names}

        results = {}

        # Check to see if there's anything to do
        if not raw_field_data:
            return results

        # Loop through all fields and add field in if it exists
        for ifield, (field_name, raw_field) in enumerate(raw_field_data.items()):
            if raw_field is None:
                logging.warning(
                    f"Field data {field_name} over time not found, expected the file "
                    f"bin.cygro.kxky_{field_name} to exist. Setting this field to 0."
                )
                continue

            # If linear, convert from kx to ballooning space.
            # Use nradial instead of nkx, ntheta_plot instead of ntheta
            if gk_input.is_linear():
                shape = (2, nradial, ntheta_plot, nky, full_ntime)
            else:
                shape = (2, nkx, ntheta, nky, full_ntime)

            field_data = raw_field[: np.prod(shape)].reshape(shape, order="F")
            # Adjust sign to match pyrokinetics frequency convention
            # (-ve is electron direction)
            mode_sign = np.sign(gk_input.data.get("IPCCW", -1))

            field_data = (field_data[0] + mode_sign * 1j * field_data[1]) / coords[
                "rho_star"
            ]

            # If nonlinear, we can simply save the fields and continue
            if gk_input.is_nonlinear():
                fields = field_data.swapaxes(0, 1)
            else:
                # If theta_plot != theta_grid, we get eigenfunction data and multiply by the
                # field amplitude
                if ntheta_plot != ntheta_grid:
                    # Get eigenfunction data
                    raw_eig_data = raw_data.get(f"eigenfunctions_{field_name}", None)
                    if raw_eig_data is None:
                        logging.warning(
                            f"When setting fields, eigenfunction data for {field_name} not "
                            f"found, expected the file bin.cygro.{field_name}b to exist. "
                            f"Not setting the field {field_name}."
                        )
                        continue
                    eig_shape = [2, ntheta, full_ntime]
                    eig_data = raw_eig_data[: np.prod(eig_shape)].reshape(
                        eig_shape, order="F"
                    )
                    eig_data = eig_data[0] + 1j * eig_data[1]
                    # Get field amplitude
                    middle_kx = (nradial // 2) + 1
                    field_amplitude = np.abs(field_data[middle_kx, 0, 0, :])
                    # Multiply together
                    # FIXME We only set kx=ky=0 here, any other values are left undefined
                    #       as fields is created using np.empty. Should we instead set
                    #       all kx and ky to these values? Should we expect that nx=ny=1?
                    field_data = np.reshape(
                        eig_data * field_amplitude,
                        (nradial, ntheta_grid, nky, full_ntime),
                    )

                # Poisson Sum (no negative in exponent to match frequency convention)
                q = np.abs(gk_input.get_local_geometry_miller().q)
                nx0 = gk_input.data.get("PX0", 0.0)
                for i_radial in range(nradial):
                    nx = -nradial // 2 + (i_radial - 1)
                    field_data[i_radial, ...] *= np.exp(2j * pi * (nx + nx0) * q)

                fields = field_data.reshape([ntheta, nkx, nky, full_ntime])

            fields = fields[:, :, :, ::downsize]
            results[field_name] = fields

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputCGYRO,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        moment_names = {"n": "density", "e": "temperature", "v": "velocity"}

        nkx = len(coords["kx"])
        nradial = coords["nradial"]
        nky = len(coords["ky"])
        ntheta = len(coords["theta"])
        ntheta_plot = coords["ntheta_plot"]
        ntime = len(coords["time"])
        nspec = len(coords["species"])
        residual = coords["residual"]
        downsize = coords["downsize"]
        full_ntime = ntime * downsize + residual

        raw_moment_data = {
            value: raw_data.get(f"moment_{key}", None)
            for key, value in moment_names.items()
        }
        results = {}

        # Check to see if there's anything to do
        if not raw_moment_data:
            return results

        # Loop through all moments and add moment in if it exists
        for imoment, (moment_name, raw_moment) in enumerate(raw_moment_data.items()):
            if raw_moment is None:
                logging.warning(
                    f"moment data {moment_name} over time not found, expected the file "
                    f"bin.cygro.kxky_{moment_name} to exist. Setting this moment to 0."
                )
                continue

            # If linear, convert from kx to ballooning space.
            # Use nradial instead of nkx, ntheta_plot instead of ntheta
            if gk_input.is_linear():
                shape = (2, nradial, ntheta_plot, nspec, nky, full_ntime)
            else:
                shape = (2, nkx, ntheta, nspec, nky, full_ntime)

            moment_data = raw_moment[: np.prod(shape)].reshape(shape, order="F")
            # Adjust sign to match pyrokinetics frequency convention
            # (-ve is electron direction)
            mode_sign = -np.sign(
                np.sign(gk_input.data.get("Q", 2.0)) * -gk_input.data.get("BTCCW", -1)
            )

            moment_data = (moment_data[0] + mode_sign * 1j * moment_data[1]) / coords[
                "rho_star"
            ]

            # If nonlinear, we can simply save the moments and continue
            if gk_input.is_nonlinear():
                moments = moment_data.swapaxes(0, 1)
            else:
                # Poisson Sum (no negative in exponent to match frequency convention)
                q = gk_input.get_local_geometry_miller().q
                for i_radial in range(nradial):
                    nx = -nradial // 2 + (i_radial - 1)
                    moment_data[i_radial, ...] *= np.exp(2j * pi * nx * q)

                moments = moment_data.reshape([ntheta, nkx, nspec, nky, full_ntime])

            moments = moments[:, :, :, :, ::downsize]
            results[moment_name] = moments

        temp_spec = np.ones((ntheta, nkx, nspec, nky, ntime))
        for i in range(nspec):
            temp_spec[:, :, i, :, :] = gk_input.data.get(f"TEMP_{i+1}", 1.0)

        if "temperature" in results:
            # Convert CGYRO energy fluctuation to temperature
            results["temperature"] = (
                2 * results["temperature"] - results["density"] * temp_spec
            )

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
        downsize = coords["downsize"]
        residual = coords["residual"]
        # cflux is more appropriate for CGYRO simulations
        # with GAMMA_E > 0 and SHEAR_METHOD = 2.
        # However, for cross-code consistency, gflux is used for now.
        # if gk_input.data.get("GAMMA_E", 0.0) == 0.0:
        #     flux_key = "flux"
        # else:
        #     flux_key = "cflux"
        flux_key = "flux"

        if flux_key in raw_data:
            coord_names = ["species", "flux", "field", "ky"]
            shape = [len(coords[coord_name]) for coord_name in coord_names]
            shape.append(ntime * downsize + residual)
            fluxes = raw_data[flux_key][: np.prod(shape)].reshape(shape, order="F")

        fluxes = np.swapaxes(fluxes, 0, 2)
        for iflux, flux in enumerate(coords["flux"]):
            results[flux] = fluxes[:, iflux, :, :, ::downsize]

        return results

    @classmethod
    def _get_eigenvalues(
        self, raw_data: Dict[str, Any], coords: Dict, gk_input: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(kx, ky, time)
        data['mode_frequency'] = mode_frequency(kx, ky, time)
        data['growth_rate'] = growth_rate(kx, ky, time)

        This should be called after _set_fields, and is only valid for linear runs.
        Unlike the version in the super() class, CGYRO may need to get extra info from
        an eigenvalue file.

        Args:
            data (xr.Dataset): The dataset to be modified.
            dirname (PathLike): Directory containing CGYRO output files.
        Returns:
            Dict: The modified dataset which was passed to 'data'.
        """

        ntime = len(coords["time"])
        nky = len(coords["ky"])
        nkx = len(coords["kx"])
        shape = (2, nky, ntime)

        if "eigenvalues_bin" in raw_data:
            eigenvalue_over_time = raw_data["eigenvalues_bin"][
                : np.prod(shape)
            ].reshape(shape, order="F")
        elif "eigenvalues_out" in raw_data:
            eigenvalue_over_time = (
                raw_data["eigenvalues_out"].transpose()[:, :ntime].reshape(shape)
            )
        else:
            raise RuntimeError(
                "Eigenvalues over time not found, expected the files bin.cgyro.freq or "
                "out.cgyro.freq to exist. Could not set data_vars 'growth_rate', "
                "'mode_frequency' and 'eigenvalue'."
            )
        mode_sign = -np.sign(
            np.sign(gk_input.data.get("Q", 2.0)) * -gk_input.data.get("BTCCW", -1)
        )

        mode_frequency = mode_sign * eigenvalue_over_time[0, :, :]

        growth_rate = eigenvalue_over_time[1, :, :]
        # Add kx axis for compatibility with GS2 eigenvalues
        # FIXME Is this appropriate? Should we drop the kx coordinate?
        shape_with_kx = (nkx, nky, ntime)
        mode_frequency = np.ones(shape_with_kx) * mode_frequency
        growth_rate = np.ones(shape_with_kx) * growth_rate

        result = {
            "growth_rate": growth_rate,
            "mode_frequency": mode_frequency,
        }

        return result

    @staticmethod
    def _get_eigenfunctions(raw_data: Dict[str, Any], coords: Dict) -> np.ndarray:
        """
        Loads eigenfunctions into data with the following coordinates:

        data['eigenfunctions'] = eigenfunctions(kx, ky, field, theta, time)

        This should be called after _set_fields, and is only valid for linear runs.
        """

        raw_eig_data = [
            raw_data.get(f"eigenfunctions_{f}", None) for f in coords["field"]
        ]

        ntime = len(coords["time"])
        ntheta = len(coords["theta"])
        nkx = len(coords["kx"])
        nky = len(coords["ky"])

        raw_shape = [2, ntheta, nkx, nky, ntime]

        # FIXME Currently using kx and ky for compatibility with GS2 results, but
        #       these coordinates are not used. Should we remove these coordinates?
        coord_names = ["field", "theta", "kx", "ky", "time"]
        eigenfunctions = np.empty(
            [len(coords[coord_name]) for coord_name in coord_names], dtype=complex
        )

        # Loop through all fields and add eigenfunction if it exists
        for ifield, raw_eigenfunction in enumerate(raw_eig_data):
            if raw_eigenfunction is not None:
                eigenfunction = raw_eigenfunction[: np.prod(raw_shape)].reshape(
                    raw_shape, order="F"
                )
                eigenfunctions[ifield, ...] = eigenfunction[0] + 1j * eigenfunction[1]

        square_fields = np.sum(np.abs(eigenfunctions) ** 2, axis=0)
        field_amplitude = np.sqrt(np.trapz(square_fields, coords["theta"], axis=0)) / (
            2 * np.pi
        )
        result = eigenfunctions / field_amplitude

        return result
