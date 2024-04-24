from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from cleverdict import CleverDict

from ..constants import deuterium_mass, electron_mass, hydrogen_mass, pi
from ..file_utils import FileReader
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    LocalGeometryMXH,
    default_miller_inputs,
    default_mxh_inputs,
)
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation
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


class GKInputTGLF(GKInput, FileReader, file_type="TGLF", reads=GKInput):
    """Reader for TGLF input files"""

    code_name = "TGLF"
    default_file_name = "input.TGLF"
    norm_convention = "cgyro"
    tglf_max_ntheta = 32
    _convention_dict = {}

    pyro_tglf_miller = {
        "rho": "rmin_loc",
        "Rmaj": "rmaj_loc",
        "q": "q_loc",
        "kappa": "kappa_loc",
        "s_kappa": "s_kappa_loc",
        "delta": "delta_loc",
        "shift": "drmajdx_loc",
    }

    pyro_tglf_miller_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "q": 2.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "shift": 0.0,
    }

    pyro_tglf_mxh = {
        "rho": "rmin_loc",
        "Rmaj": "rmaj_loc",
        "Z0": "zmaj_loc",
        "dZ0dr": "dzmajdx_loc",
        "q": "q_loc",
        "kappa": "kappa_loc",
        "s_kappa": "s_kappa_loc",
        "delta": "delta_loc",
        "s_delta": "s_delta_loc",
        "zeta": "zeta_loc",
        "s_zeta": "s_zeta_loc",
        "shift": "drmajdx_loc",
    }

    pyro_tglf_mxh_defaults = {
        "rho": 0.5,
        "Rmaj": 3.0,
        "Z0": 0.0,
        "dZ0dr": 0.0,
        "q": 2.0,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "zeta": 0.0,
        "s_zeta": 0.0,
        "shat": 1.0,
        "shift": 0.0,
    }

    @staticmethod
    def pyro_TGLF_species(iSp=1):
        return {
            "mass": f"mass_{iSp}",
            "z": f"zs_{iSp}",
            "dens": f"as_{iSp}",
            "temp": f"taus_{iSp}",
            "inverse_lt": f"rlts_{iSp}",
            "inverse_ln": f"rlns_{iSp}",
        }

    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads TGLF input file into a dictionary
        """
        with open(filename) as f:
            contents = f.read()

        return self.read_str(contents)

    def read_dict(self, input_dict: dict) -> Dict[str, Any]:
        """
        Reads TGLF input file given as dict
        Uses default read_dict, which assumes input is a dict
        """
        return super().read_dict(input_dict)

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads TGLF input file given as string
        """
        # TGLF input files are _almost_ Fortran namelists, so if we
        # change the comments to use '!' instead of '#', and wrap it
        # in a namelist syntax, we can just use the base `read_str`
        as_namelist = f"&nml\n{input_string.replace('#', '!')}\n/"

        # We need to strip off our fake namelist wrapper when we store
        # it internally
        self.data = super().read_str(as_namelist)["nml"]
        return self.data

    def verify_file_type(self, filename: PathLike):
        """
        Ensure this file is a valid TGLF input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """

        expected_keys = ["rmin_loc", "rmaj_loc", "nky"]
        self.verify_expected_keys(filename, expected_keys)

    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm: Normalisation = None,
        code_normalisation: str = None,
    ):
        """
        Write input file for TGLF
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        if local_norm is None:
            local_norm = Normalisation("write")

        if code_normalisation is None:
            code_normalisation = self.code_name.lower()

        convention = getattr(local_norm, code_normalisation)

        self.data = convert_dict(self.data, convention)

        with open(filename, "w+") as new_TGLF_input:
            for key, value in self.data.items():
                if isinstance(value, float):
                    value_str = f"{value:{float_format}}"
                elif isinstance(value, bool):
                    value_str = "T" if value else "F"
                else:
                    value_str = str(value)

                new_TGLF_input.write(f"{key.upper()} = {value_str}\n")

    def is_nonlinear(self) -> bool:
        return self.data.get("use_transport_model", 1) == 1

    def add_flags(self, flags) -> None:
        """
        Add extra flags to TGLF input file
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
            norms = Normalisation("get_local_species")
            convention = getattr(norms, self.norm_convention)

        tglf_eq_flag = self.data["geometry_flag"]
        tglf_eq_mapping = ["SAlpha", "MXH", "Fourier", "ELITE"]
        tglf_eq = tglf_eq_mapping[tglf_eq_flag]

        if tglf_eq == "MXH":
            if self.data.get("ZETA", 0.0) == 0 and self.data.get("S_ZETA", 0.0) == 0:
                tglf_eq = "Miller"

        if tglf_eq not in ["Miller", "MXH"]:
            raise NotImplementedError(
                f"TGLF equilibrium option '{tglf_eq_flag}' ('{tglf_eq}') not implemented"
            )

        if tglf_eq == "MXH":
            local_geometry = self.get_local_geometry_mxh()
        else:
            local_geometry = self.get_local_geometry_miller()

        local_geometry.normalise(norms=convention)

        return local_geometry

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from TGLF file
        """

        miller_data = default_miller_inputs()

        for (pyro_key, tglf_key), tglf_default in zip(
            self.pyro_tglf_miller.items(), self.pyro_tglf_miller_defaults.values()
        ):
            miller_data[pyro_key] = self.data.get(tglf_key, tglf_default)

        miller_data["s_delta"] = self.data.get("s_delta_loc", 0.0) / np.sqrt(
            1 - miller_data["delta"] ** 2
        )
        miller_data["shat"] = (
            self.data.get("q_prime_loc", 16.0)
            * (miller_data["rho"] / miller_data["q"]) ** 2
        )

        miller_data["ip_ccw"] = 1
        miller_data["bt_ccw"] = 1
        # Must construct using from_gk_data as we cannot determine
        # bunit_over_b0 here. We also need it to set B0 and
        # beta_prime, so we have to make a miller instance first
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data.get("betae", 0.0) * ne_norm * Te_norm
        miller.B0 = 1 / beta**0.5 if beta != 0 else None

        # FIXME: This actually needs to be scaled (or overwritten?) by
        # local_species.inverse_lp and self.data["BETA_STAR_SCALE"]. So we
        # need to get all the species data first?
        miller.beta_prime = (
            self.data.get("p_prime_loc", 0.0)
            * miller_data["rho"]
            / miller_data["q"]
            * miller.bunit_over_b0**2
            * (8 * np.pi)
        )

        return miller

    def get_local_geometry_mxh(self) -> LocalGeometryMXH:
        """
        Load mxh object from TGLF file
        """

        mxh_data = default_mxh_inputs()

        for (pyro_key, tglf_key), tglf_default in zip(
            self.pyro_tglf_mxh.items(), self.pyro_tglf_mxh_defaults.values()
        ):
            mxh_data[pyro_key] = self.data.get(tglf_key, tglf_default)

        mxh_data["shat"] = (
            self.data.get("q_prime_loc", 16.0) * (mxh_data["rho"] / mxh_data["q"]) ** 2
        )

        # Must construct using from_gk_data as we cannot determine
        # bunit_over_b0 here. We also need it to set B0 and
        # beta_prime, so we have to make a mxh instance first
        mxh_data["ip_ccw"] = 1
        mxh_data["bt_ccw"] = 1

        mxh = LocalGeometryMXH.from_gk_data(mxh_data)

        ne_norm, Te_norm = self.get_ne_te_normalisation()
        beta = self.data.get("betae", 0.0) * ne_norm * Te_norm
        mxh.B0 = 1 / beta**0.5 if beta != 0 else None

        # FIXME: This actually needs to be scaled (or overwritten?) by
        # local_species.inverse_lp and self.data["BETA_STAR_SCALE"]. So we
        # need to get all the species data first?
        mxh.beta_prime = (
            self.data.get("p_prime_loc", 0.0)
            * mxh_data["rho"]
            / mxh_data["q"]
            * (8 * np.pi)
        )

        return mxh

    def get_local_species(self):
        """
        Load LocalSpecies object from TGLF file
        """
        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.names = []

        if hasattr(self, "convention"):
            convention = self.convention
        else:
            norms = Normalisation("get_local_species")

            convention = getattr(norms, self.norm_convention)

        ion_count = 0

        domega_drho = -self.data.get("vpar_shear_1", 0.0) / self.data["rmaj_loc"]
        # Load each species into a dictionary
        for i_sp in range(self.data["ns"]):
            pyro_TGLF_species = self.pyro_TGLF_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_TGLF_species.items():
                species_data[p_key] = self.data[c_key]

            species_data.omega0 = (
                self.data.get(f"vpar_{i_sp}", 0.0)
                * convention.vref
                / convention.lref
                / self.data["rmaj_loc"]
            )
            species_data.domega_drho = (
                domega_drho * convention.vref / convention.lref**2
            )

            if species_data.z == -1:
                name = "electron"
                species_data.nu = self.data["xnue"] * convention.vref / convention.lref
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

        # Get collision frequency of ion species
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

        local_species.normalise()

        local_species.zeff = self.data.get("zeff", 1.0) * convention.qref

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
        numerics_data["apar"] = bool(self.data.get("use_bper", False))
        numerics_data["bpar"] = bool(self.data.get("use_bpar", False))

        numerics_data["ky"] = self.data["ky"]

        numerics_data["nky"] = self.data.get("nky", 1)
        numerics_data["theta0"] = self.data.get("kx0_loc", 0.0) * 2 * pi
        numerics_data["ntheta"] = self.data.get("nxgrid", 16)
        numerics_data["nonlinear"] = self.is_nonlinear()

        numerics_data["beta"] = self.data["betae"]

        numerics_data["gamma_exb"] = self.data.get("vexb_shear", 0.0)

        return Numerics(**numerics_data).with_units(convention)

    def get_reference_values(self, local_norm: Normalisation) -> Dict[str, Any]:
        """
        Reads in reference values from input file

        """
        return {}

    def _detect_normalisation(self):
        """
        Determines the necessary inputs and passes information to the base method _detect_normalisation.
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
        }

        references = copy(default_references)

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

        for i_sp in range(self.data["ns"]):
            dens = self.data[f"as_{i_sp + 1}"]
            temp = self.data[f"taus_{i_sp + 1}"]
            mass = self.data[f"mass_{i_sp + 1}"]

            if self.data[f"zs_{i_sp + 1}"] == -1:
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

        major_radius = self.data["rmaj_loc"]
        minor_radius = 1.0

        super()._detect_normalisation(
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
                template_file = gk_templates["TGLF"]
            self.read_from_file(template_file)

        if local_norm is None:
            local_norm = Normalisation("set")

        if code_normalisation is None:
            code_normalisation = self.norm_convention

        convention = getattr(local_norm, code_normalisation)

        # Set Miller Geometry bits
        if isinstance(local_geometry, LocalGeometryMXH):
            eq_type = "MXH"
        elif isinstance(local_geometry, LocalGeometryMiller):
            eq_type = "Miller"
        else:
            raise NotImplementedError(
                f"Writing LocalGeometry type {local_geometry.__class__.__name__} "
                "for GENE not yet supported"
            )

        # Geometry (Miller/MXH)
        self.data["geometry_flag"] = 1

        if eq_type == "Miller":
            # Assign Miller values to input file
            for key, value in self.pyro_tglf_miller.items():
                self.data[value] = local_geometry[key]

            self.data["s_delta_loc"] = local_geometry.s_delta * np.sqrt(
                1 - local_geometry.delta**2
            )

        elif eq_type == "MXH":
            # Assign MXH values to input file
            for key, value in self.pyro_tglf_mxh.items():
                self.data[value] = getattr(local_geometry, key)

        self.data["q_prime_loc"] = (
            local_geometry.shat * (local_geometry.q / local_geometry.rho) ** 2
        )

        # Set local species bits
        self.data["ns"] = local_species.nspec
        for iSp, name in enumerate(local_species.names):
            tglf_species = self.pyro_TGLF_species(iSp + 1)

            for pyro_key, TGLF_key in tglf_species.items():
                self.data[TGLF_key] = local_species[name][pyro_key]

            self.data[f"vpar_{iSp+1}"] = (
                local_species[name]["omega0"] * self.data["rmaj_loc"]
            )
            self.data[f"vpar_shear_{iSp+1}"] = (
                -local_species[name]["domega_drho"]
                * self.data["rmaj_loc"]
                * local_norm.tglf.lref
            )

        self.data["xnue"] = local_species.electron.nu

        self.data["zeff"] = local_species.zeff

        beta_ref = convention.beta if local_norm else 0.0
        self.data["betae"] = numerics.beta if numerics.beta is not None else beta_ref

        self.data["p_prime_loc"] = (
            local_geometry.beta_prime
            * local_geometry.q
            / local_geometry.rho
            / local_geometry.bunit_over_b0**2
            / (8 * np.pi)
        )

        # Numerics
        self.data["use_bper"] = numerics.apar
        self.data["use_bpar"] = numerics.bpar

        # Set time stepping
        self.data["use_transport_model"] = numerics.nonlinear

        self.data["ky"] = numerics.ky
        self.data["nky"] = numerics.nky

        self.data["nxgrid"] = min(numerics.ntheta, self.tglf_max_ntheta)
        self.data["kx0_loc"] = numerics.theta0 / (2 * pi)

        if not numerics.nonlinear:
            self.data["write_wavefunction_flag"] = 1

        self.data["vexb_shear"] = numerics.gamma_exb

        if not local_norm:
            return

        self.data = convert_dict(self.data, convention)

    def get_ne_te_normalisation(self):
        found_electron = False
        for i_sp in range(self.data["ns"]):
            if self.data[f"zs_{i_sp+1}"] == -1:
                ne = self.data[f"as_{i_sp+1}"]
                Te = self.data[f"taus_{i_sp+1}"]
                found_electron = True
                break

        if not found_electron:
            raise TypeError(
                "Pyro currently requires an electron species in TGLF input files"
            )

        return ne, Te


class TGLFFile:
    def __init__(self, path: PathLike, required: bool):
        self.path = Path(path)
        self.required = required
        self.fmt = self.path.name.split(".")[0]


class GKOutputReaderTGLF(FileReader, file_type="TGLF", reads=GKOutput):
    def read_from_file(
        self,
        filename: PathLike,
        norm: SimulationNormalisation,
        downsize: int = 1,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
    ) -> GKOutput:
        raw_data, gk_input, input_str = self._get_raw_data(filename)
        coords = self._get_coords(raw_data, gk_input)
        fields = self._get_fields(raw_data, coords) if load_fields else None
        fluxes = self._get_fluxes(raw_data, coords) if load_fluxes else None
        moments = self._get_moments(raw_data, coords) if load_moments else None
        eigenvalues = self._get_eigenvalues(raw_data, coords, gk_input)
        eigenfunctions = (
            self._get_eigenfunctions(raw_data, coords) if coords["linear"] else None
        )

        # Assign units and return GKOutput
        convention = getattr(norm, gk_input.norm_convention)

        field_dims = ("ky", "mode")
        flux_dims = ("field", "species", "ky")
        moment_dims = ("field", "species", "ky")
        eigenvalues_dims = ("ky", "mode")
        eigenfunctions_dims = ("theta", "mode", "field")
        return GKOutput(
            coords=Coords(
                time=coords["time"],
                kx=coords["kx"],
                ky=coords["ky"],
                theta=coords["theta"],
                mode=coords["mode"],
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
                Eigenvalues(**eigenvalues, dims=eigenvalues_dims).with_units(convention)
                if eigenvalues
                else None
            ),
            eigenfunctions=(
                None
                if eigenfunctions is None
                else Eigenfunctions(eigenfunctions, dims=eigenfunctions_dims)
            ),
            linear=coords["linear"],
            gk_code="TGLF",
            input_file=input_str,
        )

    @staticmethod
    def _required_files(dirname: PathLike):
        dirname = Path(dirname)
        return {
            "input": TGLFFile(dirname / "input.tglf", required=True),
            "run": TGLFFile(dirname / "out.tglf.run", required=True),
        }

    def verify_file_type(self, dirname: PathLike):
        dirname = Path(dirname)
        for f in self._required_files(dirname).values():
            if not f.path.exists():
                raise RuntimeError(f"Couldn't find TGLF file '{f}'")

    @staticmethod
    def infer_path_from_input_file(filename: PathLike) -> Path:
        """
        Given path to input file, guess at the path for associated output files.
        For TGLF, simply returns dir of the path.
        """
        return Path(filename).parent

    @classmethod
    def _get_raw_data(
        cls, dirname: PathLike
    ) -> Tuple[Dict[str, Any], GKInputTGLF, str]:
        dirname = Path(dirname)
        if not dirname.exists():
            raise RuntimeError(
                f"GKOutputReaderTGLF: Provided path {dirname} does not exist. "
                "Please supply the name of a directory containing TGLF output files."
            )
        if not dirname.is_dir():
            raise RuntimeError(
                f"GKOutputReaderTGLF: Provided path {dirname} is not a directory. "
                "Please supply the name of a directory containing TGLF output files."
            )

        # The following list of TGLF files may exist
        expected_files = {
            **cls._required_files(dirname),
            "wavefunction": TGLFFile(dirname / "out.tglf.wavefunction", required=False),
            "run": TGLFFile(dirname / "out.tglf.run", required=False),
            "field": TGLFFile(dirname / "out.tglf.field_spectrum", required=False),
            "ky": TGLFFile(dirname / "out.tglf.ky_spectrum", required=False),
            "ql_flux": TGLFFile(dirname / "out.tglf.QL_flux_spectrum", required=False),
            "sum_flux": TGLFFile(
                dirname / "out.tglf.sum_flux_spectrum", required=False
            ),
            "eigenvalues": TGLFFile(
                dirname / "out.tglf.eigenvalue_spectrum", required=False
            ),
        }
        # Read in files
        raw_data = {}
        for key, tglf_file in expected_files.items():
            if not tglf_file.path.exists():
                if tglf_file.required:
                    raise RuntimeError(
                        f"GKOutputReaderTGLF: The file {tglf_file.path.name} is needed"
                    )
                continue
            # Read in file according to format
            if key == "ky":
                raw_data[key] = np.loadtxt(tglf_file.path, skiprows=2)

            else:
                with open(tglf_file.path, "r") as f:
                    raw_data[key] = f.read()

        input_str = raw_data["input"]
        gk_input = GKInputTGLF()
        gk_input.read_str(input_str)
        gk_input._detect_normalisation()

        return raw_data, gk_input, input_str

    @staticmethod
    def _get_coords(raw_data: Dict[str, Any], gk_input: GKInputTGLF) -> Dict[str, Any]:
        """
        Sets coords and attrs of a Pyrokinetics dataset from a collection of TGLF
        files.

        Args:
            raw_data (Dict[str,Any]): Dict containing TGLF output.
            gk_input (GKInputTGLF): Processed TGLF input file.

        Returns:
            Dict: Dict with coords
        """

        bunit_over_b0 = gk_input.get_local_geometry().bunit_over_b0.m

        if gk_input.is_linear():
            f = raw_data["wavefunction"].splitlines()
            grid = f[0].strip().split(" ")
            grid = [x for x in grid if x]

            nmode_data = int(grid[0])
            nmode = gk_input.data.get("nmodes", 2)
            nfield = int(grid[1])
            ntheta = int(grid[2])

            full_data = " ".join(f[2:]).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            full_data = np.reshape(full_data, (ntheta, (nmode_data * 2 * nfield) + 1))
            theta = full_data[:, 0]

            mode = list(range(1, 1 + nmode))
            field = ["phi", "apar", "bpar"][:nfield]
            species = gk_input.get_local_species().names

            run = raw_data["run"].splitlines()
            ky = (
                float([line for line in run if "ky" in line][0].split(":")[-1].strip())
                / bunit_over_b0
            )

            # Store grid data as Dict
            return {
                "flux": None,
                "moment": None,
                "species": species,
                "field": field,
                "theta": theta,
                "mode": mode,
                "ky": [ky],
                "kx": [0.0],
                "time": [0.0],
                "linear": gk_input.is_linear(),
            }
        else:
            raw_grid = raw_data["ql_flux"].splitlines()[3].split(" ")
            grids = [int(g) for g in raw_grid if g]

            nflux = grids[0]
            nspecies = grids[1]
            nfield = grids[2]
            nmode = grids[4]

            flux = ["particle", "heat", "momentum", "par_momentum", "exchange"][:nflux]
            species = gk_input.get_local_species().names
            if nspecies != len(species):
                raise RuntimeError(
                    "GKOutputReaderTGLF: Different number of species in input and output."
                )
            field = ["phi", "apar", "bpar"][:nfield]
            ky = raw_data["ky"] / bunit_over_b0
            mode = list(range(1, 1 + nmode))

            # Store grid data as xarray DataSet
            return {
                "flux": flux,
                "moment": None,
                "species": species,
                "field": field,
                "theta": None,
                "ky": ky,
                "kx": [0.0],
                "mode": mode,
                "time": [0.0],
                "linear": gk_input.is_linear(),
            }

    @staticmethod
    def _get_fields(
        raw_data: Dict[str, Any], coords: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Sets fields over  for eac ky.
        The field coordinates should be (ky, mode, field)
        """

        # Check to see if there's anything to do
        if "field" not in raw_data.keys():
            return {}

        nky = len(coords["ky"])
        nmode = len(coords["mode"])
        nfield = len(coords["field"])

        f = raw_data["field"].splitlines()

        full_data = " ".join(f[6:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        fields = np.reshape(full_data, (nky, nmode, 4))
        fields = fields[:, :, 1 : nfield + 1]

        results = {}
        for ifield, field_name in enumerate(coords["field"]):
            results[field_name] = fields[:, :, ifield]

        return results

    @staticmethod
    def _get_moments(
        raw_data: Dict[str, Any],
        gk_input: GKInputTGLF,
        coords: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Sets 3D moments over time.
        The moment coordinates should be (moment, theta, kx, species, ky, time)
        """
        raise NotImplementedError

    @staticmethod
    def _get_fluxes(
        raw_data: Dict[str, Any], coords: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Set flux data over time.
        The flux coordinates should be (species, field, ky, moment)
        """

        results = {}

        if "sum_flux" in raw_data:
            nky = len(coords["ky"])
            nfield = len(coords["field"])
            nflux = len(coords["flux"])
            nspecies = len(coords["species"])

            f = raw_data["sum_flux"].splitlines()
            full_data = [x for x in f if "species" not in x]
            full_data = " ".join(full_data).split(" ")

            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            fluxes = np.reshape(full_data, (nspecies, nfield, nky, nflux))
            # Order should be (flux, field, species, ky)
            fluxes = fluxes.transpose((3, 1, 0, 2))

            # Pyro doesn't handle parallel/exchange fluxs yet
            pyro_fluxes = ["particle", "heat", "momentum"]

            for iflux, flux in enumerate(coords["flux"]):
                if flux in pyro_fluxes:
                    results[flux] = fluxes[iflux, ...]

        return results

    @staticmethod
    def _get_eigenvalues(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
        gk_input: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Takes an xarray Dataset that has had coordinates and fields set.
        Uses this to add eigenvalues:

        data['eigenvalues'] = eigenvalues(ky, mode)
        data['mode_frequency'] = mode_frequency(ky, mode)
        data['growth_rate'] = growth_rate(ky, mode)

        This is only valid for transport runs.
        Unlike the version in the super() class, TGLF needs to get extra info from
        an eigenvalue file.

        Args:
            data: The Xarray dataset to be modified.
            dirname (PathLike): Directory containing TGLF output files.
        Returns:
            Xarray.Dataset: The modified dataset which was passed to 'data'.
        """

        results = {}
        # Use default method to calculate growth/freq if possible
        if "eigenvalues" in raw_data and not gk_input.is_linear():
            nky = len(coords["ky"])
            nmode = len(coords["mode"])

            f = raw_data["eigenvalues"].splitlines()

            full_data = " ".join(f).split(" ")
            full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

            eigenvalues = np.reshape(full_data, (nky, nmode, 2))
            eigenvalues = -eigenvalues[:, :, 1] + 1j * eigenvalues[:, :, 0]

            results["growth_rate"] = np.imag(eigenvalues)
            results["mode_frequency"] = np.real(eigenvalues)

        elif gk_input.is_linear():
            nmode = len(coords["mode"])
            f = raw_data["run"].splitlines()
            lines = f[-nmode:]

            eigenvalues = np.array(
                [
                    list(filter(None, eig.strip().split(":")[-1].split("  ")))
                    for eig in lines
                ],
                dtype="float",
            )

            eigenvalues = eigenvalues.reshape((1, nmode, 2))
            mode_frequency = eigenvalues[:, :, 0]
            growth_rate = eigenvalues[:, :, 1]

            eigenvalues = mode_frequency + 1j * growth_rate
            results["growth_rate"] = growth_rate
            results["mode_frequency"] = mode_frequency

        return results

    @staticmethod
    def _get_eigenfunctions(
        raw_data: Dict[str, Any],
        coords: Dict[str, Any],
    ) -> np.ndarray:
        """
        Returns eigenfunctions with the coordinates ``(mode, field, theta)``

        Only possible with single ky runs (USE_TRANSPORT_MODEL=False)
        """

        # Load wavefunction if file exists
        if "wavefunction" not in raw_data:
            return None

        f = raw_data["wavefunction"].splitlines()
        grid = f[0].strip().split(" ")
        grid = [x for x in grid if x]

        # In case no unstable modes are found
        nmode_data = int(grid[0])
        nmode = len(coords["mode"])
        nfield = len(coords["field"])
        ntheta = len(coords["theta"])

        eigenfunctions = np.zeros((ntheta, nmode, nfield), dtype="complex")

        full_data = " ".join(f[1:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]
        full_data = np.reshape(full_data, (ntheta, (nmode_data * 2 * nfield) + 1))

        reshaped_data = np.reshape(full_data[:, 1:], (ntheta, nmode_data, nfield, 2))

        eigenfunctions[:, :nmode_data, :] = (
            reshaped_data[:, :, :, 1] + 1j * reshaped_data[:, :, :, 0]
        )
        return eigenfunctions


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
