import copy
import f90nml
import numpy as np
import pint
from cleverdict import CleverDict
from typing import Dict, Any, Optional
from ..typing import PathLike
from ..constants import pi, electron_mass, deuterium_mass
from ..local_species import LocalSpecies
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    LocalGeometryMillerTurnbull,
    default_miller_turnbull_inputs,
    default_miller_inputs,
)
from ..numerics import Numerics
from ..normalisation import ureg, SimulationNormalisation as Normalisation, convert_dict
from ..templates import gk_templates
from .GKInput import GKInput
import warnings

class GKInputGENE(GKInput):
    """
    Class that can read GENE input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GENE"
    default_file_name = "input.gene"
    norm_convention = "gene"

    pyro_gene_miller = {
        "q": ["geometry", "q0"],
        "kappa": ["geometry", "kappa"],
        "s_kappa": ["geometry", "s_kappa"],
        "delta": ["geometry", "delta"],
        "s_delta": ["geometry", "s_delta"],
        "shat": ["geometry", "shat"],
        "shift": ["geometry", "drr"],
    }

    pyro_gene_miller_default = {
        "q": None,
        "kappa": 1.0,
        "s_kappa": 0.0,
        "delta": 0.0,
        "s_delta": 0.0,
        "shat": 0.0,
        "shift": 0.0,
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
    }

    def read(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads GENE input file into a dictionary
        Uses default read, which assumes input is a Fortran90 namelist
        """
        return super().read(filename)

    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads GENE input file given as string
        Uses default read_str, which assumes input is a Fortran90 namelist
        """
        return super().read_str(input_string)

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid gene input file, and that it contains sufficient
        info for Pyrokinetics to work with
        """
        expected_keys = ["general", "geometry", "box"]
        if not self.verify_expected_keys(filename, expected_keys):
            raise ValueError(f"Unable to verify {filename} as GENE file")

    def write(self, filename: PathLike, float_format: str = "", local_norm=None):
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

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, local_norm.gene)

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
            if self.data.get("zeta", 0.0) != 0.0 or self.data.get("zeta", 0.0):
                return self.get_local_geometry_miller_turnbull()
            else:
                return self.get_local_geometry_miller()
        elif geometry_type == "circular":
            return self.get_local_geometry_circular()
        else:
            raise NotImplementedError(
                f"LocalGeometry type {geometry_type} not implemented for GENE"
            )

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GENE file
        """
        miller_data = default_miller_inputs()

        for (pyro_key, (gene_param, gene_key)), gene_default in zip(
            self.pyro_gene_miller.items(), self.pyro_gene_miller_default.values()
        ):
            miller_data[pyro_key] = self.data[gene_param].get(gene_key, gene_default)

        # TODO Need to handle case where minor_r not defined
        miller_data["Rmaj"] = self.data["geometry"].get("major_r", 1.0) / self.data[
            "geometry"
        ].get("minor_r", 1.0)
        miller_data["rho"] = (
            self.data["geometry"].get("trpeps", 0.0) * miller_data["Rmaj"]
        )

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying miller after creation
        beta = self.data["general"]["beta"]
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0 / beta)
        else:
            miller.B0 = None

        miller.beta_prime = -self.data["geometry"].get("amhd", 0.0) / (
            miller.q**2 * miller.Rmaj
        )

        dpdx = self.data["geometry"].get("dpdx_pm", -2)

        if dpdx != -2 and dpdx != -miller.beta_prime:
            if dpdx == -1:
                local_species = self.get_local_species()
                beta_prime_ratio = -miller.beta_prime / (
                local_species.a_lp * beta
                )
                if not np.isclose(beta_prime_ratio, 1.0):
                    warnings.warn("GENE dpdx_pm not set consistently with amhd - drifts may not behave as expected")
            else:
                warnings.warn("GENE dpdx_pm not set consistently with amhd - drifts may not behave as expected")

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

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        miller = LocalGeometryMillerTurnbull.from_gk_data(miller_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying miller after creation
        beta = self.data["general"]["beta"]
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0 / beta)
        else:
            miller.B0 = None

        miller.beta_prime = -self.data["geometry"].get("amhd", 0.0) / (
            miller.q**2 * miller.Rmaj
        )

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

        circular = LocalGeometryMillerTurnbull.from_gk_data(circular_data)

        beta = self.data["general"]["beta"]
        if beta != 0.0:
            circular.B0 = np.sqrt(1.0 / beta)
        else:
            circular.B0 = None

        return circular

    def get_local_species(self):
        """
        Load LocalSpecies object from GENE file
        """
        local_species = LocalSpecies()
        ion_count = 0

        a_minor_lref = self.data["geometry"].get("minor_r", 1.0)
        gene_nu_ei = self.data["general"]["coll"] / a_minor_lref

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

            species_data["a_lt"] = gene_data["omt"] * a_minor_lref
            species_data["a_ln"] = gene_data["omn"] * a_minor_lref
            species_data["vel"] = 0.0
            species_data["a_lv"] = 0.0

            if species_data.z == -1:
                name = "electron"
                species_data.nu = (
                    gene_nu_ei * 4 * (deuterium_mass / electron_mass) ** 0.5
                ) * (ureg.vref_nrl / ureg.lref_minor_radius)
            else:
                ion_count += 1
                name = f"ion{ion_count}"
                species_data.nu = None

            species_data.name = name

            # normalisations
            species_data.dens *= ureg.nref_electron
            species_data.mass *= ureg.mref_deuterium
            species_data.temp *= ureg.tref_electron
            species_data.z *= ureg.elementary_charge

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        # TODO is this normalisation handled by LocalSpecies itself? If so, can remove
        local_species.normalise()

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

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

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

        numerics_data["beta"] = self.data["general"]["beta"] * ureg.beta_ref_ee_B0

        return Numerics(numerics_data)

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
                template_file = gk_templates["GENE"]
            self.read(template_file)

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

        # Set GENE normalisation dependant on the value of minor_r
        if local_norm:
            if self.data["geometry"].get("major_r", 1.0) == 1.0:
                try:
                    local_norm.gene.lref = getattr(
                        local_norm.units, f"lref_major_radius_{local_norm.name}"
                    )
                except pint.errors.UndefinedUnitError:
                    local_norm.gene.lref = getattr(
                        local_norm.units, "lref_major_radius"
                    )
            elif self.data["geometry"]["minor_r"] == 1.0:
                try:
                    local_norm.gene.lref = getattr(
                        local_norm.units, f"lref_minor_radius_{local_norm.name}"
                    )
                except pint.errors.UndefinedUnitError:
                    local_norm.gene.lref = getattr(
                        local_norm.units, "lref_minor_radius"
                    )
            else:
                raise ValueError(
                    f'Only Lref = R_major or a_minor supported in GENE, {self.data["geometry"]["minor_r"]} {self.data["geometry"]["major_r"]}'
                )

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

            for key, val in self.pyro_gene_species.items():
                single_species[val] = local_species[name][key]

            # TODO Allow for major radius to be used as normalising length
            single_species["omt"] = local_species[name].a_lt
            single_species["omn"] = local_species[name].a_ln

        self.data["general"]["zeff"] = local_species.zeff

        beta_ref = local_norm.gene.beta if local_norm else 0.0
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

        if not local_norm:
            return

        for name, namelist in self.data.items():
            self.data[name] = convert_dict(namelist, local_norm.gene)
