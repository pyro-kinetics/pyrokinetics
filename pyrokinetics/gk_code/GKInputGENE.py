import copy
import f90nml
import numpy as np
from cleverdict import CleverDict
from typing import Dict, Any, Optional
from ..typing import PathLike
from ..constants import pi, electron_charge, electron_mass, deuterium_mass
from ..local_species import LocalSpecies
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    default_miller_inputs,
)
from ..numerics import Numerics
from ..templates import gk_templates
from .GKInput import GKInput


class GKInputGENE(GKInput):
    """
    Class that can read GENE input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GENE"
    default_file_name = "input.gene"

    pyro_gene_miller = {
        "q": ["geometry", "q0"],
        "kappa": ["geometry", "kappa"],
        "s_kappa": ["geometry", "s_kappa"],
        "delta": ["geometry", "delta"],
        "s_delta": ["geometry", "s_delta"],
        "shat": ["geometry", "shat"],
        "shift": ["geometry", "drr"],
    }

    pyro_gene_circular = {
        "q": ["geometry", "q0"],
        "shat": ["geometry", "shat"],
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

    def write(self, filename: PathLike, float_format: str = ""):
        """
        Write self.data to a gyrokinetics input file.
        Uses default write, which writes to a Fortan90 namelist
        """
        super().write(filename, float_format=float_format)

    def is_nonlinear(self) -> bool:
        return bool(self.data.get("nonlinear", 0))

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

        for pyro_key, (gene_param, gene_key) in self.pyro_gene_miller.items():
            miller_data[pyro_key] = self.data[gene_param][gene_key]

        miller_data["Rmaj"] = (
            self.data["geometry"]["major_r"] / self.data["geometry"]["minor_r"]
        )
        miller_data["rho"] = self.data["geometry"]["trpeps"] * miller_data["Rmaj"]

        # must construct using from_gk_data as we cannot determine bunit_over_b0 here
        miller = LocalGeometryMiller.from_gk_data(miller_data)

        # Assume pref*8pi*1e-7 = 1.0
        # FIXME Should not be modifying miller after creation
        # FIXME Is this assumption general enough? Can't we get pref from local_species?
        # FIXME B0 = None can cause problems when writing
        beta = self.data["general"]["beta"]
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0 / beta)
        else:
            miller.B0 = None

        if miller.B0 is not None:
            miller.beta_prime = -self.data["geometry"]["amhd"] / (
                miller.q**2 * miller.Rmaj
            )

        return miller

    # Treating circular as a special case of miller
    def get_local_geometry_circular(self) -> LocalGeometryMiller:
        """
        Load Circular object from GENE file
        """
        circular_data = default_miller_inputs()

        for pyro_key, (gene_param, gene_key) in self.pyro_gene_circular.items():
            circular_data[pyro_key] = self.data[gene_param][gene_key]
        circular_data["local_geometry"] = "Miller"

        circular_data["Rmaj"] = (
            self.data["geometry"]["major_r"] / self.data["geometry"]["minor_r"]
        )
        circular_data["rho"] = self.data["geometry"]["trpeps"] * circular_data["Rmaj"]

        circular = LocalGeometryMiller.from_gk_data(circular_data)

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

        gene_nu_ei = self.data["general"]["coll"]

        # Load each species into a dictionary
        for i_sp in range(self.data["box"]["n_spec"]):
            species_data = CleverDict()

            gene_data = self.data["species"][i_sp]

            for pyro_key, gene_key in self.pyro_gene_species.items():
                species_data[pyro_key] = gene_data[gene_key]

            species_data["a_lt"] = gene_data["omt"] * self.data["geometry"]["minor_r"]
            species_data["a_ln"] = gene_data["omn"] * self.data["geometry"]["minor_r"]
            species_data["vel"] = 0.0
            species_data["a_lv"] = 0.0

            if species_data.z == -1:
                name = "electron"
                te = species_data.temp
                ne = species_data.dens
                me = species_data.mass

                species_data.nu = (
                    gene_nu_ei * 4 * (deuterium_mass / electron_mass) ** 0.5
                )

            else:
                ion_count += 1
                name = f"ion{ion_count}"
                species_data.nu = None

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        # TODO is this normalisation handled by LocalSpecies itself? If so, can remove
        for name in local_species.names:
            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

        nu_ee = local_species.electron.nu

        for ion in range(ion_count):
            key = f"ion{ion + 1}"

            nion = local_species[key]["dens"]
            tion = local_species[key]["temp"]
            mion = local_species[key]["mass"]
            # Not exact at log(Lambda) does change but pretty close...
            local_species[key]["nu"] = (
                nu_ee
                * (nion / tion**1.5 / mion**0.5)
                / (ne / te**1.5 / me**0.5)
            )

        return local_species

    def get_numerics(self) -> Numerics:
        """Gather numerical info (grid spacing, time steps, etc)"""

        numerics_data = dict()

        # Set number of fields
        numerics_data["phi"] = True

        numerics_data["apar"] = bool(self.data["general"].get("beta", 0))
        numerics_data["bpar"] = bool(self.data["general"].get("bpar", 0))

        numerics_data["delta_time"] = self.data["general"].get("DELTA_T", 0.01)
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
        numerics_data["nenergy"] = 0.5 * self.data["box"].get("nv0", 16)
        numerics_data["npitch"] = self.data["box"].get("nw0", 16)

        numerics_data["nonlinear"] = self.data.get("nonlinear", 0)

        if numerics_data["nonlinear"]:
            numerics_data["nkx"] = self.data["box"]["nx0"]
            numerics_data["nperiod"] = 1
        else:
            numerics_data["nkx"] = 1
            numerics_data["nperiod"] = self.data["box"]["nx0"] - 1

        return Numerics(numerics_data)

    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
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
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"Writing LocalGeometry type {local_geometry.__class__.__name__} "
                "for GENE not yet supported"
            )

        self.data["geometry"]["magn_geometry"] = "miller"
        for pyro_key, (gene_param, gene_key) in self.pyro_gene_miller.items():
            self.data[gene_param][gene_key] = local_geometry[pyro_key]

        self.data["geometry"]["amhd"] = (
            -(local_geometry.q**2) * local_geometry.Rmaj * local_geometry.beta_prime
        )
        self.data["geometry"]["trpeps"] = local_geometry.rho / local_geometry.Rmaj
        self.data["geometry"]["minor_r"] = 1.0
        self.data["geometry"]["major_r"] = local_geometry.Rmaj

        # Kinetic data
        self.data["box"]["n_spec"] = local_species.nspec

        for iSp, name in enumerate(local_species.names):
            if name == "electron":
                self.data["species"][iSp]["name"] = "electron"
            else:
                try:
                    self.data["species"][iSp]["name"] = "ion"
                except IndexError:
                    if f90nml.__version__ < "1.4":
                        self.data["species"].append(copy.copy(self.data["species"][0]))
                        self.data["species"][iSp]["name"] = "ion"
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
                        copied["name"] = "ion"
                        d[f"_grp_species_{iSp}"] = copied
                        self.data = f90nml.Namelist(d)

            for key, val in self.pyro_gene_species.items():
                self.data["species"][iSp][val] = local_species[name][key]

            self.data["species"][iSp]["omt"] = local_species[name].a_lt
            self.data["species"][iSp]["omn"] = local_species[name].a_ln

        # Calculate beta. If B0 is not defined, it takes the following
        # default value
        beta = 0.0
        if local_geometry.B0 is not None:
            # If species are defined...
            if local_species.nref is not None:
                pref = local_species.nref * local_species.tref * electron_charge
                bref = local_geometry.B0
                beta = pref / bref**2 * 8 * pi * 1e-7
            # Calculate from reference  at centre of flux surface
            else:
                beta = 1 / local_geometry.B0**2

        self.data["general"]["beta"] = beta

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
            self.data["box"]["nkx"] = numerics["nkx"]
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
