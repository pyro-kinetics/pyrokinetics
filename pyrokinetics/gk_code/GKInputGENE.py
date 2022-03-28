import numpy as np
import f90nml
from cleverdict import CleverDict
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..typing import PathLike
from ..constants import pi, electron_charge, electron_mass,  deuterium_mass
from ..local_species import LocalSpecies
from ..local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    get_default_miller_inputs,
)
from ..numerics import Numerics
from ..templates import template_dir
from .GKInput import GKInput


class GKInputGENE(GKInput):
    """
    Class that can read GENE input files, and produce
    Numerics, LocalSpecies, and LocalGeometry objects
    """

    code_name = "GENE"

    pyro_gene_miller = {
        "rho": ["geometry", "minor_r"],
        "Rmaj": ["geometry", "major_r"],
        "q": ["geometry", "q0"],
        "kappa": ["geometry", "kappa"],
        "s_kappa": ["geometry", "s_kappa"],
        "delta": ["geometry", "delta"],
        "s_delta": ["geometry", "s_delta"],
        "shat": ["geometry", "shat"],
        "shift": ["geometry", "drr"],
    }

    pyro_gene_species = {
        "mass": "mass",
        "z": "charge",
        "dens": "dens",
        "temp": "temp",
        "a_lt": "omt",
        "a_ln": "omn",
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
        Uses default read, which assumes input is a Fortran90 namelist
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
        return bool(self.data.get("nonlinear",0))

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
        if geometry_type != "miller":
            raise NotImplementedError(
                f"LocalGeometry type {geometry_type} not implemented for GENE"
            )
        return self.get_local_geometry_miller()

    def get_local_geometry_miller(self) -> LocalGeometryMiller:
        """
        Load Miller object from GENE file
        """
        miller_data = get_default_miller_inputs()

        for pyro_key, (gene_param, gene_key) in self.pyro_gene_miller.items():
            miller_data[pyro_key] = self.data[gene_param][gene_key]

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

        # Need species to set up beta_prime
        local_species = self.get_local_species()
        if miller.B0 is not None:
            miller.beta_prime = -local_species.a_lp / miller.B0**2
        else:
            miller.beta_prime = 0.0

        return miller

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
            numerics_data["theta0"] = (
                -self.data["box"]["kx_center"]
                / (self.data["box"]["kymin"] * self.data["geometry"]["shat"])
            )
        except KeyError:
            numerics_data["theta0"] = 0.0

        numerics_data["nky"] = self.data["box"]["nky0"]
        numerics_data["ky"] = self.data["box"]["kymin"]

        # Set to zero if box.lx not present
        numerics_data["kx"] = 2 * pi / self.data["box"].get("lx", np.inf)

        # Velocity grid

        numerics_data["ntheta"] = self.data["box"].get("nz0",24)
        numerics_data["nenergy"] = 0.5 * self.data["box"].get("nv0", 16)
        numerics_data["npitch"] = self.data["box"].get("nw0", 16)

        numerics_data["nonlinear"] = self.data.get("nonlinear",0)

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
                template_file = template_dir / "input.gene"
            self.read(template_file)

        # Geometry data
        if not isinstance(local_geometry, LocalGeometryMiller):
            raise NotImplementedError(
                f"Writing LocalGeometry type {local_geometry.__class__.__name__} "
                "for GENE not yet supported"
            )

        self.data["geometry"]["magn_geometry"] = "miller"
        for pyro_key, (gene_param,gene_key) in self.pyro_gene_miller.items():
            self.data[gene_param][gene_key] = local_geometry[pyro_key]

        self.data["geometry"]["amhd"] = (
            -(local_geometry.q**2) * local_geometry.Rmaj * local_geometry.beta_prime
        )
        self.data["geometry"]["trpeps"] = local_geometry.rho / local_geometry.Rmaj

        # Kinetic data
        self.data["box"]["n_spec"] = local_species.nspec

        for iSp, name in enumerate(local_species.names):

            species_key = "species"

            if name == "electron":
                self.data["species"][iSp]["name"] = "electron"
            else:
                try:
                    self.data["species"][iSp]["name"] = "ion"
                except IndexError:
                    self.data["species"].append(
                        copy.copy(self.data["species"][0])
                    )
                    self.data["species"][iSp]["name"] = "ion"

            for key, val in self.pyro_gene_species.items():
                self.data["species"][iSp][val] = local_species[name][key]

        # If species are defined calculate beta
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge
            # FIXME: local_geometry.B0 may be None
            bref = local_geometry.B0 
            beta = pref / bref**2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if local_geometry.B0 is not None:
                beta = (local_geometry.Rgeo / (local_geometry.B0 * local_geometry.Rmaj)) ** 2
            else:
                beta = 0.0

        self.data["general"]["beta"] = beta

        self.data["general"]["coll"] = (
            local_species.electron.nu / (4 * np.sqrt(deuterium_mass / electron_mass))
        )

        # Numerics
        if numerics.bpar and not numerics.apar:
            raise ValueError("Can't have bpar without apar in GENE")

        self.data["general"]["bpar"] = numerics.bpar

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
