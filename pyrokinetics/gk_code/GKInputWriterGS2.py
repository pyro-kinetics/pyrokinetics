import f90nml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from copy import copy

from ..typing import PathLike
from ..constants import sqrt2, pi, electron_charge
from ..local_geometry import LocalGeometry, LocalGeometryMiller
from ..local_species import LocalSpecies
from ..numerics import Numerics
from ..templates import template_dir
from .gs2_utils import pyro_gs2_miller, pyro_gs2_species
from .GKInputWriter import GKInputWriter
from .GKInputReaderGS2 import GKInputReaderGS2


class GKInputWriterGS2(GKInputWriter):
    def __init__(self, template_file: Optional[PathLike] = None):
        # Use default template file if none exists
        if template_file is None:
            template_file = template_dir / "input.gs2"
        self.template_file = template_file

    def write(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        filename: PathLike = "input.in",
        directory: Optional[PathLike] = None,
        float_format: str = "",
    ):
        """
        For a given pyro object write a GS2 input file

        """
        # Read template file to get starting dict
        gs2_input = GKInputReaderGS2(self.template_file).data

        # Geometry data
        if isinstance(local_geometry, LocalGeometryMiller):
            gs2_input = self.miller_to_gs2_input(gs2_input, local_geometry)
        else:
            raise NotImplementedError(
                f"LocalGeometry type {local_geometry.__class__.__name__} for GS2 not supported yet"
            )

        # Kinetic data
        gs2_input = self.local_species_to_gs2_input(
            gs2_input, local_species, local_geometry
        )

        # Numerics
        gs2_input = self.numerics_to_gs2_input(gs2_input, numerics, local_geometry)

        # Create directories if they don't exist already
        filename = Path(filename)
        if directory is not None:
            filename = Path(directory).joinpath(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Create Fortran namelist and write
        gs2_nml = f90nml.Namelist(gs2_input)
        gs2_nml.float_format = float_format
        gs2_nml.write(filename, force=True)

    @staticmethod
    def miller_to_gs2_input(gs2_input: Dict[str, Any], miller: LocalGeometryMiller):
        """Modifies gs2_input dict to include data from LocalGeometryMiller"""
        # Ensure Miller settings
        gs2_input["theta_grid_knobs"]["equilibrium_option"] = "eik"
        gs2_input["theta_grid_eik_knobs"]["iflux"] = 0
        gs2_input["theta_grid_eik_knobs"]["local_eq"] = True
        gs2_input["theta_grid_parameters"]["geoType"] = 0

        # Assign Miller values to input file
        for key, val in pyro_gs2_miller.items():
            gs2_input[val[0]][val[1]] = miller[key]

        gs2_input["theta_grid_parameters"]["akappri"] = (
            miller.s_kappa * miller.kappa / miller.rho
        )
        gs2_input["theta_grid_parameters"]["tri"] = np.arcsin(miller.delta)
        gs2_input["theta_grid_parameters"]["tripri"] = miller["s_delta"] / miller.rho
        gs2_input["theta_grid_parameters"]["Rgeo"] = miller.Rmaj

        return gs2_input

    @staticmethod
    def local_species_to_gs2_input(
        gs2_input: Dict[str, Any],
        local_species: LocalSpecies,
        local_geometry: LocalGeometry,
    ):
        """Modifies gs2_input dict to include data from LocalSpecies"""
        gs2_input["species_knobs"]["nspec"] = local_species.nspec
        for iSp, name in enumerate(local_species.names):

            # add new outer params for each species
            species_key = f"species_parameters_{iSp + 1}"

            if name == "electron":
                gs2_input[species_key]["type"] = "electron"
            else:
                try:
                    gs2_input[species_key]["type"] = "ion"
                except KeyError:
                    gs2_input[species_key] = copy.copy(
                        gs2_input["species_parameters_1"]
                    )
                    gs2_input[species_key]["type"] = "ion"

                    gs2_input[f"dist_fn_species_knobs_{iSp + 1}"] = gs2_input[
                        f"dist_fn_species_knobs_{iSp}"
                    ]

            for key, val in pyro_gs2_species.items():
                gs2_input[species_key][val] = local_species[name][key]

            # Account for sqrt(2) in vth
            gs2_input[species_key]["vnewk"] = local_species[name]["nu"] / sqrt2

        # If species are defined calculate beta
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge
            bref = local_geometry.B0

            beta = pref / bref**2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if isinstance(local_geometry, LocalGeometryMiller):
                if local_geometry.B0 is not None:
                    beta = 1 / local_geometry.B0**2

                else:
                    beta = 0.0
            else:
                raise NotImplementedError

        gs2_input["parameters"]["beta"] = beta
        return gs2_input

    @staticmethod
    def numerics_to_gs2_input(
        gs2_input: Dict[str, Any],
        numerics: Numerics,
        local_geometry: LocalGeometry,
    ):
        """Modifies gs2_input dict to include data from Numerics"""
        # Set no. of fields
        gs2_input["knobs"]["fphi"] = 1.0 if numerics.phi else 0.0
        gs2_input["knobs"]["fapar"] = 1.0 if numerics.apar else 0.0
        gs2_input["knobs"]["fbpar"] = 1.0 if numerics.bpar else 0.0

        # Set time stepping
        gs2_input["knobs"]["delt"] = numerics.delta_time * sqrt2
        gs2_input["knobs"]["nstep"] = int(numerics.max_time / numerics.delta_time)

        if numerics.nky == 1:
            gs2_input["kt_grids_knobs"]["grid_option"] = "single"

            if "kt_grids_single_parameters" not in gs2_input.keys():
                gs2_input["kt_grids_single_parameters"] = {}

            gs2_input["kt_grids_single_parameters"]["aky"] = numerics.ky * sqrt2
            gs2_input["kt_grids_single_parameters"]["theta0"] = numerics.theta0
            gs2_input["theta_grid_parameters"]["nperiod"] = numerics.nperiod

        else:
            gs2_input["kt_grids_knobs"]["grid_option"] = "box"

            if "kt_grids_box_parameters" not in gs2_input.keys():
                gs2_input["kt_grids_box_parameters"] = {}

            gs2_input["kt_grids_box_parameters"]["nx"] = int(
                ((numerics.nkx - 1) * 3 / 2) + 1
            )
            gs2_input["kt_grids_box_parameters"]["ny"] = int(
                ((numerics.nky - 1) * 3) + 1
            )

            gs2_input["kt_grids_box_parameters"]["y0"] = -numerics.ky * sqrt2

            # Currently forces NL sims to have nperiod = 1
            gs2_input["theta_grid_parameters"]["nperiod"] = 1

            shat = local_geometry.shat
            if abs(shat) < 1e-6:
                gs2_input["kt_grids_box_parameters"]["x0"] = (
                    2 * pi / numerics.kx / sqrt2
                )
            else:
                gs2_input["kt_grids_box_parameters"]["jtwist"] = int(
                    (numerics.ky * shat * 2 * pi / numerics.kx) + 0.1
                )

        gs2_input["theta_grid_parameters"]["ntheta"] = numerics.ntheta

        gs2_input["le_grids_knobs"]["negrid"] = numerics.nenergy
        gs2_input["le_grids_knobs"]["ngauss"] = numerics.npitch // 2

        if numerics.nonlinear:
            if "nonlinear_terms_knobs" not in gs2_input.keys():
                gs2_input["nonlinear_terms_knobs"] = {}

            gs2_input["nonlinear_terms_knobs"]["nonlinear_mode"] = "on"
        else:
            try:
                gs2_input["nonlinear_terms_knobs"]["nonlinear_mode"] = "off"
            except KeyError:
                pass
        return gs2_input
