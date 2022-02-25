import f90nml
import copy

import numpy as np

from ..typing import PathLike
from ..constants import electron_charge, pi, sqrt2
from ..local_species import LocalSpecies
from ..numerics import Numerics
from .GKCode import GKCode
from .GKOutput import GKOutput
import os
from path import Path
from cleverdict import CleverDict


class GS2(GKCode):
    """
    Basic GS2 object

    """

    def __init__(self):

        self.base_template_file = os.path.join(
            Path(__file__).dirname(), "..", "templates", "input.gs2"
        )
        self.code_name = "GS2"
        self.default_file_name = "input.in"

    def read(self, pyro, data_file=None, template=False):
        """
        Reads GS2 input file into a dictionary
        """

        if template and data_file is None:
            data_file = self.base_template_file

        gs2 = f90nml.read(data_file).todict()

        pyro.initial_datafile = copy.copy(gs2)

        if gs2["kt_grids_knobs"]["grid_option"] in ["single", "default"]:
            pyro.linear = True
        else:
            pyro.linear = False
            if "wstar_units" in gs2["knobs"].keys():
                del gs2["knobs"]["wstar_units"]

        pyro.gs2_input = gs2

        # Loads pyro object with equilibrium data
        if not template:
            self.load_pyro(pyro)

        # Load Pyro with numerics if they don't exist
        if not hasattr(pyro, "numerics"):
            self.load_numerics(pyro, gs2)

    def verify(self, filename: PathLike):
        """
        Ensure this file is a valid gs2 input file"
        """
        data = f90nml.read(filename).todict()
        expected_keys = ["kt_grids_knobs", "theta_grid_knobs", "theta_grid_eik_knobs"]
        if not np.all(np.isin(expected_keys, list(data.keys()))):
            raise ValueError(f"Expected GS2 file, received {filename}")

    def load_pyro(self, pyro):
        """
        Loads GS2 input into Pyro object
        """

        # Geometry
        gs2 = pyro.gs2_input

        gs2_eq = gs2["theta_grid_knobs"]["equilibrium_option"]

        if gs2_eq in ["eik", "default"]:

            try:
                local_eq = gs2["theta_grid_eik_knobs"]["local_eq"]
            except KeyError:
                local_eq = True

            try:
                iflux = gs2["theta_grid_eik_knobs"]["iflux"]
            except KeyError:
                iflux = 0

            if local_eq:
                if iflux == 0:
                    pyro.local_geometry = "Miller"
                else:
                    pyro.local_geometry = "Fourier"

                self.load_local_geometry(pyro, gs2)

            else:
                pyro.local_geometry = "Global"

        else:
            raise NotImplementedError(
                f"GS2 equilibrium option {gs2_eq} not implemented"
            )

        # Load GS2 with species data
        self.load_local_species(pyro, gs2)

        # Load Pyro with numerics
        self.load_numerics(pyro, gs2)

    def write(self, pyro, filename, directory="."):
        """
        For a given pyro object write a GS2 input file

        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        path_to_file = os.path.join(directory, filename)

        gs2_input = pyro.gs2_input

        # Geometry data
        if pyro.local_geometry_type == "Miller":
            miller = pyro.local_geometry

            # Ensure Miller settings in input file
            gs2_input["theta_grid_knobs"]["equilibrium_option"] = "eik"
            gs2_input["theta_grid_eik_knobs"]["iflux"] = 0
            gs2_input["theta_grid_eik_knobs"]["local_eq"] = True
            gs2_input["theta_grid_parameters"]["geoType"] = 0

            # Reference B field
            bref = miller.B0

            shat = miller.shat
            # Assign Miller values to input file
            pyro_gs2_miller = self.pyro_to_code_miller()

            for key, val in pyro_gs2_miller.items():
                gs2_input[val[0]][val[1]] = miller[key]

            gs2_input["theta_grid_parameters"]["akappri"] = (
                miller.s_kappa * miller.kappa / miller.rho
            )
            gs2_input["theta_grid_parameters"]["tri"] = np.arcsin(miller.delta)
            gs2_input["theta_grid_parameters"]["tripri"] = (
                miller["s_delta"] / miller.rho
            )
            gs2_input["theta_grid_parameters"]["Rgeo"] = miller.Rmaj

        else:
            raise NotImplementedError(
                f"Writing {pyro.local_geometry_type} for GS2 not supported yet"
            )

        # Kinetic data
        local_species = pyro.local_species
        gs2_input["species_knobs"]["nspec"] = local_species.nspec

        pyro_gs2_species = self.pyro_to_code_species()

        for iSp, name in enumerate(local_species.names):

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

            beta = pref / bref**2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if pyro.local_geometry_type == "Miller":
                miller = pyro.local_geometry
                if miller.B0 is not None:
                    beta = 1 / miller.B0**2

                else:
                    beta = 0.0
            else:
                raise NotImplementedError

        gs2_input["parameters"]["beta"] = beta

        # Numerics
        numerics = pyro.numerics

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

        gs2_nml = f90nml.Namelist(gs2_input)
        gs2_nml.float_format = pyro.float_format
        gs2_nml.write(path_to_file, force=True)

    def load_local_geometry(self, pyro, gs2):
        """
        Loads local geometry
        """

        if pyro.local_geometry_type == "Miller":
            self.load_miller(pyro, gs2)

    def load_miller(self, pyro, gs2):
        """
        Load Miller object from GS2 file
        """

        # Set some defaults here
        gs2["theta_grid_eik_knobs"]["bishop"] = 4
        gs2["theta_grid_eik_knobs"]["irho"] = 2
        gs2["theta_grid_eik_knobs"]["iflux"] = 0
        pyro_gs2_miller = self.pyro_to_code_miller()

        miller = pyro.local_geometry

        for key, val in pyro_gs2_miller.items():
            miller[key] = gs2[val[0]][val[1]]

        miller.delta = np.sin(gs2["theta_grid_parameters"]["tri"])
        miller.s_kappa = (
            gs2["theta_grid_parameters"]["akappri"] * miller.rho / miller.kappa
        )
        miller.s_delta = gs2["theta_grid_parameters"]["tripri"] * miller.rho

        # Get beta and beta_prime normalised to R_major(in case R_geo != R_major)
        Rgeo = gs2["theta_grid_parameters"].get("Rgeo", miller.Rmaj)

        beta = gs2["parameters"]["beta"] * (miller.Rmaj / Rgeo) ** 2
        miller.beta_prime *= (miller.Rmaj / Rgeo) ** 2

        # Can only know Bunit/B0 from local Miller
        miller.bunit_over_b0 = miller.get_bunit_over_b0()

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0 / beta)
        else:
            # If beta = 0
            miller.B0 = None

    def load_local_species(self, pyro, gs2):
        """
        Load LocalSpecies object from GS2 file
        """
        nspec = gs2["species_knobs"]["nspec"]
        pyro_gs2_species = self.pyro_to_code_species()

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nspec = nspec
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(nspec):

            species_data = CleverDict()

            gs2_key = f"species_parameters_{i_sp + 1}"

            gs2_data = gs2[gs2_key]

            for pyro_key, gs2_key in pyro_gs2_species.items():
                species_data[pyro_key] = gs2_data[gs2_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
                te = species_data.temp
                ne = species_data.dens
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            # Account for sqrt(2) in vth
            species_data.nu = gs2_data["vnewk"] * sqrt2

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        for name in local_species.names:
            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

        # Add local_species
        pyro.local_species = local_species

    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and gs2 parameter names
        for miller parameters
        """

        pyro_gs2_param = {
            "rho": ["theta_grid_parameters", "rhoc"],
            "Rmaj": ["theta_grid_parameters", "rmaj"],
            "q": ["theta_grid_parameters", "qinp"],
            "kappa": ["theta_grid_parameters", "akappa"],
            "shat": ["theta_grid_eik_knobs", "s_hat_input"],
            "shift": ["theta_grid_parameters", "shift"],
            "beta_prime": ["theta_grid_eik_knobs", "beta_prime_input"],
        }

        return pyro_gs2_param

    def pyro_to_code_species(self):
        """
        Generates dictionary of equivalent pyro and gs2 parameter names
        for miller parameters
        """

        pyro_gs2_species = {
            "mass": "mass",
            "z": "z",
            "dens": "dens",
            "temp": "temp",
            "nu": "vnewk",
            "a_lt": "tprim",
            "a_ln": "fprim",
            "a_lv": "uprim",
        }

        return pyro_gs2_species

    def add_flags(self, pyro, flags):
        """
        Add extra flags to GS2 input file

        """

        for key, parameter in flags.items():
            for param, val in parameter.items():
                pyro.gs2_input[key][param] = val

    def load_numerics(self, pyro, gs2):
        """
        Load GS2 numerics into Pyro

        """

        grid_type = gs2["kt_grids_knobs"]["grid_option"]
        numerics = Numerics()

        # Set no. of fields
        numerics.phi = gs2["knobs"].get("fphi", 0.0) > 0.0
        numerics.apar = gs2["knobs"].get("fapar", 0.0) > 0.0
        numerics.bpar = gs2["knobs"].get("fbpar", 0.0) > 0.0

        # Set time stepping
        numerics.delta_time = gs2["knobs"].get("delt", 0.005) / sqrt2
        numerics.max_time = gs2["knobs"].get("nstep", 50000) * numerics.delta_time

        # Need shear for map theta0 to kx
        shat = pyro.local_geometry.shat

        # Fourier space grid
        # Linear simulation
        if grid_type in ["single", "default"]:
            numerics.nky = 1
            numerics.nkx = 1
            numerics.ky = gs2["kt_grids_single_parameters"]["aky"] / sqrt2

            numerics.kx = 0.0

            try:
                numerics.theta0 = gs2["kt_grids_single_parameters"]["theta0"]
            except KeyError:
                numerics.theta0 = 0.0

        # Nonlinear/multiple modes in box
        elif grid_type == "box":
            box = "kt_grids_box_parameters"
            keys = gs2[box].keys()

            # Set up ky grid
            if "ny" in keys:
                numerics.nky = int((gs2[box]["ny"] - 1) / 3 + 1)
            elif "n0" in keys:
                numerics.nky = gs2[box]["n0"]
            elif "nky" in keys:
                numerics.nky = gs2[box]["naky"]
            else:
                raise NotImplementedError(f"ky grid details not found in {keys}")

            if "y0" in keys:
                if gs2[box]["y0"] < 0.0:
                    numerics.ky = -gs2[box]["y0"] / sqrt2
                else:
                    numerics.ky = 1 / gs2[box]["y0"] / sqrt2
            else:
                raise NotImplementedError(f"Min ky details not found in {keys}")

            if "nx" in keys:
                numerics.nkx = int((2 * gs2[box]["nx"] - 1) / 3 + 1)
            elif "ntheta0" in keys():
                numerics.nkx = int((2 * gs2[box]["ntheta0"] - 1) / 3 + 1)
            else:
                raise NotImplementedError("kx grid details not found in {keys}")

            if abs(shat) > 1e-6:
                numerics.kx = numerics.ky * shat * 2 * pi / gs2[box]["jtwist"]
            else:
                numerics.kx = 2 * pi / gs2[box]["x0"] / sqrt2

        # Theta grid
        numerics.ntheta = gs2["theta_grid_parameters"]["ntheta"]
        numerics.nperiod = gs2["theta_grid_parameters"]["nperiod"]

        # Velocity grid
        try:
            numerics.nenergy = (
                gs2["le_grids_knobs"]["nesub"] + gs2["le_grids_knobs"]["nesuper"]
            )
        except KeyError:
            numerics.nenergy = gs2["le_grids_knobs"]["negrid"]

        # Currently using number of un-trapped pitch angles
        numerics.npitch = gs2["le_grids_knobs"]["ngauss"] * 2

        try:
            nl_mode = gs2["nonlinear_terms_knobs"]["nonlinear_mode"]
        except KeyError:
            nl_mode = "off"

        if nl_mode == "on":
            numerics.nonlinear = True
        else:
            numerics.nonlinear = False

        pyro.numerics = numerics

    def load_gk_output(self, pyro):
        """
        Loads GK Outputs
        """

        pyro.gk_output = GKOutput()

        self.load_grids(pyro)

        self.load_fields(pyro)

        self.load_fluxes(pyro)

        if not pyro.numerics.nonlinear:

            self.load_eigenvalues(pyro)

            self.load_eigenfunctions(pyro)

    def load_grids(self, pyro):
        """
        Loads GS2 grids to GKOutput

        out.cgyro.grids stores all the grid data in one long 1D array
        Output is in a standardised order

        """

        import xarray as xr
        import netCDF4 as nc

        gk_output = pyro.gk_output

        run_directory = pyro.run_directory
        netcdf_file_name = Path(pyro.file_name).with_suffix(".out.nc")

        netcdf_path = os.path.join(run_directory, netcdf_file_name)

        netcdf_data = nc.Dataset(netcdf_path)

        ky = netcdf_data["ky"][:] / sqrt2
        gk_output.ky = ky
        gk_output.nky = len(ky)

        try:
            if pyro.gs2_input["knobs"]["wstar_units"]:
                time = netcdf_data["t"][:] / ky
            else:
                time = netcdf_data["t"][:] / sqrt2

        except KeyError:
            time = netcdf_data["t"][:] / sqrt2

        gk_output.time = time
        gk_output.ntime = len(time)

        # Shift kx=0 to middle of array
        kx = np.fft.fftshift(netcdf_data["kx"][:]) / sqrt2

        gk_output.kx = kx
        gk_output.nkx = len(kx)

        nspecies = netcdf_data.dimensions["species"].size
        gk_output.nspecies = nspecies

        theta = netcdf_data["theta"][:]
        gk_output.theta = theta
        gk_output.ntheta = len(theta)

        try:
            energy = netcdf_data["egrid"][:]
        except IndexError:
            energy = netcdf_data["energy"][:]

        gk_output.energy = energy
        gk_output.nenergy = len(energy)

        pitch = netcdf_data["lambda"][:]
        gk_output.pitch = pitch
        gk_output.npitch = len(pitch)

        gs2_knobs = pyro.gs2_input["knobs"]
        nfield = 0
        if gs2_knobs["fphi"] > 0.0:
            nfield += 1
        if gs2_knobs["fapar"] > 0.0:
            nfield += 1
        if gs2_knobs["fbpar"] > 0.0:
            nfield += 1

        gk_output.nfield = nfield

        field = ["phi", "apar", "bpar"]
        field = field[:nfield]

        moment = ["particle", "energy", "momentum"]
        species = pyro.local_species.names

        # Store grid data as xarray DataSet
        ds = xr.Dataset(
            coords={
                "time": time,
                "field": field,
                "moment": moment,
                "species": species,
                "kx": kx,
                "ky": ky,
                "theta": theta,
            }
        )

        gk_output.data = ds

    def load_fields(self, pyro):
        """
        Loads 3D fields into GKOutput.data DataSet
        pyro.gk_output.data['fields'] = fields(field, theta, kx, ky, time)
        """

        import netCDF4 as nc

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory
        netcdf_file_name = Path(pyro.file_name).with_suffix(".out.nc")

        netcdf_path = os.path.join(run_directory, netcdf_file_name)

        netcdf_data = nc.Dataset(netcdf_path)

        fields = np.empty(
            (
                gk_output.nfield,
                gk_output.nkx,
                gk_output.ntheta,
                gk_output.nky,
                gk_output.ntime,
            ),
            dtype=np.complex,
        )

        field_appendices = ["phi_t", "apar_t", "bpar_t"]

        # Loop through all fields and add field in it exists
        for ifield, field_appendix in enumerate(field_appendices):

            raw_field = netcdf_data[field_appendix][:] * sqrt2
            field_data = np.moveaxis(raw_field, [0, 1, 2, 3, 4], [4, 3, 1, 2, 0])

            fields[ifield, :, :, :, :] = (
                field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]
            )

        # Shift kx=0 to middle of axis
        fields = np.fft.fftshift(fields, axes=1)

        data["fields"] = (("field", "kx", "theta", "ky", "time"), fields)

    def load_fluxes(self, pyro):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """

        import netCDF4 as nc
        import logging

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory
        netcdf_file_name = Path(pyro.file_name).with_suffix(".out.nc")

        netcdf_path = os.path.join(run_directory, netcdf_file_name)

        netcdf_data = nc.Dataset(netcdf_path)

        field_keys = ["es", "apar", "bpar"]
        if pyro.numerics.nonlinear:
            moment_keys = ["part_by_k", "heat_by_k", "mom_by_k"]
        else:
            moment_keys = ["part_flux", "heat_flux", "mom_flux"]

        fluxes = np.empty(
            (gk_output.nspecies, 3, gk_output.nfield, gk_output.nky, gk_output.ntime)
        )

        if f"{field_keys[0]}_{moment_keys[0]}" not in netcdf_data.variables.keys():
            logging.warning("Flux data not written to netCDF file, setting fluxes to 0")

        else:
            for ifield, field in enumerate(field_keys):
                for imoment, moment in enumerate(moment_keys):
                    key = f"{field}_{moment}"

                    if pyro.numerics.nonlinear:
                        # Sum over kx
                        flux = np.sum(netcdf_data[key], axis=-1)
                        flux = np.moveaxis(flux, [1, 2, 0], [0, 1, 2])

                        # Divide non-zonal components by 2 due to reality condition
                        flux[:, 1:, :] *= 0.5

                    else:
                        flux = np.swapaxes(netcdf_data[key], 0, 1)
                        flux = flux[:, np.newaxis, :]

                    fluxes[:, imoment, ifield, :, :] = flux

        data["fluxes"] = (("species", "moment", "field", "ky", "time"), fluxes)
