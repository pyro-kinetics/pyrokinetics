import copy

import numpy as np

from ..typing import PathLike
from ..constants import electron_charge, pi
from ..local_species import LocalSpecies
from ..numerics import Numerics
from ..templates import template_dir
from .GKCode import GKCode
from .GKOutput import GKOutput
import os
from cleverdict import CleverDict


class GKCodeTGLF(GKCode):
    """
    Basic TGLF object inheriting method from GKCode

    """

    def __init__(self):

        self.base_template_file = template_dir / "input.tglf"
        self.code_name = "TGLF"
        self.default_file_name = "input.tglf"

        self.gk_output = None

    def read(self, pyro, data_file=None, template=False):
        """
        Reads a TGLF input file and loads pyro object with the data
        """
        if template and data_file is None:
            data_file = self.base_template_file

        tglf = self.tglf_parser(data_file)

        pyro.initial_datafile = copy.copy(tglf)

        try:

            if tglf["USE_TRANSPORT_MODEL"] in ("T", True):
                pyro.linear = False
            else:
                pyro.linear = True
        except KeyError:
            pyro.linear = False

        pyro.tglf_input = tglf

        if not template:
            self.load_pyro(pyro)

        # Load Pyro with numerics if they don't exist yet
        if not hasattr(pyro, "numerics"):
            self.load_numerics(pyro, tglf)

    def verify(self, filename: PathLike):
        """read tglf file, check the dict returned holds the expected data"""
        data = self.tglf_parser(filename)
        expected_keys = ["RMIN_LOC", "RMAJ_LOC", "NKY"]
        if not np.all(np.isin(expected_keys, list(data.keys()))):
            raise ValueError(f"Expected tglf file, received {filename}")

    def load_pyro(self, pyro):
        """
        Loads LocalSpecies, LocalGeometry, Numerics classes from pyro.tglf_input

        """

        # Geometry
        tglf = pyro.tglf_input

        tglf_eq = tglf["GEOMETRY_FLAG"]

        if tglf_eq == 1:
            pyro.local_geometry = "Miller"
        elif tglf_eq == 2:
            pyro.local_geometry = "Fourier"
        elif tglf_eq == 0:
            pyro.local_geometry = "SAlpha"

        # Load local geometry
        self.load_local_geometry(pyro, tglf)

        # Load Species
        self.load_local_species(pyro, tglf)

        self.load_numerics(pyro, tglf)

    def write(self, pyro, file_name, directory="."):
        """
        Write a tglf input file from a pyro object

        """
        tglf_max_ntheta = 32

        tglf_input = pyro.tglf_input

        # Geometry data
        if pyro.local_geometry_type == "Miller":
            miller = pyro.local_geometry

            # Ensure Miller settings in input file
            tglf_input["GEOMETRY_FLAG"] = 1

            # Reference B field - Bunit = q/r dpsi/dr
            if miller.B0 is not None:
                b_ref = miller.B0 * miller.bunit_over_b0
            else:
                b_ref = None

            # Assign Miller values to input file
            pyro_tglf_miller = self.pyro_to_code_miller()

            for key, val in pyro_tglf_miller.items():
                tglf_input[val] = miller[key]

            tglf_input["S_DELTA_LOC"] = miller.s_delta * np.sqrt(1 - miller.delta**2)
            tglf_input["Q_PRIME_LOC"] = miller.shat * (miller.q / miller.rho) ** 2

        else:
            raise NotImplementedError

        # Kinetic data
        local_species = pyro.local_species
        tglf_input["NS"] = local_species.nspec

        # Electrons need to be first in TGLF
        species_names = local_species.names
        species_names.remove("electron")
        species_names.insert(0, "electron")

        for i_sp, name in enumerate(species_names):
            pyro_tglf_species = self.pyro_to_code_species(i_sp + 1)

            for pyro_key, tglf_key in pyro_tglf_species.items():
                tglf_input[tglf_key] = local_species[name][pyro_key]

        tglf_input["XNUE"] = local_species.electron.nu

        beta = 0.0

        # If species are defined calculate beta and beta_prime_scale
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge

            pe = pref * local_species.electron.dens * local_species.electron.temp

            beta = pe / b_ref**2 * 8 * pi * 1e-7

        # Calculate beta from existing value from input
        else:
            if pyro.local_geometry_type == "Miller":
                if miller.B0 is not None:
                    beta = 1.0 / (miller.B0 * miller.bunit_over_b0) ** 2

                else:
                    beta = 0.0

        tglf_input["BETAE"] = beta

        tglf_input["P_PRIME_LOC"] = (
            miller.beta_prime
            * miller.q
            / miller.rho
            / miller.bunit_over_b0**2
            / (8 * np.pi)
        )

        # Numerics
        numerics = pyro.numerics

        tglf_input["USE_BPER"] = numerics.apar
        tglf_input["USE_BPAR"] = numerics.bpar

        # Set time stepping
        tglf_input["USE_TRANSPORT_MODEL"] = numerics.nonlinear

        tglf_input["KY"] = numerics.ky
        tglf_input["NKY"] = numerics.nky

        tglf_input["NXGRID"] = min(numerics.ntheta, tglf_max_ntheta)
        tglf_input["KX0_LOC"] = numerics.theta0 / (2 * pi)

        if not numerics.nonlinear:
            tglf_input["WRITE_WAVEFUNCTION_FLAG"] = 1

        self.to_file(
            tglf_input, file_name, directory=directory, float_format=pyro.float_format
        )

    def tglf_parser(self, data_file):
        """
        Parse tglf input file to a dictonary
        """
        import re

        f = open(data_file)

        keys = []
        values = []

        for line in f:
            raw_data = line.strip()
            if raw_data != "":
                # Ignore commented lines
                if raw_data[0] != "#":

                    # Splits by #,= and remves whitespace
                    input_data = [
                        data.strip() for data in re.split("=", raw_data) if data != ""
                    ]

                    keys.append(input_data[0])
                    if not input_data[1].isalpha():
                        if input_data[1] in [".true.", ".false."]:
                            values.append(input_data[1])
                        else:
                            values.append(eval(input_data[1]))
                    else:
                        values.append(input_data[1])

        tglf_dict = {}
        for key, value in zip(keys, values):
            tglf_dict[key] = value

        return tglf_dict

    def to_file(self, tglf_dict, filename, float_format="", directory="."):
        """
        Writes input file for tglf from a dictionary

        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        path_to_file = os.path.join(directory, filename)

        new_tglf_input = open(path_to_file, "w+")

        for key, value in tglf_dict.items():
            if isinstance(value, float):
                line = f"{key} = {value:{float_format}}\n"
            else:
                line = f"{key} = {value}\n"
            new_tglf_input.write(line)

        new_tglf_input.close()

    def load_local_geometry(self, pyro, tglf):
        """
        Loads LocalGeometry class from pyro.tglf_input
        """

        if pyro.local_geometry_type == "Miller":
            self.load_miller(pyro, tglf)

    def load_miller(self, pyro, tglf):
        """
        Loads Miller class from pyro.tglf_input
        """

        pyro_tglf_miller = self.pyro_to_code_miller()

        miller = pyro.local_geometry

        for key, val in pyro_tglf_miller.items():
            miller[key] = tglf[val]

        miller.s_delta = tglf["S_DELTA_LOC"] / np.sqrt(1 - miller.delta**2)
        miller.shat = tglf["Q_PRIME_LOC"] / (miller.rho / miller.q) ** 2

        beta = tglf["BETAE"]
        miller.bunit_over_b0 = miller.get_bunit_over_b0()

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0:
            miller.B0 = 1 / (beta**0.5) / miller.bunit_over_b0
        else:
            miller.B0 = None

        miller.beta_prime = (
            tglf["P_PRIME_LOC"]
            * miller.rho
            / miller.q
            * miller.bunit_over_b0**2
            * (8 * np.pi)
        )

    def load_local_species(self, pyro, tglf):
        """
        Load LocalSpecies object from pyro.gene_input
        """

        nspec = tglf["NS"]

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nspec = nspec
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(nspec):

            pyro_tglf_species = self.pyro_to_code_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_tglf_species.items():
                species_data[p_key] = tglf[c_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
                species_data.nu = tglf["XNUE"]
                te = species_data.temp
                ne = species_data.dens
                me = species_data.mass
            else:
                ion_count += 1
                name = f"ion{ion_count}"

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species.add_species(name=name, species_data=species_data)

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        for name in local_species.names:
            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

        # Get collision frequency of ion species
        nu_ee = tglf["XNUE"]

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

        # Add local_species
        pyro.local_species = local_species

    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and tglf parameter names
        for miller parameters
        """

        pyro_tglf_param = {
            "rho": "RMIN_LOC",
            "Rmaj": "RMAJ_LOC",
            "q": "Q_LOC",
            "kappa": "KAPPA_LOC",
            "s_kappa": "S_KAPPA_LOC",
            "delta": "DELTA_LOC",
            "shift": "DRMAJDX_LOC",
        }

        return pyro_tglf_param

    def pyro_to_code_species(self, iSp=1):
        """
        Generates dictionary of equivalent pyro and tglf parameter names
        for miller parameters
        """

        pyro_tglf_species = {
            "mass": f"MASS_{iSp}",
            "z": f"ZS_{iSp}",
            "dens": f"AS_{iSp}",
            "temp": f"TAUS_{iSp}",
            "a_lt": f"RLTS_{iSp}",
            "a_ln": f"RLNS_{iSp}",
        }

        return pyro_tglf_species

    def add_flags(self, pyro, flags):
        """
        Add extra flags to tglf input file

        """

        for key, value in flags.items():
            pyro.tglf_input[key] = value

    def load_numerics(self, pyro, tglf):
        """
        Loads up Numerics object into pyro
        """

        numerics = Numerics()

        numerics.phi = True
        if tglf["USE_BPER"] in ["T", True]:
            numerics.apar = True
        else:
            numerics.apar = False

        if tglf["USE_BPAR"] in ["T", True]:
            numerics.bpar = True
        else:
            numerics.bpar = False

        numerics.ky = tglf["KY"]

        try:
            numerics.nky = tglf["NKY"]
        except KeyError:
            numerics.nky = 1

        try:
            numerics.theta0 = tglf["KX0_LOC"] * 2 * pi
        except KeyError:
            numerics.theta0 = 0.0

        try:
            numerics.ntheta = tglf["NXGRID"]
        except KeyError:
            numerics.ntheta = 16

        try:
            nl_mode = tglf["USE_TRANSPORT_MODEL"]
        except KeyError:
            nl_mode = 1

        if nl_mode == "T":
            numerics.nonlinear = True
        else:
            numerics.nonlinear = False
            numerics.nky = 1

        pyro.numerics = numerics

    def load_gk_output(self, pyro):
        """
        Loads GK Outputs
        """

        pyro.gk_output = GKOutput()

        if pyro.numerics.nonlinear:
            self.load_nonlinear_grids(pyro)
            self.load_fields(pyro)

            self.load_fluxes(pyro)

            self.load_eigenvalues(pyro)

        else:
            self.load_linear(pyro)

    def load_nonlinear_grids(self, pyro):

        """
        Loads tglf grids to GKOutput.data as Dataset

        """

        import xarray as xr

        gk_output = pyro.gk_output

        run_directory = pyro.run_directory

        ky_file = os.path.join(run_directory, "out.tglf.ky_spectrum")

        ky = np.loadtxt(ky_file, skiprows=2)

        nky = len(ky)

        field_spectrum = open(run_directory / "out.tglf.field_spectrum")

        field = ["phi"]
        for i in range(4):
            field_spectrum.readline()

        if field_spectrum.readline().strip() == "a_par_yes":
            field.append("apar")

        if field_spectrum.readline() == "b_par_yes":
            field.append("bpar")

        field_spectrum.close()

        nfield = len(field)

        ql_spectrum = open(run_directory / "out.tglf.QL_flux_spectrum")

        nmode = int(ql_spectrum.readlines()[3].split(" ")[-1])
        mode = list(range(1, 1 + nmode))

        moment = ["particle", "energy", "tor_momentum", "par_momentum", "exchange"]
        species = pyro.local_species.names
        nspecies = len(species)

        # Grid sizes
        gk_output.nky = nky
        gk_output.nspecies = nspecies
        gk_output.nfield = nfield
        gk_output.nmode = nmode
        gk_output.nmoment = 5

        # Grid values
        gk_output.ky = ky

        # Store grid data as xarray DataSet
        ds = xr.Dataset(
            coords={
                "field": field,
                "moment": moment,
                "species": species,
                "ky": ky,
                "nmode": mode,
            }
        )

        gk_output.data = ds

    def load_fields(self, pyro):
        """
        Loads fields into GKOutput.data DataSet
        pyro.gk_output.data['fields'] = fields(ky, nmode, field)
        """

        gk_output = pyro.gk_output
        data = gk_output.data

        nky = gk_output.nky
        nmode = gk_output.nmode
        nfield = gk_output.nfield

        filename = pyro.run_directory / "out.tglf.field_spectrum"

        f = open(filename, "r")
        full_data = " ".join(f.readlines()[6:]).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]
        f.close()

        fields = np.reshape(full_data, (nky, nmode, 4))
        fields = fields[:, :, :nfield]
        data["fields"] = (("ky", "nmode", "field"), fields)

    def load_fluxes(self, pyro):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """
        gk_output = pyro.gk_output
        data = gk_output.data

        nky = gk_output.nky
        nspecies = gk_output.nspecies
        nfield = gk_output.nfield
        nmoment = gk_output.nmoment

        filename = pyro.run_directory / "out.tglf.sum_flux_spectrum"

        f = open(filename, "r")

        full_data = [x for x in f.readlines() if "species" not in x]
        full_data = " ".join(full_data).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        f.close()
        fluxes = np.reshape(full_data, (nspecies, nfield, nky, nmoment))

        data["fluxes"] = (("species", "field", "ky", "moment"), fluxes)

    def load_eigenvalues(self, pyro):
        """
        Loads eigenvalues into GKOutput.data DataSet
        pyro.gk_output.data['eigenvalues'] = eigenvalues(ky, nmode)
        pyro.gk_output.data['mode_frequency'] = mode_frequency(ky, nmode)
        pyro.gk_output.data['growth_rate'] = growth_rate(ky, nmode)

        """
        gk_output = pyro.gk_output
        data = gk_output.data

        nky = gk_output.nky
        nmode = gk_output.nmode

        filename = pyro.run_directory / "out.tglf.eigenvalue_spectrum"

        f = open(filename, "r")

        full_data = " ".join(f.readlines()).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        f.close()
        eigenvalues = np.reshape(full_data, (nky, nmode, 2))
        eigenvalues = eigenvalues[:, :, 1] + 1j * eigenvalues[:, :, 0]

        data["eigenvalues"] = (("ky", "mode"), eigenvalues)
        data["growth_rate"] = (("ky", "mode"), np.imag(eigenvalues))
        data["mode_frequency"] = (("ky", "mode"), np.real(eigenvalues))

    def load_linear(self, pyro):
        """
        Loads linear into GKOutput.data Dataset
        pyro.gk_output.data['eigenfunctions'] = eigenvalues(field, theta, time)

        """
        import xarray as xr

        gk_output = pyro.gk_output

        filename = pyro.run_directory / "out.tglf.wavefunction"

        f = open(filename, "r")
        grid = f.readline().strip().split(" ")
        grid = [x for x in grid if x]
        nmode = int(grid[0])
        nfield = int(grid[1])
        ntheta = int(grid[2])

        full_data = " ".join(f.readlines()).split(" ")
        full_data = [float(x.strip()) for x in full_data if is_float(x.strip())]

        f.close()

        full_data = np.reshape(full_data, (ntheta, (nmode * 2 * nfield) + 1))
        theta = full_data[:, 0]
        field = ["phi", "apar", "bpar"][:nfield]
        mode = list(range(1, 1 + nmode))

        # Grid sizes
        gk_output.ntheta = ntheta
        gk_output.nfield = nfield
        gk_output.nmode = nmode

        # Grid values
        gk_output.theta = theta
        gk_output.field = field
        gk_output.mode = mode

        # Store grid data as xarray DataSet
        ds = xr.Dataset(
            coords={
                "field": field,
                "theta": theta,
                "mode": mode,
            }
        )

        eigenfunctions = np.reshape(full_data[:, 1:], (ntheta, nmode, nfield, 2))

        eigenfunctions = eigenfunctions[:, :, :, 1] + 1j * eigenfunctions[:, :, :, 0]

        ds["eigenfunctions"] = (
            (
                "theta",
                "mode",
                "field",
            ),
            eigenfunctions,
        )

        filename = pyro.run_directory / "out.tglf.run"

        f = open(filename, "r")
        lines = f.readlines()[-nmode:]
        f.close()
        eigenvalues = np.array(
            [eig.strip().split(":")[-1].split("  ") for eig in lines], dtype="float"
        )

        mode_frequency = eigenvalues[:, 0]
        growth_rate = eigenvalues[:, 1]
        eigenvalues = eigenvalues[:, 0] + 1j * eigenvalues[:, 1]

        ds["eigenvalues"] = (
            ("mode",),
            eigenvalues,
        )

        ds["growth_rate"] = (
            ("mode",),
            growth_rate,
        )

        ds["mode_frequency"] = (
            ("mode",),
            mode_frequency,
        )

        gk_output.data = ds


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
