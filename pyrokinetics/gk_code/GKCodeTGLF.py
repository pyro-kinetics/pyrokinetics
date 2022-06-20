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

        self.base_template_file = template_dir / "input.TGLF"
        self.code_name = "TGLF"
        self.default_file_name = "input.TGLF"

        self.gk_output = None

    def read(self, pyro, data_file=None, template=False):
        """
        Reads a TGLF input file and loads pyro object with the data
        """
        if template and data_file is None:
            data_file = self.base_template_file

        TGLF = self.TGLF_parser(data_file)

        pyro.initial_datafile = copy.copy(TGLF)

        try:

            if TGLF["USE_TRANSPORT_MODEL"]:

                pyro.linear = False
            else:
                pyro.linear = True
        except KeyError:
            pyro.linear = False

        pyro.TGLF_input = TGLF

        if not template:
            self.load_pyro(pyro)

        # Load Pyro with numerics if they don't exist yet
        if not hasattr(pyro, "numerics"):
            self.load_numerics(pyro, TGLF)

    def verify(self, filename: PathLike):
        """read TGLF file, check the dict returned holds the expected data"""
        data = self.TGLF_parser(filename)
        expected_keys = ["RMIN_LOC", "RMAJ_LOC", "NKY"]
        if not np.all(np.isin(expected_keys, list(data.keys()))):
            raise ValueError(f"Expected TGLF file, received {filename}")

    def load_pyro(self, pyro):
        """
        Loads LocalSpecies, LocalGeometry, Numerics classes from pyro.TGLF_input

        """

        # Geometry
        TGLF = pyro.TGLF_input

        TGLF_eq = TGLF["GEOMETRY_FLAG"]

        if TGLF_eq == 1:
            pyro.local_geometry = "Miller"
        elif TGLF_eq == 2:
            pyro.local_geometry = "Fourier"
        elif TGLF_eq == 0:
            pyro.local_geometry = "SAlpha"

        # Load local geometry
        self.load_local_geometry(pyro, TGLF)

        # Load Species
        self.load_local_species(pyro, TGLF)

        # Need species to set up beta_prime
        beta_prime_scale = TGLF.get("BETA_STAR_SCALE", 1.0)

        if pyro.local_geometry_type == "Miller":
            if pyro.local_geometry.B0 is not None:
                pyro.local_geometry.beta_prime = (
                    -pyro.local_species.a_lp
                    / pyro.local_geometry.B0 ** 2
                    * beta_prime_scale
                )
            else:
                pyro.local_geometry.beta_prime = 0.0
        else:
            raise NotImplementedError

        self.load_numerics(pyro, TGLF)

    def write(self, pyro, file_name, directory="."):
        """
        Write a TGLF input file from a pyro object

        """
        tglf_max_ntheta = 32

        TGLF_input = pyro.TGLF_input

        # Geometry data
        if pyro.local_geometry_type == "Miller":
            miller = pyro.local_geometry

            # Ensure Miller settings in input file
            TGLF_input["GEOMETRY_MODEL"] = 1

            # Reference B field - Bunit = q/r dpsi/dr
            if miller.B0 is not None:
                b_ref = miller.B0 * miller.bunit_over_b0
            else:
                b_ref = None

            # Assign Miller values to input file
            pyro_TGLF_miller = self.pyro_to_code_miller()

            for key, val in pyro_TGLF_miller.items():
                TGLF_input[val] = miller[key]

            TGLF_input["S_DELTA_LOC"] = miller.s_delta * np.sqrt(1 - miller.delta ** 2)
            TGLF_input["Q_PRIME_LOC"] = miller.shat * (miller.q / miller.rho) ** 2

        else:
            raise NotImplementedError

        # Kinetic data
        local_species = pyro.local_species
        TGLF_input["NS"] = local_species.nspec

        for i_sp, name in enumerate(local_species.names):
            pyro_TGLF_species = self.pyro_to_code_species(i_sp + 1)

            for pyro_key, TGLF_key in pyro_TGLF_species.items():
                TGLF_input[TGLF_key] = local_species[name][pyro_key]

        TGLF_input["XNUE"] = local_species.electron.nu

        beta = 0.0

        # If species are defined calculate beta and beta_prime_scale
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge

            pe = pref * local_species.electron.dens * local_species.electron.temp

            beta = pe / b_ref ** 2 * 8 * pi * 1e-7

        # Calculate beta from existing value from input
        else:
            if pyro.local_geometry_type == "Miller":
                if miller.B0 is not None:
                    beta = 1.0 / (miller.B0 * miller.bunit_over_b0) ** 2

                else:
                    beta = 0.0

        TGLF_input["BETAE"] = beta

        TGLF_input["P_PRIME_LOC"] = (
            miller.beta_prime
            * miller.q
            / miller.rho
            / miller.bunit_over_b0 ** 2
            / (8 * np.pi)
        )

        # Numerics
        numerics = pyro.numerics

        TGLF_input["USE_BPER"] = numerics.apar
        TGLF_input["USE_BPAR"] = numerics.bpar

        # Set time stepping
        TGLF_input["USE_TRANSPORT_MODEL"] = numerics.nonlinear

        TGLF_input["KY"] = numerics.ky
        TGLF_input["NKY"] = numerics.nky

        TGLF_input["Nz"] = min(numerics.ntheta, tglf_max_ntheta)
        TGLF_input["KX0_LOC"] = numerics.theta0 / (2 * pi)

        if not numerics.nonlinear:
            TGLF_input["WRITE_WAVEFUNCTION_FLAG"] = 1

        self.to_file(
            TGLF_input, file_name, directory=directory, float_format=pyro.float_format
        )

    def TGLF_parser(self, data_file):
        """
        Parse TGLF input file to a dictonary
        """
        import re

        f = open(data_file)

        keys = []
        values = []

        for line in f:
            raw_data = line.strip().split("  ")[0]
            if raw_data != "":
                # Ignore commented lines
                if raw_data[0] != "#":

                    # Splits by #,= and remves whitespace
                    input_data = [
                        data.strip() for data in re.split("=", raw_data) if data != ""
                    ]

                    keys.append(input_data[0])

                    if not input_data[1].isalpha():
                        values.append(eval(input_data[1]))
                    else:
                        values.append(input_data[1])

        TGLF_dict = {}
        for key, value in zip(keys, values):
            TGLF_dict[key] = value

        return TGLF_dict

    def to_file(self, TGLF_dict, filename, float_format="", directory="."):
        """
        Writes input file for TGLF from a dictionary

        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        path_to_file = os.path.join(directory, filename)

        new_TGLF_input = open(path_to_file, "w+")

        for key, value in TGLF_dict.items():
            if isinstance(value, float):
                line = f"{key} = {value:{float_format}}\n"
            else:
                line = f"{key} = {value}\n"
            new_TGLF_input.write(line)

        new_TGLF_input.close()

    def load_local_geometry(self, pyro, TGLF):
        """
        Loads LocalGeometry class from pyro.TGLF_input
        """

        if pyro.local_geometry_type == "Miller":
            self.load_miller(pyro, TGLF)

    def load_miller(self, pyro, TGLF):
        """
        Loads Miller class from pyro.TGLF_input
        """

        # Set some defaults here
        TGLF["EQUILIBRIUM_MODEL"] = 2

        pyro_TGLF_miller = self.pyro_to_code_miller()

        miller = pyro.local_geometry

        for key, val in pyro_TGLF_miller.items():
            miller[key] = TGLF[val]

        miller.s_delta = TGLF["S_DELTA_LOC"] / np.sqrt(1 - miller.delta ** 2)
        miller.shat = TGLF["Q_PRIME_LOC"] / (miller.rho / miller.q) ** 2

        beta = TGLF["BETAE"]
        miller.bunit_over_b0 = miller.get_bunit_over_b0()

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0:
            miller.B0 = 1 / (beta ** 0.5) / miller.bunit_over_b0
        else:
            miller.B0 = None

        miller.beta_prime = (
            TGLF["P_PRIME_LOC"]
            * miller.rho
            / miller.q
            * miller.bunit_over_b0 ** 2
            * (8 * np.pi)
        )

    def load_local_species(self, pyro, TGLF):
        """
        Load LocalSpecies object from pyro.gene_input
        """

        nspec = TGLF["NS"]

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nspec = nspec
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(nspec):

            pyro_TGLF_species = self.pyro_to_code_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_TGLF_species.items():
                species_data[p_key] = TGLF[c_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = "electron"
                species_data.nu = TGLF["XNUE"]
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
        nu_ee = TGLF["NU_EE"]

        for ion in range(ion_count):
            key = f"ion{ion + 1}"

            nion = local_species[key]["dens"]
            tion = local_species[key]["temp"]
            mion = local_species[key]["mass"]
            # Not exact at log(Lambda) does change but pretty close...
            local_species[key]["nu"] = (
                nu_ee
                * (nion / tion ** 1.5 / mion ** 0.5)
                / (ne / te ** 1.5 / me ** 0.5)
            )

        # Add local_species
        pyro.local_species = local_species

    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and TGLF parameter names
        for miller parameters
        """

        pyro_TGLF_param = {
            "rho": "RMIN_LOC",
            "Rmaj": "RMAJ_LOC",
            "q": "Q_LOC",
            "kappa": "KAPPA_LOC",
            "s_kappa": "S_KAPPA_LOC",
            "delta": "DELTA_LOC",
            "shift": "DRMAJDX_LOC",
        }

        return pyro_TGLF_param

    def pyro_to_code_species(self, iSp=1):
        """
        Generates dictionary of equivalent pyro and TGLF parameter names
        for miller parameters
        """

        pyro_TGLF_species = {
            "mass": f"MASS_{iSp}",
            "z": f"ZS_{iSp}",
            "dens": f"AS_{iSp}",
            "temp": f"TAUS_{iSp}",
            "a_lt": f"RLTS_{iSp}",
            "a_ln": f"RLNS_{iSp}",
        }

        return pyro_TGLF_species

    def add_flags(self, pyro, flags):
        """
        Add extra flags to TGLF input file

        """

        for key, value in flags.items():
            pyro.TGLF_input[key] = value

    def load_numerics(self, pyro, TGLF):
        """
        Loads up Numerics object into pyro
        """

        numerics = Numerics()

        numerics.phi = True
        numerics.apar = TGLF["BPER"]
        numerics.bpar = TGLF["BPAR"]

        numerics.ky = TGLF["KY"]

        try:
            numerics.nky = TGLF["NKY"]
        except KeyError:
            numerics.nky = 1

        try:
            numerics.theta0 = TGLF["KX0_LOC"] * 2 * pi
        except KeyError:
            numerics.theta0 = 0.0

        try:
            numerics.ntheta = TGLF["NX"]
        except KeyError:
            numerics.ntheta = 16

        try:
            nl_mode = TGLF["USE_TRANSPORT_MODEL"]
        except KeyError:
            nl_mode = 1

        if nl_mode == 1:
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
        Loads TGLF grids to GKOutput.data as Dataset

        """

        import xarray as xr

        gk_output = pyro.gk_output

        run_directory = pyro.run_directory

        ky_file = os.path.join(run_directory, "out.TGLF.ky_spectrum")

        ky = np.loadtxt(ky_file, skiprows=2)

        nky = len(ky)

        field_spectrum = open(run_directory / "out.tglf.field_spectrum")

        field = ["phi"]
        field_spectrum.readlines(3)

        if field_spectrum.readline() == "a_par_yes":
            field.append("apar")

        if field_spectrum.readline() == "b_par_yes":
            field.append("bpar")

        field_spectrum.close()

        nfield = len(field)

        ql_spectrum = open(run_directory / "out.tglf.QL_flux_spectrum")

        ql_spectrum.readlines(3)
        nmodes = ql_spectrum.readline().split(" ")[-1]

        # No theta based output for NL run
        if not pyro.numerics.nonlinear:

            theta = [0.0]
            ntheta = 1

        else:
            theta = 1

        moment = ["particle", "energy", "momentum"]
        species = pyro.local_species.names
        nspecies = len(species)

        # Grid sizes
        gk_output.nky = nky
        gk_output.nspecies = nspecies
        gk_output.nfield = nfield
        gk_output.ntheta = ntheta

        # Grid values
        gk_output.ky = ky
        gk_output.theta = theta

        # Store grid data as xarray DataSet
        ds = xr.Dataset(
            coords={
                "field": field,
                "moment": moment,
                "species": species,
                "ky": ky,
                "nmodes": nmodes,
            }
        )

        gk_output.data = ds

    def load_fields(self, pyro):
        """
        Loads 3D fields into GKOutput.data DataSet
        pyro.gk_output.data['fields'] = fields(field, theta, kx, ky, time)
        """

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory

        base_file = os.path.join(run_directory, "bin.TGLF.kxky_")

        fields = np.empty(
            (
                gk_output.nfield,
                gk_output.ntheta,
                gk_output.nkx,
                gk_output.nky,
                gk_output.ntime,
            ),
            dtype=np.complex,
        )

        field_appendices = ["phi", "apar", "bpar"]

        # Linear and theta_plot != theta_grid load field structure from eigenfunction file
        if (
            not pyro.numerics.nonlinear
            and gk_output.ntheta_plot != gk_output.ntheta_grid
        ):
            self.load_eigenfunctions(pyro, no_fields=True)

            for ifield in range(gk_output.nfield):
                fields[ifield, :, 0, 0, :] = data["eigenfunctions"].isel(field=ifield)

        # Loop through all fields and add field in if it exists
        for ifield, field_appendix in enumerate(field_appendices):

            field_file = f"{base_file}{field_appendix}"

            if os.path.exists(field_file):
                raw_field = self.read_binary_file(field_file)
                sliced_field = raw_field[
                    : 2
                    * gk_output.nkx
                    * gk_output.ntheta
                    * gk_output.nky
                    * gk_output.ntime
                ]

                # Load in non-linear field
                if pyro.numerics.nonlinear:
                    field_data = (
                        np.reshape(
                            sliced_field,
                            (
                                2,
                                gk_output.nkx,
                                gk_output.ntheta,
                                gk_output.nky,
                                gk_output.ntime,
                            ),
                            "F",
                        )
                        / gk_output.rho_star
                    )

                    # Using -1j here to match pyrokinetics frequency convention (-ve is electron direction)
                    complex_field = (
                        field_data[0, :, :, :, :] - 1j * field_data[1, :, :, :, :]
                    )

                    fields[ifield, :, :, :, :] = np.swapaxes(
                        np.reshape(
                            complex_field,
                            (
                                gk_output.nkx,
                                gk_output.ntheta,
                                gk_output.nky,
                                gk_output.ntime,
                            ),
                        ),
                        0,
                        1,
                    )

                # Linear convert from kx to ballooning space
                else:
                    nradial = pyro.TGLF_input["N_RADIAL"]

                    # If theta_plot != theta_grid get amplitude of fields from binary files
                    if gk_output.ntheta_plot != gk_output.ntheta_grid:
                        field_amplitude = (
                            np.reshape(
                                sliced_field,
                                (
                                    2,
                                    nradial,
                                    gk_output.ntheta_plot,
                                    gk_output.nky,
                                    gk_output.ntime,
                                ),
                                "F",
                            )
                            / gk_output.rho_star
                        )

                        middle_kx = int(nradial / 2) + 1
                        field_amplitude = field_amplitude[0, middle_kx, 0, 0, :]

                        fields[ifield, :, 0, 0, :] *= field_amplitude

                    # If all theta point are there then read in data
                    else:
                        field_data = (
                            np.reshape(
                                sliced_field,
                                (
                                    2,
                                    nradial,
                                    gk_output.ntheta_plot,
                                    gk_output.nky,
                                    gk_output.ntime,
                                ),
                                "F",
                            )
                            / gk_output.rho_star
                        )

                        # Using -1j here to match pyrokinetics frequency convention (-ve is electron direction)
                        complex_field = (
                            field_data[0, :, :, :, :] - 1j * field_data[1, :, :, :, :]
                        )

                        # Poisson Sum (no negative in exponent to match frequency convention)
                        for i_radial in range(nradial):
                            nx = -nradial // 2 + (i_radial - 1)
                            complex_field[i_radial, :, :, :] *= np.exp(
                                2 * pi * 1j * nx * pyro.local_geometry.q
                            )

                        fields[ifield, :, :, :, :] = np.reshape(
                            complex_field,
                            (
                                gk_output.ntheta,
                                gk_output.nkx,
                                gk_output.nky,
                                gk_output.ntime,
                            ),
                        )

            else:
                if ifield <= pyro.gk_output.nfield - 1:
                    print(f"No field file for {field_appendix}")
                    fields[ifield, :, :, :, :] = None

        data["fields"] = (("field", "theta", "kx", "ky", "time"), fields)

    def load_fluxes(self, pyro):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory

        flux_file = os.path.join(run_directory, "bin.TGLF.ky_flux")

        fluxes = np.empty(
            (gk_output.nspecies, 3, gk_output.nfield, gk_output.nky, gk_output.ntime)
        )

        if os.path.exists(flux_file):
            raw_flux = self.read_binary_file(flux_file)
            sliced_flux = raw_flux[
                : gk_output.nspecies
                * 3
                * gk_output.nfield
                * gk_output.nky
                * gk_output.ntime
            ]
            fluxes = np.reshape(
                sliced_flux,
                (
                    gk_output.nspecies,
                    3,
                    gk_output.nfield,
                    gk_output.nky,
                    gk_output.ntime,
                ),
                "F",
            )

        data["fluxes"] = (("species", "moment", "field", "ky", "time"), fluxes)

    def load_eigenvalues(self, pyro):
        """
        Loads eigenvalues into GKOutput.data DataSet
        pyro.gk_output.data['eigenvalues'] = eigenvalues(ky, time)
        pyro.gk_output.data['mode_frequency'] = mode_frequency(ky, time)
        pyro.gk_output.data['growth_rate'] = growth_rate(ky, time)

        """

        gk_output = pyro.gk_output
        data = gk_output.data

        # Use default method to calculate growth/freq if possible
        if not np.isnan(data["fields"].data).any():
            super().load_eigenvalues(pyro)

        else:
            run_directory = pyro.run_directory

            eigenvalue_file = os.path.join(run_directory, "bin.TGLF.freq")

            if os.path.exists(eigenvalue_file):
                raw_data = self.read_binary_file(eigenvalue_file)
                sliced_data = raw_data[: 2 * gk_output.nky * gk_output.ntime]
                eigenvalue_over_time = np.reshape(
                    sliced_data, (2, gk_output.nky, gk_output.ntime), "F"
                )
            else:
                eigenvalue_file = os.path.join(run_directory, "out.TGLF.freq")
                raw_data = np.loadtxt(eigenvalue_file).transpose()
                sliced_data = raw_data[:, : gk_output.ntime]
                eigenvalue_over_time = np.reshape(
                    sliced_data, (2, gk_output.nky, gk_output.ntime)
                )

            mode_frequency = eigenvalue_over_time[0, :, :]
            growth_rate = eigenvalue_over_time[1, :, :]
            eigenvalue = mode_frequency + 1j * growth_rate

            data["growth_rate"] = (("ky", "time"), growth_rate)
            data["mode_frequency"] = (("ky", "time"), mode_frequency)
            data["eigenvalues"] = (("ky", "time"), eigenvalue)

            self.get_growth_rate_tolerance(pyro)

    def load_eigenfunctions(self, pyro, no_fields=False):
        """
        Loads eigenfunctions into GKOutput.data Dataset
        pyro.gk_output.data['eigenfunctions'] = eigenvalues(field, theta, time)

        """

        gk_output = pyro.gk_output
        data = gk_output.data

        if no_fields:
            no_nan = False
        else:
            no_nan = not np.isnan(data["fields"].data).any()

        if gk_output.ntheta_plot == gk_output.ntheta_grid:
            all_ballooning = True
        else:
            all_ballooning = False

        # Use default method to calculate growth/freq if possible
        if no_nan and all_ballooning:
            super().load_eigenfunctions(pyro)

        # Read TGLF output file
        else:
            run_directory = pyro.run_directory

            base_file = os.path.join(run_directory, "bin.TGLF.")

            eigenfunctions = np.empty(
                (gk_output.nfield, gk_output.ntheta, gk_output.ntime), dtype=np.complex
            )

            field_appendices = ["phi", "apar", "bpar"]

            # Loop through all fields and add eigenfunction if it exists
            for ifield, field_appendix in enumerate(field_appendices):

                eigenfunction_file = f"{base_file}{field_appendix}b"

                if os.path.exists(eigenfunction_file):
                    raw_eigenfunction = self.read_binary_file(eigenfunction_file)[
                        : 2 * gk_output.ntheta * gk_output.ntime
                    ]

                    sliced_eigenfunction = raw_eigenfunction[
                        : 2 * gk_output.ntheta * gk_output.ntime
                    ]
                    eigenfunction_data = np.reshape(
                        sliced_eigenfunction,
                        (2, gk_output.ntheta, gk_output.ntime),
                        "F",
                    )

                    eigenfunctions[ifield, :, :] = (
                        eigenfunction_data[0, :, :] + 1j * eigenfunction_data[1, :, :]
                    )

            data["eigenfunctions"] = (("field", "theta", "time"), eigenfunctions)

    def read_binary_file(self, file_name):
        """
        Read TGLF binary files
        """

        raw_data = np.fromfile(file_name, dtype="float32")

        return raw_data
