import copy

import numpy as np

from .constants import *
from .local_species import LocalSpecies
from .numerics import Numerics
from .gk_code import GKCode
from .gk_output import GKOutput
import os
from path import Path
from cleverdict import CleverDict


class CGYRO(GKCode):
    """
    Basic CGYRO object inheriting method from GKCode

    """

    def __init__(self):

        self.base_template_file = os.path.join(Path(__file__).dirname(), 'templates', 'input.cgyro')
        self.default_file_name = 'input.cgyro'

        self.gk_output = None

    def read(self, pyro, data_file=None, template=False):
        """
        Reads a CGYRO input file and loads pyro object with the data
        """
        if template and data_file is None:
            data_file = self.base_template_file

        cgyro = self.cgyro_parser(data_file)

        pyro.initial_datafile = copy.copy(cgyro)

        try:
            nl_flag = cgyro['NONLINEAR_FLAG']

            if nl_flag == 0:

                pyro.linear = True
            else:
                pyro.linear = False
        except KeyError:
            pyro.linear = True

        pyro.cgyro_input = cgyro

        if not template:
            self.load_pyro(pyro)

        # Load Pyro with numerics if they don't exist yet
        if not hasattr(pyro, 'numerics'):
            self.load_numerics(pyro, cgyro)

    def load_pyro(self, pyro):
        """
        Loads LocalSpecies, LocalGeometry, Numerics classes from pyro.cgyro_input

        """

        # Geometry
        cgyro = pyro.cgyro_input

        cgyro_eq = cgyro['EQUILIBRIUM_MODEL']

        if cgyro_eq == 2:
            pyro.local_geometry = 'Miller'
        elif cgyro_eq == 3:
            pyro.local_geometry = 'Fourier'
        elif cgyro_eq == 1:
            pyro.local_geometry = 'SAlpha'

        # Load local geometry
        self.load_local_geometry(pyro, cgyro)

        # Load Species
        self.load_local_species(pyro, cgyro)

        # Need species to set up beta_prime
        beta_prime_scale = cgyro['BETA_STAR_SCALE']

        if pyro.local_geometry_type == 'Miller':
            if pyro.local_geometry.Bunit is not None:
                pyro.local_geometry.beta_prime = - pyro.local_species.a_lp / pyro.local_geometry.B0 ** 2 * \
                                                 beta_prime_scale
            else:
                pyro.local_geometry.beta_prime = 0.0
        else:
            raise NotImplementedError

        self.load_numerics(pyro, cgyro)

    def write(self, pyro, file_name, directory='.'):
        """
        Write a CGYRO input file from a pyro object

        """

        cgyro_input = pyro.cgyro_input

        # Geometry data
        if pyro.local_geometry_type == 'Miller':
            miller = pyro.local_geometry

            # Ensure Miller settings in input file
            cgyro_input['EQUILIBRIUM_MODEL'] = 2

            # Reference B field - Bunit = q/r dpsi/dr
            b_ref = miller.Bunit

            shat = miller.shat

            # Assign Miller values to input file
            pyro_cgyro_miller = self.pyro_to_code_miller()

            for key, val in pyro_cgyro_miller.items():
                cgyro_input[val] = miller[key]

            cgyro_input['S_DELTA'] = miller.s_delta * np.sqrt(1 - miller.delta ** 2)

        else:
            raise NotImplementedError

        # Kinetic data
        local_species = pyro.local_species
        cgyro_input['N_SPECIES'] = local_species.nspec

        for i_sp, name in enumerate(local_species.names):
            pyro_cgyro_species = self.pyro_to_code_species(i_sp + 1)

            for pyro_key, cgyro_key in pyro_cgyro_species.items():
                cgyro_input[cgyro_key] = local_species[name][pyro_key]

        cgyro_input['NU_EE'] = local_species.electron.nu

        beta = 0.0
        beta_prime_scale = 1.0

        # If species are defined calculate beta and beta_prime_scale
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge

            pe = pref * local_species.electron.dens * local_species.electron.temp

            beta = pe / b_ref ** 2 * 8 * pi * 1e-7

            # Find BETA_STAR_SCALE from beta and p_prime
            if pyro.local_geometry_type == 'Miller':
                beta_prime_scale = - miller.beta_prime / (local_species.a_lp * beta * (miller.Bunit / miller.B0) ** 2)

        # Calculate beta from existing value from input
        else:
            if pyro.local_geometry_type == 'Miller':
                if miller.Bunit is not None:
                    beta = 1.0 / miller.Bunit ** 2
                    beta_prime_scale = - miller.beta_prime / (
                            local_species.a_lp * beta * (miller.Bunit / miller.B0) ** 2)
                else:
                    beta = 0.0
                    beta_prime_scale = 1.0

        cgyro_input['BETAE_UNIT'] = beta

        cgyro_input['BETA_STAR_SCALE'] = beta_prime_scale

        # Numerics
        numerics = pyro.numerics

        if numerics.bpar and not numerics.apar:
            raise ValueError("Can't have bpar without apar in CGYRO")

        cgyro_input['N_FIELD'] = 1 + int(numerics.bpar) + int(numerics.apar)

        # Set time stepping
        cgyro_input["DELTA_T"] = numerics.delta_time
        cgyro_input["MAX_TIME"] = numerics.max_time

        if numerics.nonlinear:
            cgyro_input['NONLINEAR_FLAG'] = 1
            cgyro_input['N_RADIAL'] = numerics.nkx
            cgyro_input['BOX_SIZE'] = int((numerics.ky * 2 * pi * shat / numerics.kx) + 0.1)
        else:
            cgyro_input['NONLINEAR_FLAG'] = 0
            cgyro_input['N_RADIAL'] = numerics.nperiod * 2
            cgyro_input['BOX_SIZE'] = 1

        cgyro_input['KY'] = numerics.ky
        cgyro_input['N_TOROIDAL'] = numerics.nky

        cgyro_input['N_THETA'] = numerics.ntheta
        cgyro_input['THETA_PLOT'] = numerics.ntheta

        cgyro_input['N_ENERGY'] = numerics.nenergy
        cgyro_input['N_XI'] = numerics.npitch

        cgyro_input['FIELD_PRINT_FLAG'] = 1
        cgyro_input['MOMENT_PRINT_FLAG'] = 1

        self.to_file(cgyro_input, file_name, directory=directory, float_format=pyro.float_format)

    def cgyro_parser(self, data_file):
        """
        Parse CGYRO input file to a dictonary
        """
        import re

        f = open(data_file)

        keys = []
        values = []

        for line in f:
            raw_data = line.strip().split("  ")[0]
            if raw_data != '':
                # Ignore commented lines
                if raw_data[0] != '#':

                    # Splits by #,= and remves whitespace
                    input_data = [data.strip() for data in re.split(
                        '=', raw_data) if data != '']

                    keys.append(input_data[0])

                    if not input_data[1].isalpha():
                        values.append(eval(input_data[1]))
                    else:
                        values.append(input_data[1])

        cgyro_dict = {}
        for key, value in zip(keys, values):
            cgyro_dict[key] = value

        return cgyro_dict

    def to_file(self, cgyro_dict, filename, float_format='', directory='.'):
        """
        Writes input file for CGYRO from a dictionary

        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        path_to_file = os.path.join(directory, filename)

        new_cgyro_input = open(path_to_file, 'w+')

        for key, value in cgyro_dict.items():
            if isinstance(value, float):
                line = f'{key} = {value:{float_format}}\n'
            else:
                line = f'{key} = {value}\n'
            new_cgyro_input.write(line)

        new_cgyro_input.close()

    def load_local_geometry(self, pyro, cgyro):
        """
        Loads LocalGeometry class from pyro.cgyro_input
        """

        if pyro.local_geometry_type == 'Miller':
            self.load_miller(pyro, cgyro)

    def load_miller(self, pyro, cgyro):
        """
        Loads Miller class from pyro.cgyro_input
        """

        # Set some defaults here
        cgyro['EQUILIBRIUM_MODEL'] = 2

        pyro_cgyro_miller = self.pyro_to_code_miller()

        miller = pyro.local_geometry

        for key, val in pyro_cgyro_miller.items():
            miller[key] = cgyro[val]

        miller.kappri = miller.s_kappa * miller.kappa / miller.rho
        miller.tri = np.arcsin(miller.delta)
        miller.s_delta = cgyro['S_DELTA'] / np.sqrt(1 - miller.delta ** 2)

        beta = cgyro['BETAE_UNIT']

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0:
            miller.Bunit = 1 / (beta ** 0.5)
            bunit_over_b0 = miller.get_bunit_over_b0()
            miller.B0 = miller.Bunit / bunit_over_b0
        else:
            miller.Bunit = None
            miller.B0 = None

    def load_local_species(self, pyro, cgyro):
        """
        Load LocalSpecies object from pyro.gene_input
        """

        nspec = cgyro['N_SPECIES']

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nspec = nspec
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        # Load each species into a dictionary
        for i_sp in range(nspec):

            pyro_cgyro_species = self.pyro_to_code_species(i_sp + 1)
            species_data = CleverDict()
            for p_key, c_key in pyro_cgyro_species.items():
                species_data[p_key] = cgyro[c_key]

            species_data.vel = 0.0
            species_data.a_lv = 0.0

            if species_data.z == -1:
                name = 'electron'
                species_data.nu = cgyro['NU_EE']
                te = species_data.temp
                ne = species_data.dens
                me = species_data.mass
            else:
                ion_count += 1
                name = f'ion{ion_count}'

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species[name] = species_data
            local_species.names.append(name)

        pressure = 0.0
        a_lp = 0.0

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        for name in local_species.names:
            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

            pressure += species_data.temp * species_data.dens
            a_lp += species_data.temp * species_data.dens * (species_data.a_lt + species_data.a_ln)

        # CGYRO beta_prime scale
        local_species.pressure = pressure
        local_species.a_lp = a_lp

        # Get collision frequency of ion species
        nu_ee = cgyro['NU_EE']

        for ion in range(ion_count):
            key = f'ion{ion + 1}'

            nion = local_species[key]['dens']
            tion = local_species[key]['temp']
            mion = local_species[key]['mass']

            # Not exact at log(Lambda) does change but pretty close...
            local_species[key]['nu'] = nu_ee * (nion / tion ** 1.5 / mion ** 0.5) / (ne / te ** 1.5 / me ** 0.5)

        # Add local_species
        pyro.local_species = local_species

    def pyro_to_code_miller(self):
        """
        Generates dictionary of equivalent pyro and cgyro parameter names
        for miller parameters
        """

        pyro_cgyro_param = {
            'rho': 'RMIN',
            'Rmaj': 'RMAJ',
            'Rgeo': 'RMAJ',
            'q': 'Q',
            'kappa': 'KAPPA',
            's_kappa': 'S_KAPPA',
            'delta': 'DELTA',
            'shat': 'S',
            'shift': 'SHIFT',
        }

        return pyro_cgyro_param

    def pyro_to_code_species(self, iSp=1):
        """
        Generates dictionary of equivalent pyro and cgyro parameter names
        for miller parameters
        """

        pyro_cgyro_species = {
            'mass': f'MASS_{iSp}',
            'z': f'Z_{iSp}',
            'dens': f'DENS_{iSp}',
            'temp': f'TEMP_{iSp}',
            'a_lt': f'DLNTDR_{iSp}',
            'a_ln': f'DLNNDR_{iSp}',
        }

        return pyro_cgyro_species

    def add_flags(self, pyro, flags):
        """
        Add extra flags to CGYRO input file

        """

        for key, value in flags.items():
            pyro.cgyro_input[key] = value

    def load_numerics(self, pyro, cgyro):
        """
        Loads up Numerics object into pyro
        """

        numerics = Numerics()

        nfields = cgyro['N_FIELD']

        numerics.phi = nfields >= 1
        numerics.apar = nfields >= 2
        numerics.bpar = nfields >= 3

        numerics.delta_time = cgyro.get("DELTA_T", 0.01)
        numerics.max_time = cgyro.get("MAX_TIME", 1.0)

        numerics.ky = cgyro['KY']

        try:
            numerics.nky = cgyro['N_TOROIDAL']
        except KeyError:
            numerics.nky = 1

        try:
            numerics.theta0 = cgyro['PX0'] * 2 * pi
        except KeyError:
            numerics.theta0 = 0.0

        try:
            numerics.nkx = cgyro['N_RADIAL']
        except KeyError:
            numerics.nkx = 1

        numerics.nperiod = int(cgyro['N_RADIAL'] / 2)

        shat = pyro.local_geometry.shat

        try:
            box_size = cgyro['BOX_SIZE']
        except KeyError:
            box_size = 1

        numerics.kx = numerics.ky * 2 * pi * shat / box_size

        try:
            numerics.ntheta = cgyro['N_THETA']
        except KeyError:
            numerics.ntheta = 24

        try:
            numerics.nenergy = cgyro['N_ENERGY']
        except KeyError:
            numerics.nenergy = 8

        try:
            numerics.npitch = cgyro['N_XI']
        except KeyError:
            numerics.npitch = 16

        try:
            nl_mode = cgyro['NONLINEAR_FLAG']
        except KeyError:
            nl_mode = 0

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
        Loads CGYRO grids to GKOutput.data as Dataset

        """

        import xarray as xr

        gk_output = pyro.gk_output

        run_directory = pyro.run_directory
        time_file = os.path.join(run_directory, 'out.cgyro.time')

        time = np.loadtxt(time_file)[:, 0]

        gk_output.time = time

        gk_output.ntime = len(time)

        eq_file = os.path.join(run_directory, 'out.cgyro.equilibrium')

        eq_data = np.loadtxt(eq_file)

        rho_star = eq_data[23]

        grids_file = os.path.join(run_directory, 'out.cgyro.grids')

        grid_data = np.loadtxt(grids_file)

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

        ntheta_ballooning = ntheta_grid * int(nkx / box_size)

        starting_point = 11 + nkx

        theta_grid = grid_data[starting_point:starting_point + ntheta_grid]
        starting_point += ntheta_grid

        energy = grid_data[starting_point:starting_point + nenergy]
        starting_point += nenergy

        pitch = grid_data[starting_point:starting_point + npitch]
        starting_point += npitch

        theta_ballooning = grid_data[starting_point:starting_point + ntheta_ballooning]
        starting_point += ntheta_ballooning

        ky = grid_data[starting_point:starting_point + nky]

        # Convert to ballooning co-ordinate so only 1 kx
        if not pyro.numerics.nonlinear:

            theta = theta_ballooning
            ntheta = ntheta_ballooning

            kx = [0.0]
            nkx = 1

        else:
            # Output data actually given on theta_plot grid
            stride = ntheta_grid // ntheta_plot

            ntheta = ntheta_plot
            theta = np.empty(ntheta_plot)

            # Calculate sub-sampled theta grid theta grid
            for i in range(ntheta):
                theta[i] = theta_grid[stride * i]

            kx = 2 * pi * np.linspace(-int(nkx / 2), int(nkx / 2) - 1, nkx) / length_x

        field = ['phi', 'apar', 'bpar']
        field = field[:nfield]
        moment = ['particle', 'energy', 'momentum']
        species = pyro.local_species.names

        # Grid sizes
        gk_output.nky = nky
        gk_output.nkx = nkx
        gk_output.nenergy = nenergy
        gk_output.npitch = npitch
        gk_output.ntheta = ntheta
        gk_output.nspecies = nspecies
        gk_output.nfield = nfield
        gk_output.ntheta_plot = ntheta_plot
        gk_output.ntheta_grid = ntheta_grid

        # Grid values
        gk_output.ky = ky
        gk_output.kx = kx
        gk_output.energy = energy
        gk_output.pitch = pitch
        gk_output.theta = theta
        gk_output.rho_star = rho_star

        # Store grid data as xarray DataSet
        ds = xr.Dataset(coords={"time": time,
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

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory

        base_file = os.path.join(run_directory, 'bin.cgyro.kxky_')

        fields = np.empty((gk_output.nfield, gk_output.ntheta, gk_output.nkx, gk_output.nky, gk_output.ntime),
                          dtype=np.complex)

        field_appendices = ['phi', 'apar', 'bpar']

        # Linear and theta_plot != theta_grid load field structure from eigenfunction file
        if not pyro.numerics.nonlinear and gk_output.ntheta_plot != gk_output.ntheta_grid:
            self.load_eigenfunctions(pyro, no_fields=True)

            for ifield in range(gk_output.nfield):
                fields[ifield, :, 0, 0, :] = data['eigenfunctions'].isel(field=ifield)

        # Loop through all fields and add field in if it exists
        for ifield, field_appendix in enumerate(field_appendices):

            field_file = f"{base_file}{field_appendix}"

            if os.path.exists(field_file):
                raw_field = self.read_binary_file(field_file)
                sliced_field = raw_field[:2 * gk_output.nkx * gk_output.ntheta * gk_output.nky * gk_output.ntime]

                # Load in non-linear field
                if pyro.numerics.nonlinear:
                    field_data = np.reshape(sliced_field, (2, gk_output.nkx, gk_output.ntheta, gk_output.nky,
                                                           gk_output.ntime), 'F') / gk_output.rho_star

                    complex_field = field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]

                    fields[ifield, :, :, :, :] = np.reshape(complex_field, (gk_output.ntheta, gk_output.nkx,
                                                                        gk_output.nky, gk_output.ntime))
                # Linear convert from kx to ballooning space
                else:
                    nradial = pyro.cgyro_input['N_RADIAL']

                    # If theta_plot != theta_grid get amplitude of fields from binary files
                    if gk_output.ntheta_plot != gk_output.ntheta_grid:
                        field_amplitude = np.reshape(sliced_field, (2, nradial, gk_output.ntheta_plot, gk_output.nky,
                                                                    gk_output.ntime), 'F') / gk_output.rho_star

                        middle_kx = int(nradial/2) + 1
                        field_amplitude = field_amplitude[0, middle_kx, 0, 0, :]

                        fields[ifield, :, 0, 0, :] *= field_amplitude

                    # If all theta point are there then read in data
                    else:
                        field_data = np.reshape(sliced_field, (2, nradial, gk_output.ntheta_plot, gk_output.nky,
                                                               gk_output.ntime), 'F') / gk_output.rho_star

                        complex_field = field_data[0, :, :, :, :] + 1j * field_data[1, :, :, :, :]

                        # Poisson Sum
                        for i_radial in range(nradial):
                            nx = -nradial // 2 + (i_radial - 1)
                            complex_field[i_radial, :, :, :] *= np.exp(-2 * pi * 1j * nx * pyro.local_geometry.q)

                        fields[ifield, :, :, :, :] = np.reshape(complex_field, (gk_output.ntheta, gk_output.nkx,
                                                                        gk_output.nky, gk_output.ntime))

            else:
                if ifield <= pyro.gk_output.nfield - 1:
                    print(f'No field file for {field_appendix}')
                    fields[ifield, :, :, :, :] = None

        data['fields'] = (('field', 'theta', 'kx', 'ky', 'time'), fields)

    def load_fluxes(self, pyro):
        """
        Loads fluxes into GKOutput.data DataSet
        pyro.gk_output.data['fluxes'] = fluxes(species, moment, field, ky, time)
        """

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory

        flux_file = os.path.join(run_directory, 'bin.cgyro.ky_flux')

        fluxes = np.empty((gk_output.nspecies, 3, gk_output.nfield, gk_output.nky, gk_output.ntime))

        if os.path.exists(flux_file):
            raw_flux = self.read_binary_file(flux_file)
            sliced_flux = raw_flux[:gk_output.nspecies * 3 * gk_output.nfield * gk_output.nky * gk_output.ntime]
            fluxes = np.reshape(sliced_flux, (gk_output.nspecies, 3, gk_output.nfield, gk_output.nky, gk_output.ntime),
                                'F')

        data['fluxes'] = (("species", "moment", "field", "ky", "time"), fluxes)

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
        if not np.isnan(data['fields'].data).any():
            super(CGYRO, self).load_eigenvalues(pyro)

        else:
            run_directory = pyro.run_directory

            eigenvalue_file = os.path.join(run_directory, 'bin.cgyro.freq')

            if os.path.exists(eigenvalue_file):
                raw_data = self.read_binary_file(eigenvalue_file)
                sliced_data = raw_data[:2 * gk_output.nky * gk_output.ntime]
                eigenvalue_over_time = np.reshape(sliced_data, (2, gk_output.nky, gk_output.ntime), 'F')
            else:
                eigenvalue_file = os.path.join(run_directory, 'out.cgyro.freq')
                raw_data = np.loadtxt(eigenvalue_file).transpose()
                sliced_data = raw_data[:, :gk_output.ntime]
                eigenvalue_over_time = np.reshape(sliced_data, (2, gk_output.nky, gk_output.ntime))

            mode_frequency = eigenvalue_over_time[0, :, :]
            growth_rate = eigenvalue_over_time[1, :, :]
            eigenvalue = mode_frequency + 1j * growth_rate

            data['growth_rate'] = (("ky", "time"), growth_rate)
            data['mode_frequency'] = (("ky", "time"), mode_frequency)
            data['eigenvalues'] = (("ky", "time"), eigenvalue)

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
            no_nan = not np.isnan(data['fields'].data).any()

        if gk_output.ntheta_plot == gk_output.ntheta_grid:
            all_ballooning = True
        else:
            all_ballooning = False

        # Use default method to calculate growth/freq if possible
        if no_nan and all_ballooning:
            super(CGYRO, self).load_eigenfunctions(pyro)

        # Read CGYRO output file
        else:
            run_directory = pyro.run_directory

            base_file = os.path.join(run_directory, 'bin.cgyro.')

            eigenfunctions = np.empty((gk_output.nfield, gk_output.ntheta, gk_output.ntime),
                                      dtype=np.complex)

            field_appendices = ['phi', 'apar', 'bpar']

            # Loop through all fields and add eigenfunction if it exists
            for ifield, field_appendix in enumerate(field_appendices):

                eigenfunction_file = f"{base_file}{field_appendix}b"

                if os.path.exists(eigenfunction_file):
                    raw_eigenfunction = self.read_binary_file(eigenfunction_file)[:2 * gk_output.ntheta *
                                                                                  gk_output.ntime]

                    sliced_eigenfunction = raw_eigenfunction[:2 * gk_output.ntheta * gk_output.ntime]
                    eigenfunction_data = np.reshape(sliced_eigenfunction,
                                                    (2, gk_output.ntheta, gk_output.ntime),
                                                    'F')

                    eigenfunctions[ifield, :, :] = eigenfunction_data[0, :, :] + 1j * eigenfunction_data[1, :, :]

            data['eigenfunctions'] = (("field", "theta", "time"), eigenfunctions)

    def read_binary_file(self, file_name):
        """
        Read CGYRO binary files
        """

        raw_data = np.fromfile(file_name, dtype='float32')

        return raw_data
