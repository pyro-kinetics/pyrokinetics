import f90nml
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


class GENE(GKCode):
    """
    Basic GENE object

    """

    def __init__(self):

        self.base_template_file = os.path.join(Path(__file__).dirname(), 'templates', 'input.gene')
        self.default_file_name = 'input.gene'

    def read(self, pyro, data_file=None, template=False):
        """
        Reads GENE input file into a dictionary
        """

        if template and data_file is None:
            data_file = self.base_template_file

        gene = f90nml.read(data_file).todict()

        pyro.initial_datafile = copy.copy(gene)

        try:
            nl_flag = gene['general']['nonlinear']

            if nl_flag == '.F.':

                pyro.linear = True
            else:
                pyro.linear = False
        except KeyError:
            pyro.linear = True

        pyro.gene_input = gene

        # Loads pyro object with equilibrium data
        if not template:
            self.load_pyro(pyro)

        # Load Pyro with numerics if they don't exist
        if not hasattr(pyro, 'numerics'):
            self.load_numerics(pyro, gene)

    def load_pyro(self, pyro):
        """
        Loads GENE input into Pyro object
        """

        # Geometry
        gene = pyro.gene_input

        gene_eq = gene['geometry']['magn_geometry']

        if gene_eq == 's_alpha':
            pyro.local_geometry = 'SAlpha'
        elif gene_eq == 'miller':
            pyro.local_geometry = 'Miller'

        #  Load GENE with local geometry
        self.load_local_geometry(pyro, gene)

        # Load GENE with species data
        self.load_local_species(pyro, gene)

        # Need species to set up beta_prime

        if pyro.local_geometry_type == 'Miller':
            if pyro.local_geometry.B0 is not None:
                pyro.local_geometry.beta_prime = - pyro.local_species.a_lp / pyro.local_geometry.B0 ** 2
            else:
                pyro.local_geometry.beta_prime = 0.0
        else:
            raise NotImplementedError

        # Load Pyro with numerics
        self.load_numerics(pyro, gene)

    def write(self, pyro, filename, directory='.'):
        """
        For a given pyro object write a GENE input file

        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        path_to_file = os.path.join(directory, filename)

        gene_input = pyro.gene_input

        # Geometry data
        if pyro.local_geometry_type == 'Miller':
            miller = pyro.local_geometry

            # Ensure Miller settings in input file
            gene_input['geometry']['magn_geometry'] = 'miller'

            # Reference B field
            bref = miller.B0

            shat = miller.shat
            # Assign Miller values to input file
            pyro_gene_miller = self.pyro_to_gene_miller()

            # GENE uses definitions consistent with Miller. 
            for key, val in pyro_gene_miller.items():
                gene_input[val[0]][val[1]] = miller[key]

        else:
            raise NotImplementedError(f'Writing {pyro.geometry_type} for GENE not supported yet')

        # Kinetic data
        local_species = pyro.local_species
        gene_input['box']['n_spec'] = local_species.nspec

        pyro_gene_species = self.pyro_to_gene_species()

        for iSp, name in enumerate(local_species.names):

            species_key = 'species'

            if name == 'electron':
                gene_input[species_key][iSp]['name'] = 'electron'
            else:
                try:
                    gene_input[species_key][iSp]['name'] = 'ion'
                except KeyError:
                    gene_input[species_key] = copy.copy(gene_input['species_1'])
                    gene_input[species_key]['name'] = 'ion'

            for key, val in pyro_gene_species.items():
                gene_input[species_key][iSp][val] = local_species[name][key]

        # If species are defined calculate beta
        if local_species.nref is not None:

            pref = local_species.nref * local_species.tref * electron_charge

            beta = pref / bref ** 2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if pyro.local_geometry_type == 'Miller':
                if miller.B0 is not None:
                    beta = 1 / miller.B0 ** 2 * (miller.Rgeo / miller.Rmaj) ** 2
                else:
                    beta = 0.0
            else:
                raise NotImplementedError

        gene_input['general']['beta'] = beta

        # Numerics
        numerics = pyro.numerics

        if numerics['nonlinear']:

            gene_input['general']['nonlinear'] = True
            gene_input['box']['nky0'] = numerics['nky']
            gene_input['box']['nkx'] = numerics['nkx']

        else:
            gene_input['general']['nonlinear'] = False

        gene_input['box']['nky0'] = numerics.nky
        gene_input['box']['kymin'] = numerics.ky

        if numerics.kx != 0.0:
            gene_input['box']['lx'] = 2 * pi / numerics.kx

        gene_input['box']['nz0'] = numerics.ntheta
        gene_input['box']['nv0'] = 2 * numerics.nenergy
        gene_input['box']['nw0'] = numerics.npitch

        # Currently forces NL sims to have nperiod = 1
        gene_nml = f90nml.Namelist(gene_input)
        gene_nml.float_format = pyro.float_format
        gene_nml.write(path_to_file, force=True)

    def load_local_geometry(self, pyro, gene):
        """
        Loads local geometry 
        """

        if pyro.local_geometry_type == 'Miller':
            self.load_miller(pyro, gene)

    def load_miller(self, pyro, gene):
        """
        Load Miller object from GENE file
        """

        # Set some defaults here
        gene['geometry']['magn_geometry'] = 'miller'

        pyro_gene_miller = self.pyro_to_gene_miller()

        miller = pyro.local_geometry

        for key, val in pyro_gene_miller.items():
            miller[key] = gene[val[0]][val[1]]

        miller.kappri = miller.s_kappa * miller.kappa / miller.rho
        miller.tri = np.arcsin(miller.delta)

        # Get beta normalised to R_major(in case R_geo != R_major)
        beta = gene['general']['beta'] * (miller.Rmaj / miller.Rgeo) ** 2

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0 / beta)
            # Can only know Bunit/B0 from local Miller
            miller.Bunit = miller.get_bunit_over_b0() * miller.B0

        else:
            # If beta = 0
            miller.B0 = None
            miller.Bunit = None

        pyro.miller = miller

    def load_local_species(self, pyro, gene):
        """
        Load LocalSpecies object from GENE file
        """
        nspec = gene['box']['n_spec']
        pyro_gene_species = self.pyro_to_gene_species()

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species['nspec'] = nspec
        local_species['nref'] = None
        local_species['names'] = []

        ion_count = 0

        pressure = 0.0
        a_lp = 0.0
        # Load each species into a dictionary
        for i_sp in range(nspec):

            species_data = CleverDict()

            gene_key = 'species'

            gene_data = gene[gene_key][i_sp]

            gene_type = gene_data['name']

            for pyro_key, gene_key in pyro_gene_species.items():
                species_data[pyro_key] = gene_data[gene_key]

            species_data['vel'] = 0.0
            species_data['a_lv'] = 0.0
            species_data['nu'] = 0.0

            if species_data.z == -1:
                name = 'electron'
                te = species_data.temp
                ne = species_data.dens
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

        local_species.pressure = pressure
        local_species.a_lp = a_lp

        # Add local_species
        pyro.local_species = local_species

    def pyro_to_gene_miller(self):
        """
        Generates dictionary of equivalent pyro and gene parameter names
        for miller parameters
        """

        pyro_gene_param = {
            'rho': ['geometry', 'minor_r'],
            'Rmaj': ['geometry', 'major_r'],
            'Rgeo': ['geometry', 'major_r'],
            'q': ['geometry', 'q0'],
            'kappa': ['geometry', 'kappa'],
            's_kappa': ['geometry', 's_kappa'],
            'delta': ['geometry', 'delta'],
            's_delta': ['geometry', 's_delta'],
            'shat': ['geometry', 'shat'],
            'shift': ['geometry', 'drr'],
        }

        return pyro_gene_param

    def pyro_to_gene_species(self):
        """
        Generates dictionary of equivalent pyro and gene parameter names
        for species parameters
        """

        pyro_gene_species = {
            'mass': 'mass',
            'z': 'charge',
            'dens': 'dens',
            'temp': 'temp',
            'a_lt': 'omt',
            'a_ln': 'omn',
        }

        return pyro_gene_species

    def add_flags(self, pyro, flags):
        """
        Add extra flags to GENE input file

        """

        for key, parameter in flags.items():
            for param, val in parameter.items():
                pyro.gene_input[key][param] = val

    def load_numerics(self, pyro, gene):
        """
        Load GENE numerics into Pyro

        """
        # Need shear for map theta0 to kx
        # shat = pyro.miller['shat']
        # Fourier space grid
        # Linear simulation

        numerics = Numerics()

        numerics.nky = gene['box']['nky0']
        numerics.nkx = gene['box']['nx0']
        numerics.ky = gene['box']['kymin']

        try:
            numerics.kx = 2 * pi / gene['box']['lx']
        except KeyError:
            numerics.kx = 0.0

        # Velocity grid
        try:
            numerics.ntheta = gene['box']['nz0']
        except KeyError:
            numerics.ntheta = 24

        try:
            numerics.nenergy = 0.5 * gene['box']['nv0']
        except KeyError:
            numerics.nenergy = 8

        try:
            numerics.npitch = gene['box']['nw0']
        except KeyError:
            numerics.npitch = 16

        try:
            nl_mode = gene['nonlinear']
        except KeyError:
            nl_mode = 0

        if nl_mode == 1:
            numerics.nonlinear = True
        else:
            numerics.nonlinear = False

        pyro.numerics = numerics

    def load_gk_output(self, pyro, gene_output_number='0000'):
        """
        Loads GK Outputs
        """

        pyro.gk_output = GKOutput()
        pyro.gene_output_number = gene_output_number

        self.load_grids(pyro)

        self.load_fields(pyro)

        self.load_fluxes(pyro)

        if not pyro.numerics.nonlinear:
            self.load_eigenvalues(pyro)

            self.load_eigenfunctions(pyro)

    def load_grids(self, pyro):
        """
        Loads CGYRO grids to GKOutput

        out.cgyro.grids stores all the grid data in one long 1D array
        Output is in a standardised order

        """

        import xarray as xr

        gk_output = pyro.gk_output
        numerics = pyro.numerics

        nml = f90nml.read(f'parameters_{pyro.gene_output_number}')

        ntime = nml['info']['steps'][0] // nml['in_out']['istep_field'] + 1
        delta_t = nml['info']['step_time'][0]
        time = np.linspace(0, delta_t * (ntime-1), ntime)

        gk_output.time = time
        gk_output.ntime = ntime

        field = ['phi', 'apar', 'bpar']
        nfield = nml['info']['n_fields']

        field = field[:nfield]

        nky = nml['box']['nky0']

        nkx = nml['box']['nx0']

        ntheta = nml['box']['nz0']
        theta = np.linspace(-pi, pi, ntheta, endpoint=False)

        nenergy = nml['box']['nv0']
        energy = np.linspace(-1, 1, nenergy)

        npitch = nml['box']['nw0']
        pitch = np.linspace(-1, 1, npitch)

        moment = ['particle', 'energy', 'momentum']
        species = pyro.local_species.names

        if not pyro.numerics.nonlinear:

            # Set up ballooning angle
            single_theta_loop = theta
            single_ntheta_loop = ntheta

            ntheta = ntheta * (nkx - 1)
            theta = np.empty(ntheta)
            start = 0
            for i in range(nkx-1):
                pi_segment = i - nkx // 2 + 1
                theta[start:start+single_ntheta_loop] = single_theta_loop + pi_segment * 2 * pi
                start += single_ntheta_loop

            ky = [nml['box']['kymin']]

            kx = [0.0]
            nkx = 1

        # Grid sizes
        gk_output.nky = nky
        gk_output.nkx = nkx
        gk_output.nenergy = nenergy
        gk_output.npitch = npitch
        gk_output.ntheta = ntheta
        gk_output.nspecies = pyro.local_species.nspec
        gk_output.nfield = nfield

        # Grid values
        gk_output.ky = ky
        gk_output.kx = kx
        gk_output.energy = energy
        gk_output.pitch = pitch
        gk_output.theta = theta

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
        fields (field, theta, kx, ky, time)
        """
        import struct

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory

        field_file = os.path.join(run_directory, f'field_{pyro.gene_output_number}')

        fields = np.empty((gk_output.nfield, gk_output.ntheta, gk_output.nkx, gk_output.nky,
                           gk_output.ntime), dtype=np.complex)

        # Time data stored as binary (int, double, int)
        time = []
        time_data_fmt = '=idi'
        time_data_size = struct.calcsize(time_data_fmt)

        int_size = 4
        complex_size = 16

        nx = pyro.gene_input['box']['nx0']
        nconn = nx - 1
        nz = pyro.gene_input['box']['nz0']

        field_size = nx * nz * gk_output.nky * complex_size

        sliced_field = np.empty((gk_output.nfield, nx, gk_output.nky, nz,
                                gk_output.ntime), dtype=np.complex)

        fields = np.empty((gk_output.nfield, gk_output.nkx, gk_output.nky, gk_output.ntheta,
                           gk_output.ntime), dtype=np.complex)

        if os.path.exists(field_file):

            file = open(field_file, 'rb')

            for i_time in range(gk_output.ntime):
                # Read in time data (stored as int, double int)
                time_value = float(struct.unpack(time_data_fmt, file.read(time_data_size))[1])

                time.append(time_value)

                for i_field in range(gk_output.nfield):
                    dummy = struct.unpack('i', file.read(int_size))

                    binary_field = file.read(field_size)

                    raw_field = np.frombuffer(binary_field, dtype=np.complex128)

                    sliced_field[i_field, :, :, :, i_time] = np.reshape(raw_field, (nx, gk_output.nky, nz), 'F')

                    dummy = struct.unpack('i', file.read(int_size))

            if pyro.numerics.nonlinear:
                field_data = np.reshape(sliced_field, (gk_output.nfield, gk_output.nkx, gk_output.ntheta, gk_output.nky,
                                                       gk_output.ntime), 'F')

            # Convert from kx to ballooning space
            else:
                i_ball = 0

                for i_conn in range(-int(nconn / 2), int(nconn / 2)+1):
                    fields[:, 0, :, i_ball:i_ball+nz, :] = sliced_field[:, i_conn, :, :, :] * (-1)**i_conn
                    i_ball += nz

        else:
            print(f'No field file for {field_file}')
            fields[:, :, :, :, :] = None

        data['time'] = time
        gk_output.time = time
        data['fields'] = (('field', 'kx', 'ky', 'theta', 'time'), fields)

    def load_fluxes(self, pyro):
        """
        Loads fluxes into GKOutput.data DataSet
        """
        import csv

        gk_output = pyro.gk_output
        data = gk_output.data

        run_directory = pyro.run_directory
        flux_file = os.path.join(run_directory, f'nrg_{pyro.gene_output_number}')

        fluxes = np.empty((gk_output.nspecies, 3, 2, gk_output.ntime))

        nml = f90nml.read(f'parameters_{pyro.gene_output_number}')
        flux_istep = nml['in_out']['istep_nrg']
        field_istep = nml['in_out']['istep_field']

        if flux_istep < field_istep:
            time_skip = int(field_istep / flux_istep) - 1
        else:
            time_skip = 0

        if os.path.exists(flux_file):

            csv_file = open(flux_file)
            nrg_data = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)

            for i_time in range(gk_output.ntime):

                time = next(nrg_data)

                for i_species in range(gk_output.nspecies):
                    nrg_line = np.array(next(nrg_data), dtype=np.float)

                    # Particle
                    fluxes[i_species, 0, :, i_time] = nrg_line[4:6]

                    # Energy
                    fluxes[i_species, 1, :, i_time] = nrg_line[6:8]

                    # Momentum
                    fluxes[i_species, 2, :, i_time] = nrg_line[8:]

                # Skip time/data values in field print out is less
                if i_time != gk_output.ntime - 1:
                    for skip_t in range(time_skip):
                        for skip_s in range(gk_output.nspecies+1):
                            next(nrg_data)

        else:
            print(f'No flux file for {flux_file}')
            fluxes[:, :, :, :] = None

        data['fluxes'] = (("species", "moment", "field", "time"), fluxes)
