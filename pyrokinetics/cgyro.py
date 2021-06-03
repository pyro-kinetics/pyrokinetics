import copy
from .constants import *
from .local_species import LocalSpecies
from .numerics import Numerics
from .gk_code import GKCode
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

    def read(self, pyro, data_file=None, template=False):
        """
        Reads CGYRO input file into a dictionary
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
        Loads CGYRO dictions into Pyro object
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
                pyro.local_geometry.beta_prime = - pyro.local_species.a_lp / pyro.local_geometry.Bunit ** 2 * \
                                                 beta_prime_scale
            else:
                pyro.local_geometry.beta_prime = 0.0
        else:
            raise NotImplementedError

        self.load_numerics(pyro, cgyro)

    def write(self, pyro, file_name, directory='.'):
        """
        For a given pyro object write a CGYRO input file

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
                beta_prime_scale = - miller.beta_prime / (beta * local_species.a_lp) * \
                                   (miller.B0 / miller.Bunit) ** 2

        # Calculate beta from existing value from input
        else:
            if pyro.local_geometry_type == 'Miller':
                if miller.Bunit is not None:
                    beta = 1.0 / miller.Bunit ** 2
                    beta_prime_scale = - miller.beta_prime * miller.B0 ** 2 / local_species.a_lp
                else:
                    beta = 0.0
                    beta_prime_scale = 1.0

        cgyro_input['BETAE_UNIT'] = beta

        cgyro_input['BETA_STAR_SCALE'] = beta_prime_scale

        # Numerics
        numerics = pyro.numerics

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
        Parse CGYRO input file to dict
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
        Writes input file for cgyro from cgyro_dict

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
        Loads local geometry
        """

        if pyro.local_geometry_type == 'Miller':
            self.load_miller(pyro, cgyro)

    def load_miller(self, pyro, cgyro):
        """
        Load Miller object from CGYRO input file
        """

        # Set some defaults here
        cgyro['EQUILIBRIUM_MODEL'] = 2

        pyro_cgyro_miller = self.pyro_to_code_miller()

        miller = pyro.local_geometry

        for key, val in pyro_cgyro_miller.items():
            miller[key] = cgyro[val]

        miller.kappri = miller.s_kappa * miller.kappa / miller.rho
        miller.tri = np.arcsin(miller.delta)

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
        Load local_species from CGYRO input file
        """

        nspec = cgyro['N_SPECIES']

        # Dictionary of local species parameters
        local_species = LocalSpecies()
        local_species.nspec = nspec
        local_species.nref = None
        local_species.names = []

        ion_count = 0

        pressure = 0.0
        a_lp = 0.0

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

            pressure += species_data.temp * species_data.dens
            a_lp += species_data.temp * species_data.dens * (species_data.a_lt + species_data.a_ln)

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species[name] = species_data
            local_species.names.append(name)

        # CGYRO beta_prime scale
        local_species.pressure = pressure
        local_species.a_lp = a_lp / (ne * te)

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
            's_delta': 'S_DELTA',
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
