import f90nml
import copy
from .constants import *
from .local_species import LocalSpecies
from .numerics import Numerics
from .gk_code import GKCode
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
        gene_input['box']['nspec'] = local_species.nspec

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

            beta = pref/bref**2 * 8 * pi * 1e-7

        # Calculate from reference  at centre of flux surface
        else:
            if pyro.local_geometry_type == 'Miller':
                if miller.B0 is not None:
                    beta = 1 / miller.B0** 2 * (miller.Rgeo / miller.Rmaj) ** 2
                else:
                    beta = 0.0
            else:
                raise NotImplementedError

        gene_input['general']['beta'] = beta

        # Numerics
        numerics = pyro.numerics

        if numerics['nonlinear']:
            
            gene_input['general']['nonlinear']= True
            gene_input['box']['nky0'] = numerics['nky']
            gene_input['box']['nkx'] = numerics['nkx']

        else:
            gene_input['general']['nonlinear'] = False

        # Currently forces NL sims to have nperiod = 1
        gene_nml = f90nml.Namelist(gene_input)
        gene_nml.float_format = pyro.float_format
        gene_nml.write(path_to_file, force=True)

    
    def load_local_geometry(self, pyro, gene):
        """
        Loads local geometry 
        """

        if pyro.local_geometry_type == 'Miller':
            self.load_miller(pyro,gene)

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

        miller.kappri = miller.s_kappa*miller.kappa / miller.rho
        miller.tri = np.arcsin(miller.delta)

        # Get beta normalised to R_major(in case R_geo != R_major)
        beta = gene['general']['beta'] * (miller.Rmaj/miller.Rgeo)**2

        # Assume pref*8pi*1e-7 = 1.0
        if beta != 0.0:
            miller.B0 = np.sqrt(1.0/beta)
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

        pressure = 0.0
        a_lp = 0.0

        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        for name in local_species.names:

            species_data = local_species[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne
            pressure += species_data.temp * species_data.dens
            a_lp += species_data.temp * species_data.dens * (species_data.a_lt + species_data.a_ln)

            species_data.name = name

            # Add individual species data to dictionary of species
            local_species[name] = species_data
            local_species.names.append(name)

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
            'rho' : ['geometry', 'minor_r'],
            'Rmaj' : ['geometry', 'major_r'],
            'Rgeo' : ['geometry', 'major_r'],
            'q' : ['geometry', 'q0'],
            'kappa' : ['geometry', 'kappa'],
            's_kappa' : ['geometry', 's_kappa'],     #checked
            'delta' : ['geometry', 'delta'],
            's_delta' : ['geometry', 's_delta'],    #checked
            'shat' : ['geometry', 'shat'],  
            'shift' : ['geometry', 'drr'],
            }

        return pyro_gene_param

    def pyro_to_gene_species(self):
        """
        Generates dictionary of equivalent pyro and gene parameter names
        for species parameters
        """

        pyro_gene_species = {
            'mass' : 'mass',
            'z' :  'charge',
            'dens': 'dens',
            'temp': 'temp',
            'a_lt' : 'omt',
            'a_ln' : 'omn',
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

        numerics['nky'] = gene['box']['nky0']
        numerics['nkx'] = gene['box']['nx0']
        numerics['nky'] = 1
        numerics['ky'] = gene['box']['kymin'] 
        numerics['kx'] = 1.0


        # Velocity grid

        try:
            nl_mode = gene['nonlinear']
        except KeyError:
            nl_mode = 0

        if nl_mode == 1:
            numerics['nonlinear'] = True
        else:
            numerics['nonlinear'] = False

        pyro.numerics = numerics
