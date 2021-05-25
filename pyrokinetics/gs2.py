import f90nml
import copy
from .constants import *
from .local_species import LocalSpecies
from .numerics import Numerics

class GS2:
    """
    Basic GS2 object

    """

    pass


def read(pyro, data_file=None, template=False):
    
    gs2 = f90nml.read(data_file).todict()

    pyro.initial_datafile = copy.copy(gs2)
    
    if gs2['kt_grids_knobs']['grid_option'] in ['single', 'default']:
        pyro.linear = True
    else:
        pyro.linear = False

    pyro.gs2_input = gs2

    # Loads pyro object with equilibrium data
    if not template:
        load_pyro(pyro)

    # Load Pyro with numerics if they don't exist
    if not hasattr(pyro, 'numerics'):
        load_numerics(pyro, gs2)

def load_pyro(pyro):

    # Geometry
    gs2 = pyro.gs2_input

    gs2_eq = gs2['theta_grid_knobs']['equilibrium_option']

    if gs2_eq in ['eik', 'default']:

        try:
            local_eq = gs2['theta_grid_eik_knobs']['local_eq']
        except KeyError:
            local_eq = True

        try:
            iflux = gs2['theta_grid_eik_knobs']['iflux']
        except KeyError:
            iflux = 0

        if local_eq:
            if iflux == 0:
                pyro.geometry_type = 'Miller'
            else:
                pyro.geometry_type = 'Fourier'
        else:
            pyro.geometry_type = 'Global'
        
    #  Load GS2 with Miller Object
    if pyro.geometry_type == 'Miller':
        load_miller(pyro, gs2)
        
    else:
        raise NotImplementedError

    # Load GS2 with species data
    load_local_species(pyro, gs2)

    # Load Pyro with numerics
    load_numerics(pyro, gs2)

def write(pyro, filename):
    """
    For a given pyro object write a GS2 input file

    """

    gs2_input = pyro.gs2_input

    # Geometry data
    if pyro.geometry_type == 'Miller':
        miller = pyro.miller

        # Ensure Miller settings in input file
        gs2_input['theta_grid_knobs']['equilibrium_option'] = 'eik'
        gs2_input['theta_grid_eik_knobs']['iflux'] = 0
        gs2_input['theta_grid_eik_knobs']['local_eq'] = True
        gs2_input['theta_grid_parameters']['geometry_type'] = 0

        # Reference B field
        bref = miller['B0']

        shat = miller['shat']
        # Assign Miller values to input file
        pyro_gs2_miller = pyro_to_gs2_miller()

        for key, val in pyro_gs2_miller.items():
            gs2_input[val[0]][val[1]] = miller[key]

    else:
        raise NotImplementedError(f'Writing {pyro.geometry_type} for GS2 not supported yet')

    # Kinetic data
    local_species = pyro.local_species
    gs2_input['species_knobs']['nspec'] = local_species['nspec']
    
    pyro_gs2_species = pyro_to_gs2_species()

    for iSp, name in enumerate(local_species['names']):
        
        species_key = f'species_parameters_{iSp+1}'

        if name == 'electron':
            gs2_input[species_key]['type'] = 'electron'
        else:
            try:
                gs2_input[species_key]['type'] = 'ion'
            except KeyError:
                gs2_input[species_key] = copy.copy(gs2_input['species_parameters_1'])
                gs2_input[species_key]['type'] = 'ion'
                
                gs2_input[f'dist_fn_species_knobs_{iSp+1}'] = gs2_input[f'dist_fn_species_knobs_{iSp}']

        for key, val in pyro_gs2_species.items():
            gs2_input[species_key][val] = local_species[name][key]

        # Account for sqrt(2) in vth
        gs2_input[species_key]['vnewk'] = local_species[name]['nu'] / sqrt2

    # If species are defined calculate beta
    if local_species['nref'] is not None:

        pref = local_species['nref'] * local_species['tref'] * electron_charge
        
        beta = pref/bref**2 * 8 * pi * 1e-7

    # Calculate from reference  at centre of flux surface
    else:
        if pyro.geometry_type == 'Miller':
            if miller['B0'] is not None:
                beta = 1 / miller['B0'] ** 2 * (miller['rgeo'] / miller['Rmaj']) ** 2
            else:
                beta = 0.0
        else:
            raise NotImplementedError

    gs2_input['parameters']['beta'] = beta

    # Numerics
    numerics = pyro.numerics

    if numerics['nky'] == 1:
        gs2_input['kt_grids_knobs']['grid_option'] = 'single'

        gs2_input['kt_grids_single_parameters']['aky'] = numerics['ky'] * sqrt2
        gs2_input['kt_grids_single_parameters']['theta0'] = numerics['theta0']
        gs2_input['theta_grid_parameters']['nperiod'] = numerics['nperiod']

    else:
        gs2_input['kt_grids_knobs']['grid_option'] = 'box'

        gs2_input['kt_grids_box_parameters']['nx'] = int(((numerics['nky'] - 1) * 3/2) + 1)
        gs2_input['kt_grids_box_parameters']['ny'] = int(((numerics['nky']-1) * 3) + 1)

        gs2_input['kt_grids_box_parameters']['y0'] = - gs2_input['ky'] * sqrt2

        # Currently forces NL sims to have nperiod = 1
        gs2_input['theta_grid_parameters']['nperiod'] = 1

        if abs(shat) < 1e-6:
            gs2_input['kt_grids_box_parameters']['x0'] = 2 * pi / numerics['kx'] / sqrt2
        else:
            gs2_input['kt_grids_box_parameters']['jtwist'] = int((numerics['ky'] * shat * 2 * pi/ numerics['kx']) + 0.1)

    gs2_input['theta_grid_parameters']['ntheta'] = numerics['ntheta']

    gs2_input['le_grids_knobs']['negrid'] = numerics['nenergy']
    gs2_input['le_grids_knobs']['ngauss'] = numerics['npitch']

    if numerics['nonlinear']:
        gs2_input['nonlinear_terms_knobs']['nonlinear_mode'] = 'on'
    else:
        try:
            gs2_input['nonlinear_terms_knobs']['nonlinear_mode'] = 'off'
        except KeyError:
            pass

    gs2_nml = f90nml.Namelist(gs2_input)
    gs2_nml.float_format = pyro.float_format
    gs2_nml.write(filename, force=True)


def load_miller(pyro, gs2):
    """ Load Miller obejct from GS2 file
    """

    from .miller import Miller
    
    # Set some defaults here
    gs2['theta_grid_eik_knobs']['bishop'] = 4
    gs2['theta_grid_eik_knobs']['irho'] = 2
    gs2['theta_grid_eik_knobs']['iflux'] = 0
    pyro_gs2_miller = pyro_to_gs2_miller()
    
    miller = Miller()
    
    for key, val in pyro_gs2_miller.items():
        miller[key] = gs2[val[0]][val[1]]

    miller['delta'] = np.sin(miller['tri'])
    miller['s_kappa'] = miller['kappri'] * miller['rho'] / miller['kappa']

    # Get beta normalised to R_major(in case R_geo != R_major)
    beta = gs2['parameters']['beta'] * (miller['Rmaj']/miller['Rgeo'])**2

    # Assume pref*8pi*1e-7 = 1.0
    if beta != 0.0:
        miller['B0'] = np.sqrt(1.0/beta**0.5)
        # Can only know Bunit/B0 from local Miller
        miller['Bunit'] = miller.get_bunit_over_b0() * miller['B0']

    else:
        # If beta = 0
        miller['B0'] = None
        miller['Bunit'] = None

    pyro.miller = miller


def load_local_species(pyro, gs2):
    
    nspec = gs2['species_knobs']['nspec']
    pyro_gs2_species = pyro_to_gs2_species()

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

        species_data = {}
        
        gs2_key = f'species_parameters_{i_sp+1}'

        gs2_data = gs2[gs2_key]
        
        gs2_type = gs2_data['type']
                
        for pyro_key, gs2_key in pyro_gs2_species.items():
            species_data[pyro_key] = gs2_data[gs2_key]
            
        species_data['vel'] = 0.0
        
        if species_data['z'] == -1:
            name = 'electron'
            te = species_data['temp']
            ne = species_data['dens']
        else:
            ion_count+=1
            name = f'ion{ion_count}'

        pressure += species_data['temp']*species_data['dens']
        a_lp += species_data['temp']*species_data['dens'] * (species_data['a_lt'] + species_data['a_ln'])

        # Account for sqrt(2) in vth
        species_data['nu'] = gs2_data['vnewk'] * sqrt2

        species_data['name'] = name
        
        # Add individual species data to dictionary of species
        local_species[name] = species_data
        local_species['names'].append(name)

    local_species['pressure'] = pressure
    local_species['a_lp'] = a_lp / (ne * te)
    
    # Add local_species
    pyro.local_species = local_species


def pyro_to_gs2_miller():
    """
    Generates dictionary of equivalent pyro and gs2 parameter names
    for miller parameters
    """

    pyro_gs2_param = {
        'rho' : ['theta_grid_parameters', 'rhoc'],
        'Rmaj' : ['theta_grid_parameters', 'rmaj'],
        'Rgeo' : ['theta_grid_parameters', 'r_geo'],
        'q' : ['theta_grid_parameters', 'qinp'],
        'kappa' : ['theta_grid_parameters', 'akappa'],
        'kappri' : ['theta_grid_parameters', 'akappri'],
        'tri' : ['theta_grid_parameters', 'tri'],
        's_delta' : ['theta_grid_parameters', 'tripri'],
        'shat' : ['theta_grid_eik_knobs', 's_hat_input'],
        'shift' : ['theta_grid_parameters', 'shift'],
        'beta_prime' : ['theta_grid_eik_knobs', 'beta_prime_input'],
        }

    return pyro_gs2_param

def pyro_to_gs2_species():
    """
    Generates dictionary of equivalent pyro and gs2 parameter names
    for miller parameters
    """

    pyro_gs2_species = {
        'mass' : 'mass',
        'z' : 'z',
        'dens' : 'dens',
        'temp' : 'temp',
        'nu' : 'vnewk',
        'a_lt' : 'tprim',
        'a_ln' : 'fprim',
        'a_lv' : 'uprim'
        }

    return pyro_gs2_species


def add_flags(pyro, flags):
    """
    Add extra flags to GS2 input file

    """

    for key, parameter in flags.items():
        for param, val in parameter.items():

            pyro.gs2_input[key][param] = val


def load_numerics(pyro, gs2):
    """
    Load GS2 numerics into Pyro

    """

    grid_type = gs2['kt_grids_knobs']['grid_option']
    numerics = Numerics()

    # Need shear for map theta0 to kx
    shat = pyro.miller['shat']

    # Fourier space grid
    # Linear simulation
    if grid_type in ['single', 'default']:
        numerics['nky'] = 1
        numerics['nkx'] = 1
        numerics['ky'] = gs2['kt_grids_single_parameters']['aky'] / sqrt2

        numerics['kx'] = 0.0

        try:
            numerics['theta0'] = gs2['kt_grids_single_parameters']['theta0']
        except KeyError:
            numerics['theta0'] = 0.0

    # Nonlinear/multiple modes in box
    elif grid_type == 'box':
        box = 'kt_grids_box_parameters'
        keys = gs2[box].keys()

        # Set up ky grid
        if 'ny' in keys:
            numerics['nky'] = int((gs2[box]['n0'] - 1) / 3 + 1)
        elif 'n0' in keys:
            numerics['nky'] = gs2[box]['n0']
        elif 'nky' in keys:
            numerics['nky'] = gs2[box]['naky']
        else:
            raise NotImplementedError(f'ky grid details not found in {keys}')

        if 'y0' in keys:
            if gs2[box]['y0'] < 0.0:
                numerics['ky'] = - gs2[box]['y0'] / sqrt2
            else:
                numerics['ky'] = 1/gs2[box]['y0'] / sqrt2
        else:
            raise NotImplementedError(f'Min ky details not found in {keys}')

        if 'nx' in keys:
            numerics['nkx'] =  int((2*gs2[box]['nx'] - 1)/3+1)
        elif 'ntheta0' in keys():
            numerics['nkx'] = int((2*gs2[box]['ntheta0'] - 1)/3+1)
        else:
            raise NotImplementedError('kx grid details not found in {keys}')

        if abs(shat) > 1e-6:
            numerics['kx'] = numerics['ky'] * shat * 2 * pi / gs2[box]['jtwist']
        else:
            numerics['kx'] = 2 * pi / gs2[box]['x0'] / sqrt2

    # Theta grid
    numerics['ntheta'] = gs2['theta_grid_parameters']['ntheta']
    numerics['nperiod'] = gs2['theta_grid_parameters']['nperiod']

    # Velocity grid
    try:
        numerics['nenergy'] = gs2['le_grids_knobs']['nesub'] + gs2['le_grids_knobs']['nesuper']
    except KeyError:
        numerics['nenergy'] = gs2['le_grids_knobs']['negrid']

    # Currently using number of un-trapped pitch angles
    numerics['npitch'] = gs2['le_grids_knobs']['ngauss']

    try:
        nl_mode = gs2['nonlinear_terms_knobs']['nonlinear_mode']
    except KeyError:
        nl_mode = 'off'

    if nl_mode == 'on':
        numerics['nonlinear'] = True
    else:
        numerics['nonlinear'] = False

    pyro.numerics = numerics
