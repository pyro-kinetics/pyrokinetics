import f90nml
import numpy as np
import copy
from .constants import *
from .speciesLocal import SpeciesLocal

class GS2:
    """
    Basic GS2 object

    """

    pass

def read(pyro, datafile=None, template=False):
    
    gs2 = f90nml.read(datafile).todict()

    pyro.initial_datafile = copy.copy(gs2)
    
    if gs2['kt_grids_knobs']['grid_option'] in ['single', 'default']:
        pyro.linear = True
    else:
        pyro.linear = False

    pyro.gs2in = gs2

    if not template:
        loadGS2(pyro)

def loadGS2(pyro):

    # Geometry
    gs2 = pyro.gs2in

    gs2eq = gs2['theta_grid_knobs']['equilibrium_option']

    if gs2eq in ['eik', 'default']:

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
                pyro.geoType = 'Miller'
            else:
                pyro.geoType = 'Fourier'
        else:
            pyro.geoType = 'Global'
        
    #  Load GS2 with Miller Object
    if pyro.geoType == 'Miller':
        loadMiller(pyro, gs2)
        
    else:
        raise NotImplementedError

    # Load GS2 with species data
    loadSpeciesLocal(pyro, gs2)
    

def write(pyro, filename):
    """
    For a given pyro object write a GS2 input file

    """

    gs2_input = pyro.gs2in

    # Geometry data
    if pyro.geoType == 'Miller':
        mil = pyro.mil

        # Ensure Miller settings in input file
        gs2_input['theta_grid_knobs']['equilibrium_option'] = 'eik'
        gs2_input['theta_grid_eik_knobs']['iflux'] = 0
        gs2_input['theta_grid_eik_knobs']['local_eq'] = True
        gs2_input['theta_grid_parameters']['geoType'] = 0

        # Reference B field
        Bref = mil['Bgeo']
        
        # Assign Miller values to input file
        pyro_gs2_miller = gen_pyro_gs2_miller()

        for key, val in pyro_gs2_miller.items():
            gs2_input[val[0]][val[1]] = mil[key]

    # Kinetic data
    spLocal = pyro.spLocal
    gs2_input['species_knobs']['nspec'] = spLocal['nspec']
    
    pyro_gs2_species = gen_pyro_gs2_species()

    for iSp, name in enumerate(spLocal['names']):
        
        spKey = f'species_parameters_{iSp+1}'

        if name == 'electron':
            gs2_input[spKey]['type'] = 'electron'
        else:
            try:
                gs2_input[spKey]['type'] = 'ion'
            except KeyError:
                gs2_input[spKey] = copy.copy(gs2_input['species_parameters_1'])
                gs2_input[spKey]['type'] = 'ion'
                
                gs2_input[f'dist_fn_species_knobs_{iSp+1}'] = gs2_input[f'dist_fn_species_knobs_{iSp}']

        for key, val in pyro_gs2_species.items():

            gs2_input[spKey][val] = spLocal[name][key]

    
    # Calculate beta
    # Initialise as GS2 input file beta
    beta = gs2_input['parameters']['beta']
    
    if spLocal['nref'] is not None:

        pref = spLocal['nref'] * spLocal['Tref'] * eCharge
        
        beta = pref/Bref**2 * 8 * pi * 1e-7
    
    gs2_input['parameters']['beta'] = beta

    
    gs2_nml = f90nml.Namelist(gs2_input)
    
    gs2_nml.write(filename, force=True)


def loadMiller(pyro, gs2):
    """ Load Miller obejct from GS2 file
    """

    from .miller import Miller
    
    # Set some defaults here
    gs2['theta_grid_eik_knobs']['bishop'] = 4
    
    pyro_gs2_miller = gen_pyro_gs2_miller()
    
    mil = {}
    
    for key, val in pyro_gs2_miller.items():
        mil[key] = gs2[val[0]][val[1]]

    mil['delta'] = np.sin(mil['tri'])
    mil['s_kappa'] = mil['kappri'] * mil['rho'] / mil['kappa']
    mil['Bgeo'] = None
    
    pyro.mil = Miller(mil)

def loadSpeciesLocal(pyro, gs2):
    
    nspec = gs2['species_knobs']['nspec']
    pyro_gs2_species = gen_pyro_gs2_species()

    # Dictionary of local species parameters
    spLocal = SpeciesLocal()
    spLocal['nspec'] = nspec
    spLocal['nref'] = None
    
    ionCount = 0

    # Load each species into a dictionary
    for iSp in range(nspec):
        gs2Key = f'species_parameters_{iSp+1}'

        gs2Data = gs2[gs2Key]
        
        gs2Type = gs2Data['type']
        
        if gs2Type == 'electron':
            name = 'electron'
        else:
            ionCount+=1
            name = f'ion{ionCount}'

        spData = {}
        spData[name] = name
        
        for pKey, gKey in pyro_gs2_species.items():
            spData[pKey] = gs2Data[gKey]
        spData['vel'] = 0.0

        # Add individual species data to dictionary of species
        spLocal[name] = spData

    # Add spLocal 
    pyro.spLocal = spLocal

def gen_pyro_gs2_miller():
    """
    Generates dictionary of equivalent pyro and gs2 parameter names
    for miller parameters
    """


    pyro_gs2_param = {
        'rho' : ['theta_grid_parameters', 'rhoc'],
        'rmaj' : ['theta_grid_parameters', 'rmaj'],
        'rgeo' : ['theta_grid_parameters', 'r_geo'],
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

def gen_pyro_gs2_species():
    """
    Generates dictionary of equivalent pyro and gs2 parameter names
    for miller parameters
    """


    pyro_gs2_species = {
        'mass' : 'mass',
        'z'    : 'z',
        'dens' : 'dens',
        'temp' : 'temp',
        'nu'   : 'vnewk',
        'tprim' : 'tprim',
        'fprim' : 'fprim',
        'uprim' : 'uprim'
        }

    return pyro_gs2_species

    
def addFlags(pyro, flags):
    """
    Add extra flags to GS2 input file

    """
    
    for key, parameter in flags.items():
        for param, val in parameter.items():

            pyro.gs2in[key][param] = val
