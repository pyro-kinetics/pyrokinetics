import f90nml
import numpy as np
import copy
from .constants import *
from .speciesLocal import SpeciesLocal

class CGYRO:
    """
    Basic CGYRO object

    """

    pass

def read(pyro, datafile=None, template=False):

    cgyro = cgyroParser(datafile)

    pyro.initial_datafile = copy.copy(cgyro)

    try:
        nl_flag = cgyro['NONLINEAR_FLAG']

        if nl_flag == 0:
            
            pyro.linear = True
        else:
            pyro.linear = False
    except KeyError:
        pyro.linear = True

    pyro.cgyroin = cgyro
    
    if not template:
        loadCGYRO(pyro)

def loadCGYRO(pyro):

    # Geometry
    cgyro = pyro.cgyroin

    cgyroeq = cgyro['EQUILIBRIUM_MODEL']

    if cgyroeq == 2:
        pyro.geoType = 'Miller'
    elif cgyroeq == 3:
        pyro.geoType = 'Fourier'
    elif cgyroeq == 1:
        pyro.geoType = 'SAlpha'
        
    #  Load CGYRO with Miller Object
    if pyro.geoType == 'Miller':
        loadMiller(pyro, cgyro)
        
    else:
        raise NotImplementedError

    # Load Species
    loadSpeciesLocal(pyro, cgyro)
    
def write(pyro, filename):
    """
    For a given pyro object write a CGYRO input file

    """

    cgyro_input = pyro.cgyroin

    # Geometry data
    if pyro.geoType == 'Miller':
        mil = pyro.mil

        # Ensure Miller settings in input file
        cgyro_input['EQUILIBRIUM_MODEL'] = 2

        # Reference B field - Bunit = q/r dpsi/dr
        q = mil['q']
        rho = mil['rho']
        dpsidrho = mil['dpsidr']
        amin = mil['amin']
        
        Bunit = q/rho * dpsidrho / amin

        # Assign Miller values to input file
        pyro_cgyro_miller = gen_pyro_cgyro_miller()

        for key, val in pyro_cgyro_miller.items():
            cgyro_input[val] = mil[key]

    # Kinetic data
    spLocal = pyro.spLocal
    cgyro_input['N_SPECIES'] = spLocal['nspec']

    for iSp, name in enumerate(spLocal['names']):
        pyro_cgyro_species = gen_pyro_cgyro_species(iSp+1)
        
        for pKey, cKey in pyro_cgyro_species.items():
            cgyro_input[cKey] = spLocal[name][pKey]

    cgyro_input['NU_EE'] = spLocal['electron']['nu']
    
    # Calculate beta
    # Initialise as CGYRO input file beta
    beta = cgyro_input['BETAE_UNIT']
    
    if spLocal['nref'] is not None:

        pref = spLocal['nref'] * spLocal['Tref'] * eCharge
        
        beta = pref/Bunit**2 * 8 * pi * 1e-7
    
    cgyro_input['BETAE_UNIT'] = beta


    # Calculate beta_prime
    
    
    toFile(cgyro_input, filename)


def cgyroParser(datafile):
    """ Parse CGYRO input file to dict
    """
    import re
    
    f = open(datafile)

    keys = []
    values = []

    for line in f:
        rawdata = line.strip().split("  ")

        if rawdata[0] != '':
            # Ignore commented lines
            if rawdata[0][0] != '#':

                # Splits by #,= and remves whitespace
                input_data = [data for data in re.split(
                    '=', rawdata[0]) if data != '']

                keys.append(input_data[0])

                if not input_data[1].isalpha():
                    values.append(eval(input_data[1]))
                else:
                    values.append(input_data[1])

    cgyro_dict = {}
    for key, value in zip(keys, values):
        cgyro_dict[key] = value

    return cgyro_dict

def toFile(cgyro_dict, filename):
    # Writes input file for cgyro from cgyro_dict
    # into the directory specified

    new_cgyro_input = open(filename, 'w+')

    for key, value in cgyro_dict.items():
        if isinstance(value, int):
            line = '{}={}\n'.format(key, value)
        elif isinstance(value, str):
            line = '{}={}\n'.format(key, value)
        else:
            if abs(value) >= 0.002:
                line = '{}={:5.4f}\n'.format(key, value)
            else:
                line = '{}={:9.7f}\n'.format(key, value)

        new_cgyro_input.write(line)

    new_cgyro_input.close()

def loadMiller(pyro, cgyro):
    """ Load Miller obejct from CGYRO file
    """

    from .miller import Miller
    
    # Set some defaults here
    cgyro['EQUILIBRIUM_MODEL'] = 2
    
    pyro_cgyro_miller = gen_pyro_cgyro_miller()
    
    mil = {}
    
    for key, val in pyro_cgyro_miller.items():
        mil[key] = cgyro[val]

    mil['kappri'] = mil['s_kappa'] * mil['kappa'] / mil['rho']
    mil['tri'] = np.arcsin(mil['delta'])
    
    mil['Bgeo'] = 1.0
    
    pyro.mil = Miller(mil)

def loadSpeciesLocal(pyro, cgyro):
    """
    Load CGYRO with species data
    """

    nspec = cgyro['N_SPECIES']
    
    # Dictionary of local species parameters
    spLocal = SpeciesLocal()
    spLocal['nspec'] = nspec
    spLocal['nref'] = None
    
    ionCount = 0

    # Load each species into a dictionary
    for iSp in range(nspec):

        pyro_cgyro_species = gen_pyro_cgyro_species(iSp+1)
        spData = {}
        for pKey, cKey in pyro_cgyro_species.items():
            
            spData[pKey] = cgyro[cKey]

        spData['vel'] = 0.0
        spData['uprim'] = 0.0
        
        if spData['z'] == -1:
            name = 'electron'
            spData['nu'] = cgyro['NU_EE']
            te = spData['temp']
            ne = spData['dens']
        else:
            ionCount+=1
            name = f'ion{ionCount}'

        spData[name] = name
        
        # Add individual species data to dictionary of species
        spLocal[name] = spData

    # Get collision frequency of ion species
    nu_ee = cgyro['NU_EE']
    for ion in range(ionCount):
        key = f'ion{ion+1}'

        nion = spLocal[key]['dens']
        tion = spLocal[key]['temp']

        # Not exact at log(Lambda) does change but pretty close...
        spLocal[key]['nu'] = nu_ee * (nion / tion**1.5) / (ne / te**1.5)

    # Add spLocal 
    pyro.spLocal = spLocal


def gen_pyro_cgyro_miller():
    """
    Generates dictionary of equivalent pyro and cgyro parameter names
    for miller parameters
    """


    pyro_cgyro_param = {
        'rho' : 'RMIN',
        'rmaj' : 'RMAJ',
        'rgeo' : 'RMAJ',
        'q' : 'Q',
        'kappa' : 'KAPPA',
        's_kappa' : 'S_KAPPA',
        'delta' : 'DELTA',
        's_delta' : 'S_DELTA',
        'shat' :  'S',
        'shift' : 'SHIFT',
        }


    return pyro_cgyro_param

def gen_pyro_cgyro_species(iSp):
    """
    Generates dictionary of equivalent pyro and cgyro parameter names
    for miller parameters
    """


    pyro_cgyro_species = {
        'mass' : f'MASS_{iSp}',
        'z'    : f'Z_{iSp}',
        'dens' : f'DENS_{iSp}',
        'temp' : f'TEMP_{iSp}',
        'tprim' : f'DLNTDR_{iSp}',
        'fprim' : f'DLNNDR_{iSp}',
        }

    return pyro_cgyro_species

    
def addFlags(pyro, flags):
    """
    Add extra flags to CGYRO input file

    """
    
    for key, value in flags.items():
            pyro.cgyroin[key] = value
