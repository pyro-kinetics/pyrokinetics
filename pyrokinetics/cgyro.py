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

    # Need species to set up beta_prime
    beta_prime_scale = cgyro['BETA_STAR_SCALE']

    if pyro.geoType == 'Miller':
        pyro.mil['beta_prime'] = - pyro.spLocal['pprime'] /pyro.mil['Bunit']**2 * beta_prime_scale 
    else:
        raise NotImplementedError

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
        Bref = mil['Bunit']

        # Assign Miller values to input file
        pyro_cgyro_miller = gen_pyro_cgyro_miller()

        for key, val in pyro_cgyro_miller.items():
            cgyro_input[val] = mil[key]
            
    else:
        raise NotImplementedError


    # Kinetic data
    spLocal = pyro.spLocal
    cgyro_input['N_SPECIES'] = spLocal['nspec']

    for iSp, name in enumerate(spLocal['names']):
        pyro_cgyro_species = gen_pyro_cgyro_species(iSp+1)
        
        for pKey, cKey in pyro_cgyro_species.items():
            cgyro_input[cKey] = spLocal[name][pKey]

    cgyro_input['NU_EE'] = spLocal['electron']['nu']


    # If species are defined calculate beta and beta_prime_scale
    if spLocal['nref'] is not None:

        pref = spLocal['nref'] * spLocal['Tref'] * eCharge

        pe = pref * spLocal['electron']['dens'] * spLocal['electron']['temp']
        
        beta = pe/Bref**2 * 8 * pi * 1e-7

        # Find BETA_STAR_SCALE from beta and pprime
        if pyro.geoType == 'Miller':
            beta_prime_scale = - mil['beta_prime'] / (beta* spLocal['pprime']) * (mil['B0']/mil['Bunit'])**2

    # Calculate beta from existing value from input
    else:
        if pyro.geoType == 'Miller':
            if mil['Bunit'] != None:
                beta = 1.0/mil['Bunit']**2
                print(beta)
                beta_prime_scale = - mil['beta_prime'] / (beta* spLocal['pprime'])
            else:
                beta = 0.0
                beta_prime_scale = 1.0

    cgyro_input['BETAE_UNIT'] = beta
    
    cgyro_input['BETA_STAR_SCALE'] = beta_prime_scale
    
    toFile(cgyro_input, filename, floatFormat=pyro.floatFormat)

def cgyroParser(datafile):
    """ Parse CGYRO input file to dict
    """
    import re
    
    f = open(datafile)

    keys = []
    values = []

    for line in f:
        rawdata = line.strip().split("  ")[0]
        if rawdata != '':
            # Ignore commented lines
            if rawdata[0] != '#':
                
                # Splits by #,= and remves whitespace
                input_data = [data.strip() for data in re.split(
                    '=', rawdata) if data != '']

                keys.append(input_data[0])

                if not input_data[1].isalpha():
                    values.append(eval(input_data[1]))
                else:
                    values.append(input_data[1])

    cgyro_dict = {}
    for key, value in zip(keys, values):
        cgyro_dict[key] = value

    return cgyro_dict

def toFile(cgyro_dict, filename, floatFormat='.6g'):
    # Writes input file for cgyro from cgyro_dict
    # into the directory specified

    new_cgyro_input = open(filename, 'w+')

    for key, value in cgyro_dict.items():
        if isinstance(value, float):
            line = f'{key} = {value:{floatFormat}}\n'
        else:
            line = f'{key} = {value}\n'
        new_cgyro_input.write(line)

    new_cgyro_input.close()

def loadMiller(pyro, cgyro):
    """ Load Miller obejct from CGYRO file
    """

    from .miller import Miller
    
    # Set some defaults here
    cgyro['EQUILIBRIUM_MODEL'] = 2
    
    pyro_cgyro_miller = gen_pyro_cgyro_miller()
    
    mil = Miller()
    
    for key, val in pyro_cgyro_miller.items():
        mil[key] = cgyro[val]

    mil['kappri'] = mil['s_kappa'] * mil['kappa'] / mil['rho']
    mil['tri'] = np.arcsin(mil['delta'])

    beta = cgyro['BETAE_UNIT']

    # Assume pref*8pi*1e-7 = 1.0
    if beta != 0:
        mil['Bunit'] = 1/(beta**0.5)
        BunitoverB0 = mil.getBunitOverB0()
        mil['B0'] = mil['Bunut']/BunitoverB0
    else:
        mil['Bunit'] = None
        mil['B0'] = None

    pyro.mil = mil

    
def loadSpeciesLocal(pyro, cgyro):
    """
    Load CGYRO with species data
    """

    nspec = cgyro['N_SPECIES']

    # Dictionary of local species parameters
    spLocal = SpeciesLocal()
    spLocal['nspec'] = nspec
    spLocal['nref'] = None
    spLocal['names'] = []
    
    ionCount = 0

    pressure = 0.0
    pprime = 0.0
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
            Te = spData['temp']
            ne = spData['dens']
            me = spData['mass']
        else:
            ionCount+=1
            name = f'ion{ionCount}'

        pressure += spData['temp']*spData['dens']
        pprime += spData['temp']*spData['dens'] * (spData['tprim'] + spData['fprim'])

        spData['name'] = name
        
        # Add individual species data to dictionary of species
        spLocal[name] = spData
        spLocal['names'].append(name)

    #CGYRO beta_prime scale
    spLocal['pressure'] = pressure
    spLocal['pprime'] = pprime / (ne * Te)

    # Get collision frequency of ion species
    nu_ee = cgyro['NU_EE']
    
    for ion in range(ionCount):
        key = f'ion{ion+1}'

        nion = spLocal[key]['dens']
        tion = spLocal[key]['temp']
        mion = spLocal[key]['mass']

        # Not exact at log(Lambda) does change but pretty close...
        spLocal[key]['nu'] = nu_ee * (nion / tion**1.5 / mion**0.5) / (ne / Te**1.5 /me**0.5)

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

def gen_pyro_cgyro_species(iSp=1):
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
