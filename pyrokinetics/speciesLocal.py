from collections import OrderedDict
from .constants import *
import numpy as np

class SpeciesLocal(OrderedDict):
    """ 
    Dictionary of local species parameters where the
    key is different species

    For example
    SpeciesLocal['electron'] contains all the local info
    for that species in a dictionary

    Local parameters are normalised to reference values

    name : Name
    mass : Mass
    z    : Charge
    dens : Density
    temp : Temperature
    vel  : Velocity
    nu   : Collision Frequency

    tprim : a/LT
    fprim : a/Ln
    uprim : a/Lv

    Reference values are also stored in SpeciesLocal under
    
    mref
    vref
    Tref
    nref
    Lref
    
    For example
    SpeciesLocal['electron']['dens'] contains density

    and

    SpeciesLocal['nref'] contains the reference density

    """
    def __init__(self,
                 *args, **kwargs):

         
        s_args = list(args)
        
        if (args and not isinstance(args[0], OrderedDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())
                    
        super(SpeciesLocal, self).__init__(*s_args, **kwargs)

        # If no args then intialise ref values to None
        if len(args) == 0:
            
            self['Tref'] = None
            self['nref'] = None
            self['mref'] = None
            self['vref'] = None
            self['Lref'] = None
            self['Bref'] = None
            
            self['nspec'] = None
            self['names'] = {}


    def fromDict(self,
                 spDict,
                 **kwargs):

        
        if (isinstance(spDict, dict)):
            sort_spDict = sorted(spDict.items())
                    
        super(SpeciesLocal, self).__init__(*sort_SpDict, **kwargs)


        
    def fromKinetics(self,
                     kin,
                     psiN=None,
                     Tref=None,
                     nref=None,
                     Bref=None,
                     vref=None,
                     mref=None,
                     Lref=None,
                      ):

        if Tref == None:
            Tref = kin.spData['electron'].getTemp(psiN)

        if nref == None:
            nref = kin.spData['electron'].getDens(psiN)

        if mref == None:
            mref = kin.spData['deuterium'].getMass()

        if vref == None:
            vref = np.sqrt(eCharge*Tref/mref)

        if Lref == None:
            raise ValueError('Need reference length')

        self['Tref'] = Tref
        self['nref'] = nref
        self['mref'] = mref
        self['vref'] = vref
        self['Lref'] = Lref

        self['nspec'] = len(kin.spName)
        self['names'] = kin.spName

        pressure = 0.0
        pprime = 0.0 
        for species in kin.spName:

            spDict = {}
            
            spData = kin.spData[species]

            z = spData.getCharge()
            mass = spData.getMass()
            temp = spData.getTemp(psiN)
            dens = spData.getDens(psiN)
            vel = spData.getVel(psiN)

            tprim = spData.getLT(psiN)
            fprim = spData.getLn(psiN)
            uprim = spData.getLv(psiN)

            coolog = 24 - np.log(np.sqrt(dens)/(temp*eCharge/bk))

            vnewk = (np.sqrt(2) * pi * (z * eCharge)**4 * dens /
            ((temp*eCharge)**1.5 * np.sqrt(mass) * (4*pi*eps0)**2)
            * coolog)

            nu = vnewk * (Lref/vref)

            # Local values
            spDict['name'] = species
            spDict['mass'] = mass / mref
            spDict['z'] = z
            spDict['dens'] = dens / nref
            spDict['temp'] = temp / Tref
            spDict['vel'] = vel / vref
            spDict['nu'] = nu

            # Gradients
            spDict['tprim'] = tprim
            spDict['fprim'] = fprim
            spDict['uprim'] = uprim

            # Total pressure gradient
            pressure += spDict['temp']*spDict['dens']
            pprime += spDict['temp']*spDict['dens'] * (spDict['tprim'] + spDict['fprim'])

            # Add to SpeciesLocal dict
            self[species] = spDict

        self['pressure'] = pressure
        self['pprime'] = pprime
