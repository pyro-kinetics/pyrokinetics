from cleverdict import CleverDict
from .constants import *
import numpy as np


class LocalSpecies(CleverDict):
    """ 
    Dictionary of local species parameters where the
    key is different species

    For example
    LocalSpecies['electron'] contains all the local info
    for that species in a dictionary

    Local parameters are normalised to reference values

    name : Name
    mass : Mass
    z    : Charge
    dens : Density
    temp : Temperature
    vel  : Velocity
    nu   : Collision Frequency

    a_lt : a/Lt
    a_ln : a/Ln
    a_lv : a/Lv

    Reference values are also stored in LocalSpecies under
    
    mref
    vref
    tref
    nref
    lref
    
    For example
    LocalSpecies['electron']['dens'] contains density

    and

    LocalSpecies['nref'] contains the reference density

    """
    def __init__(self,
                 *args, **kwargs):

         
        s_args = list(args)
        
        if (args and not isinstance(args[0], CleverDict)
            and isinstance(args[0], dict)):
            s_args[0] = sorted(args[0].items())
                    
            super(LocalSpecies, self).__init__(*s_args, **kwargs)

        # If no args then initialise ref values to None
        if len(args) == 0:

            _data_dict = {'tref': None, 'nref': None, 'mref': None, 'vref': None, 'lref': None, 'Bref': None,
                          'nspec': None, 'names': {}}

            super(LocalSpecies, self).__init__(_data_dict)

    def from_dict(self,
                  species_dict,
                  **kwargs):
        """
        Reads local species parameters from a dictionary

        """
        
        if isinstance(species_dict, dict):
            sort_species_dict = sorted(species_dict.items())
                    
            super(LocalSpecies, self).__init__(*sort_species_dict, **kwargs)

    def from_kinetics(self,
                      kinetics,
                      psi_n=None,
                      tref=None,
                      nref=None,
                      Bref=None,
                      vref=None,
                      mref=None,
                      lref=None,
                      ):
        """
        Loads local species data from kinetics object

        """

        if psi_n is None:
            raise ValueError("Need value of psi_n")

        if lref is None:
            raise ValueError('Need reference length')

        if tref is None:
            tref = kinetics.species_data['electron'].get_temp(psi_n)

        if nref is None:
            nref = kinetics.species_data['electron'].get_dens(psi_n)

        if mref is None:
            mref = kinetics.species_data['deuterium'].get_mass()

        if vref is None:
            vref = np.sqrt(electron_charge * tref / mref)

        self['tref'] = tref
        self['nref'] = nref
        self['mref'] = mref
        self['vref'] = vref
        self['lref'] = lref

        self['nspec'] = len(kinetics.species_names)
        self['names'] = kinetics.species_names

        pressure = 0.0
        a_lp = 0.0

        for species in kinetics.species_names:

            species_dict = CleverDict()

            species_data = kinetics.species_data[species]

            z = species_data.get_charge()
            mass = species_data.get_mass()
            temp = species_data.get_temp(psi_n)
            dens = species_data.get_dens(psi_n)
            vel = species_data.get_velocity(psi_n)

            a_lt = species_data.get_norm_temp_gradient(psi_n)
            a_ln = species_data.get_norm_dens_gradient(psi_n)
            a_lv = species_data.get_norm_vel_gradient(psi_n)

            coolog = 24 - np.log(np.sqrt(dens) / (temp * electron_charge / bk))

            vnewk = (np.sqrt(2) * pi * (z * electron_charge) ** 4 * dens /
                     ((temp * electron_charge) ** 1.5 * np.sqrt(mass) * (4 * pi * eps0) ** 2)
                     * coolog)

            nu = vnewk * (lref / vref)

            # Local values
            species_dict['name'] = species
            species_dict['mass'] = mass / mref
            species_dict['z'] = z
            species_dict['dens'] = dens / nref
            species_dict['temp'] = temp / tref
            species_dict['vel'] = vel / vref
            species_dict['nu'] = nu

            # Gradients
            species_dict['a_lt'] = a_lt
            species_dict['a_ln'] = a_ln
            species_dict['a_lv'] = a_lv

            # Total pressure gradient
            pressure += species_dict['temp']*species_dict['dens']
            a_lp += species_dict['temp']*species_dict['dens'] * (species_dict['a_lt'] + species_dict['a_ln'])

            # Add to LocalSpecies dict
            self[species] = species_dict

        self['pressure'] = pressure
        self['a_lp'] = a_lp
