from cleverdict import CleverDict
from .constants import electron_charge, eps0, pi
from .kinetics import Kinetics
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

    def __init__(self, *args, **kwargs):

        s_args = list(args)

        if args and not isinstance(args[0], CleverDict) and isinstance(args[0], dict):
            s_args[0] = sorted(args[0].items())

            super(LocalSpecies, self).__init__(*s_args, **kwargs)

        # If no args then initialise ref values to None
        if len(args) == 0:

            _data_dict = {
                "tref": None,
                "nref": None,
                "mref": None,
                "vref": None,
                "lref": None,
                "Bref": None,
                "names": [],
            }

            super(LocalSpecies, self).__init__(_data_dict)

    def from_dict(self, species_dict, **kwargs):
        """
        Reads local species parameters from a dictionary

        """

        if isinstance(species_dict, dict):
            sort_species_dict = sorted(species_dict.items())

            super(LocalSpecies, self).__init__(*sort_species_dict, **kwargs)

    @classmethod
    def from_global_kinetics(cls, kinetics: Kinetics, psi_n: float, lref: float):
        # TODO this should replace from_kinetics
        local_species = cls()
        local_species.from_kinetics(kinetics, psi_n=psi_n, lref=lref)
        return local_species

    def from_kinetics(
        self,
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
            raise ValueError("Need reference length")

        if tref is None:
            tref = kinetics.species_data["electron"].get_temp(psi_n)

        if nref is None:
            nref = kinetics.species_data["electron"].get_dens(psi_n)

        if mref is None:
            mref = kinetics.species_data["deuterium"].get_mass()

        if vref is None:
            vref = np.sqrt(electron_charge * tref / mref)

        self["tref"] = tref
        self["nref"] = nref
        self["mref"] = mref
        self["vref"] = vref
        self["lref"] = lref

        ne = kinetics.species_data.electron.get_dens(psi_n)
        Te = kinetics.species_data.electron.get_temp(psi_n)
        coolog = 24 - np.log(np.sqrt(ne * 1e-6) / Te)

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

            vnewk = (
                np.sqrt(2)
                * pi
                * (z * electron_charge) ** 4
                * dens
                / (
                    (temp * electron_charge) ** 1.5
                    * np.sqrt(mass)
                    * (4 * pi * eps0) ** 2
                )
                * coolog
            )

            nu = vnewk * (lref / vref)

            # Local values
            species_dict["name"] = species
            species_dict["mass"] = mass / mref
            species_dict["z"] = z
            species_dict["dens"] = dens / nref
            species_dict["temp"] = temp / tref
            species_dict["vel"] = vel / vref
            species_dict["nu"] = nu

            # Gradients
            species_dict["a_lt"] = a_lt
            species_dict["a_ln"] = a_ln
            species_dict["a_lv"] = a_lv

            # Add to LocalSpecies dict
            self.add_species(name=species, species_data=species_dict)

    def update_pressure(self):
        """
        Calculate a_lp and pressure for species

        Returns
        -------
        self['a_lp']
        self['pressure']
        """

        pressure = 0.0
        a_lp = 0.0
        for name in self.names:
            species = self[name]
            # Total pressure
            pressure += species["temp"] * species["dens"]
            a_lp += (
                species["temp"] * species["dens"] * (species["a_lt"] + species["a_ln"])
            )

        self["pressure"] = pressure
        self["a_lp"] = a_lp

    def normalise(self):
        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        te = self["electron"].temp
        ne = self["electron"].dens
        for name in self.names:
            species_data = self[name]

            species_data.temp = species_data.temp / te
            species_data.dens = species_data.dens / ne

    def add_species(self, name, species_data):
        """
        Adds a species to LocalSpecies

        Parameters
        ----------
        name : Name of species
        species_data : Dictionary like object of Species Data

        Returns
        -------
        self[name] = SingleLocalSpecies
        """

        self[name] = self.SingleLocalSpecies(self, species_data)
        self.names.append(name)
        self.update_pressure()

    @property
    def nspec(self):
        return len(self.names)

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a LocalSpecies object

        Returns
        -------
        Copy of local_species object
        """

        new_local_species = LocalSpecies()

        for key, value in self.items():
            if key == "names" or isinstance(value, self.SingleLocalSpecies):
                pass
            else:
                setattr(new_local_species, key, value)

        # Add in each species
        for name in self["names"]:
            dict_keys = {
                "name": "name",
                "mass": "mass",
                "z": "z",
                "nu": "nu",
                "vel": "vel",
                "a_lv": "a_lv",
                "_dens": "dens",
                "_temp": "temp",
                "_a_ln": "a_ln",
                "_a_lt": "a_lt",
            }
            species_data = dict(
                (new_key, self[name][old_key])
                for (new_key, old_key) in dict_keys.items()
            )

            new_local_species.add_species(name, species_data)

        return new_local_species

    class SingleLocalSpecies:
        """
        Dictionary of local species parameters for one species

        For example
        SingleLocalSpecies['electron'] contains all the local info
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

        """

        def __init__(self, localspecies, species_dict):

            self.localspecies = localspecies
            self.name = None
            self.mass = None
            self.z = None
            self.dens = None
            self.temp = None
            self.vel = None
            self.nu = None
            self.a_lt = None
            self.a_ln = None
            self.a_lv = None

            if isinstance(species_dict, dict):
                for key, val in species_dict.items():
                    setattr(self, key, val)

        def __setitem__(self, key, value):
            self.__setattr__(key, value)

        def __getitem__(self, item):
            return self.__getattribute__(item)

        @property
        def dens(self):
            return self._dens

        @dens.setter
        def dens(self, value):
            self._dens = value
            self.localspecies.update_pressure()

        @property
        def temp(self):
            return self._temp

        @temp.setter
        def temp(self, value):
            self._temp = value
            self.localspecies.update_pressure()

        @property
        def a_ln(self):
            return self._a_ln

        @a_ln.setter
        def a_ln(self, value):
            self._a_ln = value
            self.localspecies.update_pressure()

        @property
        def a_lt(self):
            return self._a_lt

        @a_lt.setter
        def a_lt(self, value):
            self._a_lt = value
            self.localspecies.update_pressure()
