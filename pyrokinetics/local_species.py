import warnings

from cleverdict import CleverDict
from .constants import pi
from .kinetics import Kinetics
from .normalisation import ureg, SimulationNormalisation as Normalisation
import numpy as np
from typing import Dict, Optional


class LocalSpecies(CleverDict):
    r"""
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

    inverse_lt : 1/Lt
    inverse_ln : 1/Ln
    inverse_lv : 1/Lv

    zeff : Zeff :math:`\sum_{ions} n_i Z_i^2 / n_e`

    """

    def __init__(self, *args, **kwargs):
        s_args = list(args)

        if args and not isinstance(args[0], CleverDict) and isinstance(args[0], dict):
            s_args[0] = sorted(args[0].items())

            super(LocalSpecies, self).__init__(*s_args, **kwargs)

        # If no args then initialise ref values to None
        if len(args) == 0:
            _data_dict = {
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
    def from_global_kinetics(
        cls, kinetics: Kinetics, psi_n: float, local_norm: Normalisation
    ):
        # TODO this should replace from_kinetics
        local_species = cls()
        local_species.from_kinetics(kinetics, psi_n=psi_n, norm=local_norm)
        return local_species

    def from_kinetics(self, kinetics, psi_n, norm):
        """
        Loads local species data from kinetics object

        """

        ne = kinetics.species_data.electron.get_dens(psi_n)
        Te = kinetics.species_data.electron.get_temp(psi_n)

        # FIXME: What are these units?
        coolog = 24 - np.log(np.sqrt(ne.m * 1e-6) / Te.m)

        for species in kinetics.species_names:
            species_dict = CleverDict()

            species_data = kinetics.species_data[species]

            z = species_data.get_charge()
            mass = species_data.get_mass()
            temp = species_data.get_temp(psi_n)
            dens = species_data.get_dens(psi_n)
            vel = species_data.get_velocity(psi_n)

            inverse_lt = species_data.get_norm_temp_gradient(psi_n)
            inverse_ln = species_data.get_norm_dens_gradient(psi_n)
            inverse_lv = species_data.get_norm_vel_gradient(psi_n)

            vnewk = (
                np.sqrt(2)
                * pi
                * (z**4)
                * dens
                / ((temp**1.5) * np.sqrt(mass) * (4 * pi * norm.units.eps0) ** 2)
                * coolog
            )

            # Local values
            species_dict["name"] = species
            species_dict["mass"] = mass
            species_dict["z"] = z
            species_dict["dens"] = dens
            species_dict["temp"] = temp
            species_dict["vel"] = vel
            species_dict["nu"] = vnewk.to_base_units(norm)

            # Gradients
            species_dict["inverse_lt"] = inverse_lt
            species_dict["inverse_ln"] = inverse_ln
            species_dict["inverse_lv"] = inverse_lv

            # Add to LocalSpecies dict
            self.add_species(name=species, species_data=species_dict, norms=norm)

        self.normalise(norms=norm)

        self.set_zeff()
        self.check_quasineutrality(tol=1e-3)

    def set_zeff(self):
        """
        Calculates Z_eff from the kinetics object

        Returns
        -------
        self['zeff']
        """

        zeff = 0.0

        for name in self.names:
            if name == "electron":
                continue
            species = self[name]
            zeff += species["dens"] * species["z"] ** 2

        self.zeff = zeff / (-self["electron"]["dens"] * self["electron"]["z"])

    def check_quasineutrality(self, tol=1e-2):
        """
        Checks quasi-neutrality is satisfied and raises a warning if it is not

        """
        error = 0.0

        for name in self.names:
            species = self[name]
            error += species["dens"] * species["z"]

        error = error / (self["electron"]["dens"] * self["electron"]["z"])

        if abs(error) > tol:
            warnings.warn(
                f"Currently local species violates quasi-neutrality by {error.magnitude}"
            )

    def update_pressure(self, norms=None):
        """
        Calculate inverse_lp and pressure for species

        Returns
        -------
        self['inverse_lp']
        self['pressure']
        """

        pressure = 0.0
        inverse_lp = 0.0
        for name in self.names:
            species = self[name]
            # Total pressure
            pressure += species["temp"] * species["dens"]
            inverse_lp += (
                species["temp"]
                * species["dens"]
                * (species["inverse_lt"].m + species["inverse_ln"].m)
            )

        self["pressure"] = pressure

        if hasattr(inverse_lp, "magnitude"):
            # Cancel out units from pressure
            inverse_lp = inverse_lp.magnitude / ureg.lref_minor_radius

        self["inverse_lp"] = inverse_lp

    def normalise(self, norms=None):
        # Normalise to pyrokinetics normalisations and calculate total pressure gradient
        if norms is None:
            norms = Normalisation("local_species")

        for name in self.names:
            species_data = self[name]
            species_data["mass"] = species_data["mass"].to(norms.mref)
            species_data["z"] = species_data["z"].to(norms.qref)
            species_data["dens"] = species_data["dens"].to(norms.nref)
            species_data["temp"] = species_data["temp"].to(norms.tref)
            species_data["vel"] = species_data["vel"].to(norms.vref)
            species_data["nu"] = species_data["nu"].to(norms.vref / norms.lref)

            # Gradients use lref_minor_radius -> Need to switch to this norms lref using context
            species_data["inverse_lt"] = species_data["inverse_lt"].to(
                norms.lref**-1, norms.context
            )
            species_data["inverse_ln"] = species_data["inverse_ln"].to(
                norms.lref**-1, norms.context
            )
            species_data["inverse_lv"] = species_data["inverse_lv"].to(
                norms.lref**-1, norms.context
            )

        self.update_pressure(norms)

    def add_species(self, name, species_data, norms: Optional[Normalisation] = None):
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

        self[name] = self.SingleLocalSpecies(self, species_data, norms)
        self.names.append(name)
        self.update_pressure(norms)

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
                "_dens": "dens",
                "_temp": "temp",
                "_inverse_ln": "inverse_ln",
                "_inverse_lt": "inverse_lt",
                "_inverse_lv": "inverse_lv",
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

        inverse_lt : 1/Lt [units] [1 / lref_minor_radius]
        inverse_ln : 1/Ln [units] [1 / lref_minor_radius]
        inverse_lv : 1/Lv [units] [1 / lref_minor_radius]

        """

        def __init__(
            self,
            localspecies,
            species_dict: Dict[str, float],
            norms: Optional[Normalisation] = None,
        ):
            self.localspecies = localspecies
            self.norms = norms
            self.name = None
            self.mass = None
            self.z = None
            self.dens = None
            self.temp = None
            self.vel = None
            self.nu = None
            self.inverse_lt = None
            self.inverse_ln = None
            self.inverse_lv = None

            self.items = {}

            if isinstance(species_dict, dict):
                for key, val in species_dict.items():
                    setattr(self, key, val)
                    self.items[key] = val

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
            self.localspecies.update_pressure(self.norms)

        @property
        def temp(self):
            return self._temp

        @temp.setter
        def temp(self, value):
            self._temp = value
            self.localspecies.update_pressure(self.norms)

        @property
        def inverse_ln(self):
            return self._inverse_ln

        @inverse_ln.setter
        def inverse_ln(self, value):
            self._inverse_ln = value
            self.localspecies.update_pressure(self.norms)

        @property
        def inverse_lt(self):
            return self._inverse_lt

        @inverse_lt.setter
        def inverse_lt(self, value):
            self._inverse_lt = value
            self.localspecies.update_pressure(self.norms)

        @property
        def inverse_lv(self):
            return self._inverse_lv

        @inverse_lv.setter
        def inverse_lv(self, value):
            self._inverse_lv = value
            self.localspecies.update_pressure(self.norms)
