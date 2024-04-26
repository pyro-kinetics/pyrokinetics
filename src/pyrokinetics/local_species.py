import warnings
from typing import Any, Dict, Iterable, Optional

import numpy as np
from cleverdict import CleverDict

from .constants import pi
from .kinetics import Kinetics
from .normalisation import SimulationNormalisation as Normalisation
from .normalisation import ureg


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
    omega0  : Angular velocity
    nu   : Collision Frequency

    inverse_lt : 1/Lt
    inverse_ln : 1/Ln
    domega_drho : Gradient in angular velocity

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

        coolog = 24 - np.log(np.sqrt(ne.m * 1e-6) / Te.m)

        for species in kinetics.species_names:
            species_dict = CleverDict()

            species_data = kinetics.species_data[species]

            z = species_data.get_charge(psi_n)
            mass = species_data.get_mass()
            temp = species_data.get_temp(psi_n)
            dens = species_data.get_dens(psi_n)
            omega0 = species_data.get_angular_velocity(psi_n)

            inverse_lt = species_data.get_norm_temp_gradient(psi_n)
            inverse_ln = species_data.get_norm_dens_gradient(psi_n)
            domega_drho = species_data.get_angular_velocity(psi_n).to(
                norm.vref / norm.lref
            ) * species_data.get_norm_ang_vel_gradient(psi_n).to(
                norm.lref**-1, norm.context
            )

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
            species_dict["omega0"] = omega0
            species_dict["nu"] = vnewk

            # Gradients
            species_dict["inverse_lt"] = inverse_lt
            species_dict["inverse_ln"] = inverse_ln
            species_dict["domega_drho"] = domega_drho

            # Add to LocalSpecies dict
            self.add_species(name=species, species_data=species_dict, norms=norm)

        self.normalise(norms=norm)

        self.set_zeff()
        self.check_quasineutrality(tol=1e-3)

    def set_zeff(self) -> None:
        """
        Calculates Z_eff from the kinetics object
        """

        zeff = 0.0

        for name in self.names:
            if name == "electron":
                continue
            species = self[name]
            zeff += species["dens"] * species["z"] ** 2

        self.zeff = zeff / (-self["electron"]["dens"] * self["electron"]["z"])

    def check_quasineutrality(self, tol: float = 1e-2) -> bool:
        """
        Checks quasi-neutrality is satisfied and raises a warning if it is not

        """
        error = 0.0
        error_gradient = 0.0

        for name in self.names:
            species = self[name]
            error += species["dens"] * species["z"]
            error_gradient += species["dens"] * species["z"] * species["inverse_ln"]

        error = error.magnitude
        error_gradient = error_gradient.magnitude

        if abs(error) > tol or abs(error_gradient) > tol:
            warnings.warn(
                f"""Currently local species violates quasi-neutrality in the
                    density by {error} and density gradient by {error_gradient}"""
            )
            return False
        return True

    def update_pressure(self, norms=None) -> None:
        """
        Calculate inverse_lp and pressure for species
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
            inverse_lp = inverse_lp / pressure / ureg.lref_minor_radius

        self["inverse_lp"] = inverse_lp

    def normalise(self, norms=None):
        """Normalise to pyrokinetics normalisations and calculate total pressure gradient"""

        if norms is None:
            norms = Normalisation("local_species")

        for name in self.names:
            species_data = self[name]
            species_data["mass"] = species_data["mass"].to(norms.mref)
            species_data["z"] = species_data["z"].to(norms.qref)
            species_data["dens"] = species_data["dens"].to(norms.nref)
            species_data["temp"] = species_data["temp"].to(norms.tref)
            species_data["omega0"] = species_data["omega0"].to(norms.vref / norms.lref)
            species_data["nu"] = species_data["nu"].to(norms.vref / norms.lref)

            # Gradients use lref_minor_radius -> Need to switch to this norms lref using context
            species_data["inverse_lt"] = species_data["inverse_lt"].to(
                norms.lref**-1, norms.context
            )
            species_data["inverse_ln"] = species_data["inverse_ln"].to(
                norms.lref**-1, norms.context
            )
            species_data["domega_drho"] = species_data["domega_drho"].to(
                norms.vref * norms.lref**-2, norms.context
            )

            # Avoid floating point errors
            for key in species_data.items.keys():
                if key == "name":
                    continue
                if np.isclose(species_data[key].m, np.round(species_data[key].m)):
                    species_data[key] = (
                        np.round(species_data[key].m) * species_data[key].units
                    )

        self.update_pressure(norms)

    def add_species(
        self,
        name: str,
        species_data: Dict[str, Any],
        norms: Optional[Normalisation] = None,
    ) -> None:
        """
        Adds a new species to LocalSpecies

        Parameters
        ----------
        name
            Name of species
        species_data
            Dictionary like object of Species Data
        """

        self[name] = self.SingleLocalSpecies(self, species_data, norms)
        self.names.append(name)
        self.update_pressure(norms)

    def remove_species(self, *names: str) -> None:
        """
        Removes a species from the LocalSpecies

        Parameters
        ----------
        names
            Names of species to remove

        Raises
        ------
        ValueError
            If there is no species with a given name.
        """
        unrecognised = [name for name in names if name not in self.names]
        if unrecognised:
            raise ValueError(f"Unrecognised species names {', '.join(unrecognised)}")
        for name in names:
            self.pop(name)
            self.names.remove(name)
        self.update_pressure()

    def merge_species(
        self,
        base_species: str,
        merge_species: Iterable[str],
        keep_base_species_z: bool = False,
        keep_base_species_mass: bool = False,
    ) -> None:
        """
        Merge multiple species into one. Performs a weighted average depending on the
        densities of each species to preserve quasineutrality.

        Parameters
        ----------
        base_species: str
            Names of species that will absorb other species
        merge_species: Iterable[str]
            List of species names to be merged into the base_species
        keep_base_species_z: bool
            Charge of new species
                True preserves base_species charge and adjusts ion density
                False/None preserves ion density (before/after merge) and adjusts z
        keep_base_species_mass: bool
            Mass of new species
                True keeps base_species mass
                False/None results in a density-weighted average

        Raises
        ------
        ValueError
            If there is no species with a given name.
        """

        if base_species not in self.names:
            raise ValueError(f"Unrecognised base_species name {base_species}")

        unrecognised = [name for name in merge_species if name not in self.names]
        if unrecognised:
            raise ValueError(
                f"Unrecognised merge_species names {', '.join(unrecognised)}"
            )

        # Remove duplicates, ensure the base_species is included
        merge_species = list(set(merge_species) | {base_species})

        # charge and density
        if keep_base_species_z:
            new_z = self[base_species].z
            new_dens = (
                sum(self[name].dens * self[name].z for name in merge_species) / new_z
            )
        else:
            new_dens = sum(self[name].dens for name in merge_species)
            new_z = (
                sum(self[name].dens * self[name].z for name in merge_species) / new_dens
            )

        # density gradient
        new_inverse_ln = sum(
            self[name].dens * self[name].z * self[name].inverse_ln
            for name in merge_species
        ) / (new_dens * new_z)

        # mass
        if keep_base_species_mass:
            new_mass = self[base_species].mass
        else:
            new_mass = (
                sum(self[name].mass * self[name].dens for name in merge_species)
                / new_dens
            )

        self[base_species].dens = new_dens
        self[base_species].z = new_z
        self[base_species].inverse_ln = new_inverse_ln
        self[base_species].mass = new_mass

        merge_species.remove(base_species)

        self.remove_species(*merge_species)
        self.update_pressure()
        self.check_quasineutrality()

    @property
    def nspec(self):
        return len(self.names)

    @property
    def domega_drho(self):
        dens = 0.0
        highest_dens_species = None
        for name in self.names:
            species = self[name]
            if species.z.m > 0 and species.dens > dens:
                dens = species.dens
                highest_dens_species = name

        _domega_drho = self[highest_dens_species].domega_drho
        return _domega_drho

    @domega_drho.setter
    def domega_drho(self, value):
        for name in self.names:
            species = self[name]
            species.domega_drho = value

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
                "omega0": "omega0",
                "_dens": "dens",
                "_temp": "temp",
                "_inverse_ln": "inverse_ln",
                "_inverse_lt": "inverse_lt",
                "_domega_drho": "domega_drho",
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
        omega0  : Angular Velocity
        nu   : Collision Frequency

        inverse_lt : 1/Lt [units] [1 / lref_minor_radius]
        inverse_ln : 1/Ln [units] [1 / lref_minor_radius]
        domega_drho : domega/drho [units] [vref / lref_minor_radius**2]
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
            self.omega0 = None
            self.nu = None
            self.inverse_lt = None
            self.inverse_ln = None
            self.domega_drho = None

            self.items = {}

            self._already_warned = False

            if isinstance(species_dict, dict):
                for key, val in species_dict.items():
                    setattr(self, key, val)
                    self.items[key] = val

        def __setitem__(self, key, value):
            self.__setattr__(key, value)

        def __getitem__(self, item):
            return self.__getattribute__(item)

        def __setattr__(self, key, value):
            if hasattr(self, key):
                attr = getattr(self, key)
                if hasattr(attr, "units") and not hasattr(value, "units"):
                    value *= attr.units
                    if not self._already_warned and str(attr.units) != "dimensionless":
                        warnings.warn(
                            f"missing unit from {key}, adding {attr.units}. To suppress this warning, specify units. Will maintain units if not specified from now on"
                        )
                        self._already_warned = True
            super().__setattr__(key, value)

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
        def domega_drho(self):
            return self._domega_drho

        @domega_drho.setter
        def domega_drho(self, value):
            self._domega_drho = value
