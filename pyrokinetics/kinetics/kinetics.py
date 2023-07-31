from typing import List, Optional
from cleverdict import CleverDict

from ..typing import PathLike
from ..file_utils import readable_from_file


@readable_from_file
class Kinetics:
    """
    Contains all the kinetic data in the form of Species objects.
    Data can be accessed via `species_data`, which is a CleverDict with each
    key being a species name. For example, electron data can be accessed via a call
    to ``kinetics.species_data["electron"]`` or ``kinetics.species_data.electron``.

    Each Species is provided with:

    - psi_n: ArrayLike       [units] dimensionless
        1D array of normalised poloidal flux for each flux surface where data is defined
    - r/a: ArrayLike         [units] dimensionless
        1D array of normalised minor radius for each flux surface. This is needed for derivatives w.r.t rho (r/a)
    - Charge: Int      [units] elementary_charge
        Charge of each species
    - Mass: ArrayLike        [units] kg
        Mass of each species
    - Temperature: ArrayLike [units] eV
        1D array of the species temperature profile
    - Density: ArrayLike     [units] meter**-3
        1D array of the species density profile
    - Rotation: ArrayLike    [units] meter/second
        1D array of the species rotation profile

    Parameters
    ----------
    kinetics_type: str, default None
        Name of the kinetics input type, such as "SCENE", "JETTO", etc.
    **kwargs
        Used to pass in species data.
    """

    def __init__(self, kinetics_type: str, **kwargs):
        self.kinetics_type = kinetics_type
        self.species_data = CleverDict(**kwargs)
        """``CleverDict`` containing kinetics info for each species. May include
        entries such as 'electron' and 'deuterium'"""

    @property
    def kinetics_type(self):
        """Stored reference of the last kinetics type. May be inferred"""
        return self._kinetics_type

    @kinetics_type.setter
    def kinetics_type(self, value):
        if value not in self.supported_file_types():
            raise ValueError(f"Kinetics type {value} is not currently supported.")
        self._kinetics_type = value

    @property
    def nspec(self):
        """Number of species"""
        return len(self.species_data)

    @property
    def species_names(self):
        """Names of each species"""
        return self.species_data.keys()

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Kinetics object

        Returns
        -------
        Copy of kinetics object
        """
        # Create new object without calling __init__
        new_kinetics = Kinetics.__new__(Kinetics)
        # Deep copy each member besides species_data
        for key, value in self.__dict__.items():
            if key != "species_data":
                setattr(new_kinetics, key, value)
        # Build new species_data dict and populate one element at a time
        # (Note: we're not deepcopying Species. Species should have a __deepcopy__)
        new_kinetics.species_data = CleverDict()
        for name, species in self.species_data.items():
            new_kinetics.species_data[name] = species
        return new_kinetics


def read_kinetics(
    path: PathLike, file_type: Optional[str] = None, **kwargs
) -> Kinetics:
    r"""A plain-function alternative to ``Kinetics.from_file``."""
    return Kinetics.from_file(path, file_type=file_type, **kwargs)


def supported_kinetics_types() -> List[str]:
    r"""A plain-function alternative to ``Kinetics.supported_file_types``."""
    return Kinetics.supported_file_types()
