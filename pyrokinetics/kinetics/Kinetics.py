from typing import Optional
from ..typing import PathLike
from cleverdict import CleverDict
from pathlib import Path

from .KineticsReader import kinetics_readers


class Kinetics:
    """
    Contains all the kinetic data in the form of Species objects.
    Data can be accessed via `kinetics.species_data`, which is a CleverDict with each
    key being a species name. For example, electron data can be accessed via a call
    to `kinetics.species_data["electron"]` or `kinetics.species_data.electron`.

    Each Species is provided with:

    - psi_n
    - r/a
    - Temperature
    - Density
    - Rotation

    Contains the attributes:

    - supported_kinetics_types: A list of all supported kinetics input types.
    - kinetics_files: Stored reference of the last file read
    - kinetics_type: Stored reference of the last kinetics type. May be inferred.
    - nspec: Number of species
    - species_data: CleverDict containing kinetics info for each species. May include\
        entries such as 'electron' and 'deuterium'.
    - species_names: Names of each species.


    Parameters
    ----------
    kinetics_file: str or Path
        Filename of a kinetics file to read from.
    kinetics_type: str, default None
        Name of the kinetics input type, such as "SCENE", "JETTO", etc. If left as None,
        this is inferred from the input file.
    **kwargs
        Extra arguments to be passed to the reader function. Not used by
        all readers, so only include this if necessary.
    """

    def __init__(
        self,
        kinetics_file: PathLike,
        kinetics_type: Optional[str] = None,
        **kwargs,
    ):
        self.kinetics_file = Path(kinetics_file)

        if kinetics_type is not None:
            reader = kinetics_readers[kinetics_type]
            self.kinetics_type = kinetics_type
        else:
            # Infer kinetics type from file
            reader = kinetics_readers[kinetics_file]
            self.kinetics_type = reader.file_type

        self.species_data = CleverDict(reader(kinetics_file, **kwargs))

    @property
    def supported_kinetics_types(self):
        return [*kinetics_readers]

    @property
    def kinetics_type(self):
        return self._kinetics_type

    @kinetics_type.setter
    def kinetics_type(self, value):
        if value not in self.supported_kinetics_types:
            raise ValueError(f"Kinetics type {value} is not currently supported.")
        self._kinetics_type = value

    @property
    def nspec(self):
        return len(self.species_data)

    @property
    def species_names(self):
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
