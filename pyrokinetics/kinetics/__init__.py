# Import KineticsReaders
from .kinetics_reader import KineticsReader, kinetics_readers
from .scene import KineticsReaderSCENE
from .jetto import KineticsReaderJETTO
from .transp import KineticsReaderTRANSP
from .pfile import KineticsReaderpFile

# Register each reader type with factory
kinetics_readers["SCENE"] = KineticsReaderSCENE
kinetics_readers["JETTO"] = KineticsReaderJETTO
kinetics_readers["TRANSP"] = KineticsReaderTRANSP
kinetics_readers["pFile"] = KineticsReaderpFile

__all__ = ["Kinetics", "KineticsReader"]

from typing import Optional
from ..typing import PathLike
from cleverdict import CleverDict
from pathlib import Path


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
    kinetics_file: str or Path
        Filename of a kinetics file to read from.
    kinetics_type: str, default None
        Name of the kinetics input type, such as "SCENE", "JETTO", etc. If left as None,
        this is inferred from the input file.
    eq_file: str or Path, default None
        Filename of a geqdsk file to read from.
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
        """Stored reference of the last file read"""

        if kinetics_type is not None:
            reader = kinetics_readers[kinetics_type]
            self.kinetics_type = kinetics_type
        else:
            # Infer kinetics type from file
            reader = kinetics_readers[kinetics_file]
            self.kinetics_type = reader.file_type

        self.species_data = CleverDict(reader(kinetics_file, **kwargs))
        """``CleverDict`` containing kinetics info for each species. May include
        entries such as 'electron' and 'deuterium'"""

    @property
    def supported_kinetics_types(self):
        """List of all supported kinetics input types"""
        return [*kinetics_readers]

    @property
    def kinetics_type(self):
        """Stored reference of the last kinetics type. May be inferred"""
        return self._kinetics_type

    @kinetics_type.setter
    def kinetics_type(self, value):
        if value not in self.supported_kinetics_types:
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
