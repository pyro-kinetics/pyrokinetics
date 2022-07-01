import copy
import f90nml
import numpy as np
from abc import abstractmethod
from typing import Optional, Any, Dict, List
from pathlib import Path

from ..typing import PathLike
from ..readers import Reader, create_reader_factory
from ..local_geometry import LocalGeometry
from ..local_species import LocalSpecies
from ..numerics import Numerics


class GKInput(Reader):
    """
    Base for classes that store gyrokinetics code input files in a dict-like format.
    They faciliate translation between input files on disk, and  Numerics,
    LocalGeometry, and LocalSpecies objects.

    Attributes
    ----------
    data (f90nml.Namelist): A collection of raw inputs from a Fortran 90 namelist.
    """

    def __init__(self, filename: Optional[PathLike] = None):
        self.data = None
        if filename is not None:
            self.read(filename)

    @abstractmethod
    def read(self, filename: PathLike) -> Dict[str, Any]:
        """
        Reads in GK input file to store as internal dictionary.
        Sets self.data and also returns a dict

        Default version assumes a Fortran90 namelist
        """
        self.data = f90nml.read(filename)
        return self.data.todict()

    @abstractmethod
    def read_str(self, input_string: str) -> Dict[str, Any]:
        """
        Reads in GK input file as a string, stores as internal dictionary.
        Sets self.data and also returns a dict

        Default version assumes a Fortran90 namelist
        """
        self.data = f90nml.reads(input_string)
        return self.data.todict()

    @classmethod
    def from_str(cls, input_string: str):
        gk = cls()
        gk.read_str(input_string)
        return gk

    @abstractmethod
    def write(self, filename: PathLike, float_format: str = ""):
        """
        Writes self.data to an input file

        Default version assumes a Fortran90 namelist
        """
        # Create directories if they don't exist already
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Create Fortran namelist and write
        nml = f90nml.Namelist(self.data)
        nml.float_format = float_format
        nml.write(filename, force=True)

    @abstractmethod
    def verify(self, filename):
        """
        Ensure file is valid for a given GK input type.
        Reads file, but does not perform processing.
        """
        pass

    @classmethod
    def verify_expected_keys(cls, filename: PathLike, keys: List[str]) -> bool:
        """
        Checks that the expected keys are present at the top level of self.data.
        Results True if all are present, otherwise returns False.
        """
        # Create new class to read, prevents overwriting self.data
        data = cls().read(filename)
        return np.all(np.isin(keys, list(data)))

    @abstractmethod
    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        template_file: Optional[PathLike] = None,
        **kwargs,
    ):
        """
        Build self.data from geometry/species/numerics objects.
        If self.data does not exist prior to calling this, populates defaults
        using template_file.
        """
        pass

    @abstractmethod
    def is_nonlinear(self) -> bool:
        """
        Return true if the GKCode is nonlinear, otherwise return False
        """
        pass

    def is_linear(self) -> bool:
        return not self.is_nonlinear()

    @abstractmethod
    def add_flags(self, flags) -> None:
        """
        Add extra flags to a GK code input file

        Default version assumes a Fortran90 namelist
        """
        for key, parameter in flags.items():
            if key not in self.data:
                self.data[key] = dict()
            for param, val in parameter.items():
                self.data[key][param] = val

    @abstractmethod
    def get_local_geometry(self) -> LocalGeometry:
        """
        get local geometry object from self.data
        """
        pass

    @abstractmethod
    def get_local_species(self) -> LocalSpecies:
        """
        get local species object from self.data
        """
        pass

    @abstractmethod
    def get_numerics(self) -> Numerics:
        """
        get numerical info (grid spacing, time steps, etc) from self.data
        """
        pass

    def __deepcopy__(self, memodict):
        """
        Allow copy.deepcopy. Works for derived classes.
        """
        # Create new object without calling __init__
        new_object = self.__class__()
        # Deep copy each member individually
        for key, value in self.__dict__.items():
            setattr(new_object, key, copy.deepcopy(value, memodict))
        return new_object


gk_inputs = create_reader_factory(BaseReader=GKInput)
