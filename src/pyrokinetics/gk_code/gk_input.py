import copy
from abc import abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import f90nml
import numpy as np

from ..file_utils import AbstractFileReader, ReadableFromFile
from ..local_geometry import LocalGeometry
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..numerics import Numerics
from ..typing import PathLike

# Monkeypatch on f90nml Namelists to autoconvert numpy scalar arrays to their
# underlying types and drop units.

_f90repr_orig = f90nml.Namelist._f90repr


def _f90repr_patch(self, val):
    if hasattr(val, "tolist"):
        val = val.tolist()
    if hasattr(val, "units"):
        val = val.magnitude
    return _f90repr_orig(self, val)


f90nml.Namelist._f90repr = _f90repr_patch


class GKInput(AbstractFileReader, ReadableFromFile):
    """
    Base for classes that store gyrokinetics code input files in a dict-like format.
    They faciliate translation between input files on disk, and `Numerics`,
    `LocalGeometry`, and `LocalSpecies` objects.

    `GKInput` differs from `GKOutput`, `Equilibrium` and `Kinetics` in that it is
    both the 'reader' and the 'readable'. Each subclass should define the methods
    ``read_from_file`` and ``verify_file_type``. ``read_from_file`` should populate
    ``self.data`` and return this information as a dict.
    """

    norm_convention: str = "pyrokinetics"
    """`Convention` used for normalising this code's quantities"""

    def __init__(self, filename: Optional[PathLike] = None):
        self.data: Optional[f90nml.Namelist] = None
        """A collection of raw inputs from a Fortran 90 namelist"""
        if filename is not None:
            self.read_from_file(filename)

    @abstractmethod
    def read_from_file(self, filename: PathLike) -> Dict[str, Any]:
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

    @abstractmethod
    def read_dict(self, gk_dict: dict) -> Dict[str, Any]:
        """
        Reads in dictionary equivalent of a GK input file and stores as internal dictionary.
        Sets self.data and also returns a dict

        Default version assumes a dict
        """
        self.data = gk_dict
        return self.data

    @classmethod
    def from_str(cls, input_string: str):
        gk = cls()
        gk.read_str(input_string)
        return gk

    @abstractmethod
    def write(
        self,
        filename: PathLike,
        float_format: str = "",
        local_norm: Optional[Normalisation] = None,
    ):
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
    def verify_file_type(self, filename: PathLike) -> None:
        """
        Ensure file is valid for a given GK input type.
        Reads file, but does not perform processing.
        """
        pass

    @classmethod
    def verify_expected_keys(cls, filename: PathLike, keys: List[str]) -> None:
        """
        Checks that the expected keys are present at the top level of self.data.
        Results True if all are present, otherwise returns False.
        """
        # Create new class to read, prevents overwriting self.data
        try:
            data = cls().read_from_file(filename)
        except Exception as exc:
            raise RuntimeError(
                f"Couldn't read {cls.file_type} file. Is the format correct?"
            ) from exc
        if not np.all(np.isin(keys, list(data))):
            key_str = "', '".join(keys)
            msg = dedent(
                f"""
                Unable to verify {filename} as a {cls.file_type} file. The following
                keys are required: '{key_str}'
                """
            )
            raise ValueError(msg.replace("\n", " "))

    @abstractmethod
    def set(
        self,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
        local_norm: Optional[Normalisation] = None,
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


def supported_gk_input_types() -> List[str]:
    """
    Returns a list of all registered `GKInput` file types. These file types are
    readable by ``GKInput.from_file``.
    """
    return GKInput.supported_file_types()


def read_gk_input(path: PathLike, file_type: Optional[str] = None, **kwargs) -> GKInput:
    r"""
    Create and instantiate a `GKInput` subclass.

    This function differs from similar functions such as `read_equilibrium` or
    `read_gk_output`, as `GKInput` is both a reader and readable. This means we
    shouldn't discard the reader class. As a result, this function does not use
    `GKInput.from_file`.
    """
    gk_input = GKInput._factory(file_type if file_type is not None else path)
    gk_input.read_from_file(path, **kwargs)
    return gk_input
