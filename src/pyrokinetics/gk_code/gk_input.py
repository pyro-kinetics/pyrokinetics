import copy
from abc import abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import f90nml
import numpy as np

from ..constants import deuterium_mass, electron_mass, hydrogen_mass, tritium_mass
from ..file_utils import AbstractFileReader, ReadableFromFile
from ..local_geometry import LocalGeometry
from ..local_species import LocalSpecies
from ..normalisation import SimulationNormalisation as Normalisation
from ..numerics import Numerics
from ..typing import ArrayLike, PathLike

# Monkeypatch on f90nml Namelists to autoconvert numpy scalar arrays to their
# underlying types and drop units.

_f90repr_orig = f90nml.Namelist._f90repr


def _f90repr_patch(self, val):
    if hasattr(val, "units"):
        val = val.magnitude
    if hasattr(val, "tolist"):
        val = val.tolist()
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
        self._convention_dict = {}

        """A collection of raw inputs from a Fortran 90 namelist"""
        if filename is not None:
            self.read_from_file(filename)
            self._detect_normalisation()

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

    def _detect_normalisation(self):
        """
        Get relevant data from GK input file to be passed to _set_up_normalisation
        """
        pass

    def _set_up_normalisation(
        self,
        default_references: dict,
        gk_code: str,
        electron_density: float,
        electron_temperature: float,
        e_mass: float,
        electron_index: int,
        found_electron: bool,
        densities: ArrayLike,
        temperatures: ArrayLike,
        reference_density_index: ArrayLike,
        reference_temperature_index: ArrayLike,
        major_radius: float,
        rgeo_rmaj: float,
        minor_radius: float,
    ):
        r"""
        Method to determine the NormalisationConvention from a GK input file. If a
        non-default convention is found then self.norm_convention is updated to
        a bespoke normalisation (string) and self._convetion_dict is set with the
        relevant convention (by default is {})

        Sets
        _convention_dict
            Dictionary of reference species for the density, temperature
            and mass along with reference magnetic field and length. The
            electron temp, density and ratio of R_geometric/R_major is
            included where R_geometric corresponds to the R where Bref is.
            B0 means magnetic field at the centre of the local flux surface
            and Bgeo is the magnetic field at the centre of the last closed
            flux surface.
        norm_convention
            str with name of new convention to be created

        Parameters
        ----------
        default_references: dict
            Dictionary containing default reference values for the
        gk_code: str
            GK code
        electron_density: float
            Electron density from GK input
        electron_temperature: float
            Electron density from GK input
        e_mass: float
            Electron mass from GK input
        electron_index: int
            Index of electron in list of data
        found_electron: bool
            Flag on whether electron was found
        densities: ArrayLike
            List of species densities
        temperatures: ArrayLike
            List of species temperature
        reference_density_index: ArrayLike
            List of indices where the species has a density of 1.0
        reference_temperature_index: ArrayLike
            List of indices where the species has a temperature of 1.0
        major_radius: float
            Normalised major radius from GK input
        rgeo_rmaj: float
            Ratio of Geometric and flux surface major radius
        minor_radius: float
            Normalised minor radius from GK input

        """

        pyro_default_references = {
            "nref_species": "electron",
            "tref_species": "electron",
            "mref_species": "deuterium",
            "bref": "B0",
            "lref": "minor_radius",
            "ne": 1.0,
            "te": 1.0,
            "rgeo_rmaj": 1.0,
            "vref": "nrl",
            "rhoref": "pyro",
        }

        references = copy.copy(default_references)

        if not np.isclose(rgeo_rmaj, 1.0):
            references["rgeo_rmaj"] = rgeo_rmaj

        if not np.isclose(electron_density, 1.0):
            references["ne"] = electron_density

        if not np.isclose(electron_temperature, 1.0):
            references["te"] = electron_temperature

        if not found_electron:
            raise ValueError(
                f"{gk_code} currently requires an electron species in the input file. No species found with Z = -1"
            )

        if len(reference_temperature_index) == 0:
            error_list = [
                f"TEMP_{i_sp+1} = {temp}" for i_sp, temp in enumerate(temperatures)
            ]
            raise ValueError(
                f"Cannot find any species with temperature = 1.0. Found {error_list}"
            )

        if len(reference_density_index) == 0:
            error_list = [
                f"DENS_{i_sp+1} = {dens}" for i_sp, dens in enumerate(densities)
            ]
            raise ValueError(
                f"Cannot find any species with density = 1.0. Found {error_list}"
            )

        me_md = (electron_mass / deuterium_mass).m
        me_mh = (electron_mass / hydrogen_mass).m
        me_mt = (electron_mass / tritium_mass).m

        if np.isclose(e_mass, 1.0):
            references["mref_species"] = "electron"
        elif np.isclose(e_mass, me_mh, rtol=0.1):
            references["mref_species"] = "hydrogen"
        elif np.isclose(e_mass, me_md, rtol=0.1):
            references["mref_species"] = "deuterium"
        elif np.isclose(e_mass, me_mt, rtol=0.1):
            references["mref_species"] = "tritium"

        else:
            raise ValueError(
                f"Cannot determine reference mass when electron_mass / reference_mass = {e_mass}. Only "
                f"electron/hydrogen/deuterium/tritium are supported"
            )

        if electron_index in reference_density_index:
            references["nref_species"] = "electron"
        else:
            for i_sp in reference_density_index:
                if np.isclose(densities[i_sp], 1.0):
                    references["nref_species"] = references["mref_species"]

        if references["nref_species"] is None:
            raise ValueError(
                f"Cannot determine reference density species as ne = {electron_density} and no species "
                f"found with DENS = 1 and MASS = 1"
            )

        if electron_index in reference_temperature_index:
            references["tref_species"] = "electron"
        else:
            for i_sp in reference_temperature_index:
                if np.isclose(temperatures[i_sp], 1.0):
                    references["tref_species"] = references["mref_species"]

        if references["tref_species"] is None:
            raise ValueError(
                f"Cannot determine reference temperature species as Te = {electron_temperature} as no "
                f"species found with TEMP = 1 and MASS = 1"
            )

        if np.isclose(major_radius, 1.0):
            references["lref"] = "major_radius"
        elif np.isclose(minor_radius, 1.0):
            references["lref"] = "minor_radius"
        else:
            raise ValueError(
                f"Can't determine reference length as normalised major_radius = {major_radius} and normalised minor radius = {minor_radius}"
            )

        if not np.isclose(rgeo_rmaj, 1.0):
            references["bref"] = "Bgeo"

        if references == pyro_default_references:
            self.norm_convention = "pyrokinetics"
        elif references != default_references:
            self.norm_convention = f"{gk_code}_bespoke"
            self._convention_dict = references


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
    gk_input._detect_normalisation()
    return gk_input
