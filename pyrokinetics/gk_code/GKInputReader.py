import f90nml
from abc import abstractmethod
from typing import Optional

from ..typing import PathLike
from ..readers import Reader, create_reader_factory
from ..local_geometry import LocalGeometry
from ..local_species import LocalSpecies
from ..numerics import Numerics


class GKInputReader(Reader):
    """
    Base for classes that read gyrokinetics codes' input files and
    produce Numerics, LocalGeometry, and LocalSpecies objects.
    """

    def __init__(self, filename: Optional[PathLike] = None):
        if filename is not None:
            self.read(filename)

    @abstractmethod
    def read(self, filename: PathLike) -> None:
        """
        Reads in GK input file to store as internal dictionary
        """
        self.data = f90nml.read(filename).todict()

    @abstractmethod
    def verify(self, filename):
        """
        Ensure file is valid for a given GK input type.
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

        """
        pass

    @abstractmethod
    def get_local_geometry(self) -> LocalGeometry:
        """
        Load local geometry object from a GK code input file
        """
        pass

    @abstractmethod
    def get_local_species(self) -> LocalSpecies:
        """
        Load local species object from a GK code input file
        """
        pass

    @abstractmethod
    def get_numerics(self) -> Numerics:
        """
        Gather numerical info (grid spacing, time steps, etc)
        """
        pass


gk_input_readers = create_reader_factory(BaseReader=GKInputReader)
