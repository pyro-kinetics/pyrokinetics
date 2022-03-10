from abc import ABC, abstractmethod

from ..typing import PathLike
from ..factory import Factory
from ..local_geometry import LocalGeometry
from ..local_species import LocalSpecies
from ..numerics import Numerics


class GKInputWriter(ABC):
    """
    Base for classes that write input files for gyrokinetics codes.
    Takes a Numerics, LocalGeometry, and LocalSpecies to do so.
    """

    @abstractmethod
    def write(
        self,
        filename: PathLike,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
    ):
        pass

    def __call__(
        self,
        filename: PathLike,
        local_geometry: LocalGeometry,
        local_species: LocalSpecies,
        numerics: Numerics,
    ):
        self.write(self, filename, local_geometry, local_species, numerics)


gk_input_writers = Factory(BaseClass=GKInputWriter)
