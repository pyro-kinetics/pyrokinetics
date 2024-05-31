from collections import Counter
from pathlib import Path
from typing import Union

from typing_extensions import Self

from .local_geometry import LocalGeometry
from .local_species import LocalSpecies
from .normalisation import ConstNormalisation
from .numerics import Numerics
from .typing import PathLike

__all__ = ["LocalGKSimulation"]


class LocalGKSimulation:

    _geometry: LocalGeometry
    _species: LocalSpecies
    _numerics: Numerics
    _norms: ConstNormalisation

    _RUN_NAMES = Counter()

    def __init__(
        self,
        geometry: LocalGeometry,
        species: LocalSpecies,
        numerics: Numerics,
        convention: str,
        name: str = "pyro",
    ) -> None:
        """Describes a gyrokinetics simulation in a self-consistent manner.

        Gyrokinetics simulations are typically defined in terms of normalised
        units. This means that modifying one aspect of a simulation can
        necessitate modifications elsewhere to maintain consistency. For
        example, modifying the minor radius of the flux surface may change the
        reference length against which many other quanitities are defined, such
        as spatial derivatives.

        This class is used to ensure that modifications to one aspect of a
        simulation, such as the geometry or species under study, are reflected
        in the others. It achieves this by generating a new system of reference
        units at each modification, and converting all components to match.
        Instances should be considered immutable, and each modification should
        generate a new instance.

        When initialising, the input geometry, species, and numerics should
        either not have units, or have physics units. Simulation units such as
        ``lref_major_radius`` will result in an exception being raised.
        If you wish to build a new ``LocalGKSimulation``, TODO TODO TODO

        Parameters
        ----------
        geometry:
            Describes the parameterisation of the simulation's flux surface.
        species:
            Includes information such as mass, charge, and temperature for each
            species.
        numerics:
            Describes non-physical aspects of the simulation, such as grid
            dimensions.
        convention:
            The system of normalised units shared by all components of the
            simulation.
        """
        self.name = name

        # Get normalisation reference terms
        # TODO B0, major_radius, etc...

        # Create set of normalised units
        self._norms = ConstNormalisation(
            name=self._unique_name(name),
            convention=convention,
        )

        # Apply units to each component
        self._geometry = geometry.with_norms(self._norms)
        self._species = species.with_norms(self._norms)
        self._numerics = numerics.with_norms(self._norms)

    def with_geometry(geometry: LocalGeometry) -> Self:
        # TODO Create new LocalGkSimulation
        pass

    @classmethod
    def _unique_name(cls, name: Union[str, PathLike]) -> str:
        """Return a unqiuely numbered run name from `name`"""
        # name might be a Path, in which case just use the filename
        # (without extension)

        name = getattr(Path(name), "stem", name)
        name = "".join([ch for ch in name if ch.isalpha() or ch.isdigit() or ch == "_"])

        new_name = f"{name}{cls._RUN_NAMES[name]:06}"
        cls._RUN_NAMES[name] += 1
        return new_name
