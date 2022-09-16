"""Proxy objects for pint units

The problem we're trying to solve is that each unique simulation run
(i.e. `Pyro` object) will have its own set of reference values to
normalise that run to. Separately, each code has its own convention
for normalising its quantities, including pyrokinetics, as well as
other external conventions, such as GKDB/IMAS. When we have multiple
simulations, we (potentially) need the full Cartesian product of
``(simulation reference values) x (normalisation conventions)`` in
order to, for example, get all of the simulations into a single,
comparable normalisation.

To do this, we use the `pint` library, which allows us to create
unique units for each combination of reference value and normalisation
convention for a given simulation. We then wrap this up in a set of
proxy objects which gives us nice names for these units.


Normalisation for single simulation =>
Conventions =>
Unique units

"""


from contextlib import contextmanager
import copy
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

import pint

from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometry


@dataclass
class Convention:
    """A description of a normalisation convention, including what
    species reference values use, whether the velocity includes a
    ``sqrt(2)`` factor, what length scales are normalised to, and so
    on.

    TODO: Do we need to specifiy "kind" of deuterium mass? Actual mass vs 2*m_p?

    Attributes
    ----------
    tref_species:
        The species to normalise temperatures to
    nref_species:
        The species to normalise densities to
    mref_species:
        The species to normalise masses to
    vref_multiplier:
        Velocity multiplier
    lref_type:
        What to normalise length scales to
    bref_type:
        Magnetic field normalisation. Must be either ``B0`` or ``Bunit``

    """

    name: str
    tref_species: str = "electron"
    nref_species: str = "electron"
    mref_species: str = "deuterium"
    vref_multiplier: float = 1.0
    lref_type: str = "minor_radius"
    bref_type: str = "B0"


NORMALISATION_CONVENTIONS = {
    "pyrokinetics": Convention("pyrokinetics"),
    "cgyro": Convention("cgyro", bref_type="Bunit"),
    "gs2": Convention("gs2", vref_multiplier=np.sqrt(2)),
    "gene": Convention("gene", lref_type="major_radius"),
    "gkdb": Convention("gkdb", vref_multiplier=np.sqrt(2)),
    "imas": Convention("imas", vref_multiplier=np.sqrt(2)),
}
"""Particular normalisation conventions"""


def _create_unit_registry() -> pint.UnitRegistry:
    """Create a default pint.UnitRegistry with some common features we need"""

    @contextmanager
    def as_system(self, system):
        """Temporarily change the current system of units"""
        old_system = self.default_system

        if isinstance(system, str):
            self.default_system = system
        else:
            self.default_system = system._system.name
        yield
        self.default_system = old_system

    pint.UnitRegistry.as_system = as_system

    ureg = pint.UnitRegistry()

    class PyroQuantity(ureg.Quantity):
        """Specialisation of `pint.UnitRegistry.Quantity` that expands
        some methods to be aware of pyrokinetics normalisation objects.

        Note that we need to define this class after creating ``ureg``
        so we can inherit from its internal ``Quantity`` class.
        """

        def to_base_units(self, system: Optional[str] = None):
            """Convert Quantity to base units, possibly in a different system"""
            if system is None:
                return super().to_base_units()
            with self._REGISTRY.as_system(system):
                return super().to_base_units()

        def to(self, other=None, *contexts, **ctx_kwargs):
            if isinstance(other, (ConventionNormalisation, SimulationNormalisation)):
                return self.to_base_units(other)
            return super().to(other, *contexts, **ctx_kwargs)

    ureg.Quantity = PyroQuantity

    # Enable the Boltzmann context by default so we can always convert
    # eV to Kelvin
    ureg.enable_contexts("boltzmann")

    # IMAS normalises to the actual deuterium mass, so lets add that
    # as a constant
    ureg.define("deuterium_mass = 3.3435837724e-27 kg")

    return ureg


ureg = _create_unit_registry()
"""Default unit registry"""


class SimulationNormalisation:
    """Holds the normalisations for a given simulation for all the
    known conventions.

    Has a current convention which sets what the short names refer to.

    Examples
    --------

        >>> norm = SimulationNormalisation("run001")
        >>> norm.set_bref(local_geometry)
        >>> norm.set_lref(local_geometry)
        >>> norm.set_kinetic_references(kinetics)
        >>> norm.lref      # Current convention's lref
        1 <Unit('lref_pyrokinetics_run001')>
        >>> norm.gs2.lref  # Specific convention's lref
        1 <Unit('lref_gs2_run001')>
        # Change the current default convention
        >>> norm.default_convention = "gkdb"
        >>> norm.gkdb.lref
        1 <Unit('lref_gkdb_run001')>

    """

    def __init__(
        self,
        name: str,
        convention: str = "pyrokinetics",
        registry: pint.UnitRegistry = ureg,
        geometry: Optional[LocalGeometry] = None,
        kinetics: Optional[Kinetics] = None,
        psi_n: Optional[float] = None,
    ):
        self.units = ureg
        self.name = name

        self._conventions: Dict[str, ConventionNormalisation] = {
            name: ConventionNormalisation(self.name, convention, self.units)
            for name, convention in NORMALISATION_CONVENTIONS.items()
        }

        self.default_convention = convention

        if geometry:
            self.set_bref(geometry)
            self.set_lref(geometry)
        if kinetics:
            self.set_kinetic_references(kinetics, psi_n)

    def __getattr__(self, item):
        try:
            return self._conventions[item]
        except KeyError:
            raise AttributeError(name=item, obj=self)

    def __deepcopy__(self, memodict):
        """Don't actually copy

        We don't want to deep copy, because we want to make use of a
        shared global unit registry so we can convert between
        different simulations, for example, in PyroScan

        """
        new_object = self.__class__(self.name)
        new_object.units = self.units
        new_object._conventions = copy.copy(self._conventions)
        # for convention in self._conventions.keys():
        #     self._conventions[convention]._registry = self.units
        new_object._update_references()

        return new_object

    @property
    def default_convention(self):
        """Change the current convention that the short names refer to"""
        return self._system

    @default_convention.setter
    def default_convention(self, convention):
        self._system = self._conventions[convention]
        self._update_references()

    def _update_references(self):
        """Update all the short names to the current convention's
        actual units"""

        # Note that this relies on private details of the unit registry
        for key in list(self.units._cache.root_units.keys()):
            if self.name in key:
                del self.units._cache.root_units[key]

        self.bref = self._system.bref
        self.lref = self._system.lref
        self.mref = self._system.mref
        self.nref = self._system.nref
        self.qref = self._system.qref
        self.tref = self._system.tref
        self.vref = self._system.vref
        self.rhoref = self._system.rhoref

    @property
    def beta(self):
        return self._system.beta

    def set_bref(self, local_geometry: LocalGeometry):
        """Set the magnetic field reference values for all the
        conventions from the local geometry

        FIXME: Can we take just the values we want?"""
        for convention in self._conventions.values():
            convention.set_bref(local_geometry)
        self._update_references()

    def set_lref(self, local_geometry: LocalGeometry):
        """Set the length reference values for all the conventions
        from the local geometry

        FIXME: Can we take just the values we want?"""
        for convention in self._conventions.values():
            convention.set_lref(local_geometry)
        self._update_references()

    def set_kinetic_references(self, kinetics: Kinetics, psi_n: float):
        """Set the temperature, density, and mass reference values for
        all the conventions"""

        for convention in self._conventions.values():
            convention.set_kinetic_references(kinetics, psi_n)
        self._update_references()


class ConventionNormalisation:
    """A concrete set of reference values/normalisations.

    You should call `ConventionNormalistion.set_lref_bref` and then
    `ConventionNormalistion.set_kinetic_references` (in that order)
    before attempting to use most of these units

    Parameters
    ----------
    run_name:
        Name of the specific simulation run
    convention:
        Object describing how particular reference values should be set
    registry:
        The pint registry to add these units to
    definitions:
        Dictionary of definitions for each reference value. If not
        given, the default set will be used

    """

    REF_DEFS = {
        "deuterium_mass": {"def": "3.3435837724e-27 kg"},
        "bref": {"def": "nan tesla", "base": "tesla"},
        "lref": {"def": "nan metres", "base": "meter"},
        "mref": {"def": "deuterium_mass", "base": "gram"},
        "nref": {"def": "nan m**-3"},
        "qref": {"def": "elementary_charge"},
        "tref": {"def": "nan eV", "base": "kelvin"},
        "vref": {"def": "(tref / mref)**(0.5)"},
        "rhoref": {"def": "mref * vref / qref / bref"},
    }

    def __init__(
        self,
        run_name: str,
        convention: Convention,
        registry: pint.UnitRegistry,
        definitions: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.convention = convention
        self.name = f"{self.convention.name}_{run_name}"

        self._registry = registry
        self._system = registry.get_system(self.name)

        self.definitions = definitions or self.REF_DEFS

        for unit, definition in self.definitions.items():
            convention_unit = f"{unit}_{self.name}"

            unit_def = definition["def"]
            for unit_name in list(self.REF_DEFS.keys()):
                unit_def = unit_def.replace(unit_name, f"{unit_name}_{self.name}")

            if unit == "vref":
                unit_def = f"{self.convention.vref_multiplier} * {unit_def}"

            self._registry.define(f"{convention_unit} = {unit_def}")

            if "base" in definition:
                self._system.base_units[definition["base"]] = {convention_unit: 1.0}

        self._system.base_units["second"] = {
            f"lref_{self.name}": 1.0,
            f"vref_{self.name}": -1.0,
        }
        self._system.base_units["ampere"] = {
            f"qref_{self.name}": 1.0,
            f"lref_{self.name}": -1.0,
            f"vref_{self.name}": 1.0,
        }

        # getattr rather than []-indexing as latter returns a quantity
        # rather than a unit (??)
        self.bref = getattr(self._registry, f"bref_{self.name}")
        self.lref = getattr(self._registry, f"lref_{self.name}")
        self.mref = getattr(self._registry, f"mref_{self.name}")
        self.nref = getattr(self._registry, f"nref_{self.name}")
        self.qref = getattr(self._registry, f"qref_{self.name}")
        self.tref = getattr(self._registry, f"tref_{self.name}")
        self.vref = getattr(self._registry, f"vref_{self.name}")
        self.rhoref = getattr(self._registry, f"rhoref_{self.name}")

    @property
    def beta(self):
        beta = (
            2 * self._registry.mu0 * self.nref * self.tref / (self.bref**2)
        ).to_base_units(self)

        if np.isnan(beta.magnitude):
            return 0.0 * self._registry.dimensionless
        return beta

    def set_bref(self, local_geometry: LocalGeometry):

        # These are lambdas so we don't have to evaluate them just
        # yet, as local_geometry might not actually have the
        # quantities we want
        BREF_TYPES = {
            "B0": lambda: local_geometry.B0,
            "Bunit": lambda: local_geometry.B0 * local_geometry.bunit_over_b0,
        }

        bref_type = self.convention.bref_type
        if bref_type not in BREF_TYPES:
            raise ValueError(
                f"Unrecognised bref_type: got '{bref_type}', expected one of {list(BREF_TYPES.keys())}"
            )

        bref = BREF_TYPES[bref_type]()
        self._registry.define(f"bref_{self.name} = {bref} tesla")

    def set_lref(self, local_geometry: LocalGeometry):

        # These are lambdas so we don't have to evaluate them just
        # yet, as local_geometry might not actually have the
        # quantities we want
        LREF_TYPES = {
            "minor_radius": lambda: local_geometry.a_minor,
            "major_radius": lambda: local_geometry.Rmaj,
        }

        lref_type = self.convention.lref_type
        if lref_type not in LREF_TYPES:
            raise ValueError(
                f"Unrecognised lref_type: got '{lref_type}', expected one of {list(LREF_TYPES.keys())}"
            )

        lref = LREF_TYPES[lref_type]()
        self._registry.define(f"lref_{self.name} = {lref} metres")

    def set_kinetic_references(self, kinetics: Kinetics, psi_n: float):
        tref = kinetics.species_data[self.convention.tref_species].get_temp(psi_n)
        nref = kinetics.species_data[self.convention.nref_species].get_dens(psi_n)
        mref = kinetics.species_data[self.convention.mref_species].get_mass()

        self._registry.define(f"tref_{self.name} = {tref} eV")
        self._registry.define(f"nref_{self.name} = {nref} m**-3")
        self._registry.define(f"mref_{self.name} = {mref} kg")
