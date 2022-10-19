"""Classes for working with different conventions for normalisation.

Each piece of software that Pyrokinetics works with may use its own
convention for normalising physical quantities, and each individual
run or simulation may have different physical reference values. We
want to be able to convert normalised quantities between these
different conventions and simulations.

For example, GS2 normalises lengths to the minor radius, while GENE
uses the major radius. We can convert lengths between the two codes if
we know either the aspect ratio, or both the minor and major radius in
physical units (for example, from a particular machine or equilibrium
code). Both codes normalise speeds to the thermal velocity, except
GS2's definition includes an additional factor of the square root of
two; this means we can always convert normalised speeds between the
two codes simply by multipling or dividing by ``sqrt(2)`` as
appropriate.

This module aims to make this process of conversion simpler by
packaging up all the units and conventions together. Then, for
example, we could convert a growth rate from GS2 to GENE's
normalisation like so::

    growth_rate_gene = growth_rate_gs2.to(norms.gene)

where ``norms`` is a `SimulationNormalisation` instance.

We make a distinction between "simulation units" and "physical units":
simulation units are named something like ``lref_minor_radius``, are
always valid for a particular code, have a fictional dimension such as
``[lref]``, and usually need some extra information to convert them to
a different unit. Physical units, on the other hand, are associated
with a particular simulation and named something like
``lref_minor_radius_run1234``, have a real reference value, for
example ``1.5 metres``, and can be converted to other physical units
without extra information.

Importantly, for a given convention, we can convert between its
simulation and physical units. For example, a length of one in GS2
simulation units (that is, without a physical reference value) is
always equivalent to one GS2 physical length unit. This is equivalent
to defining the reference length for the simulation.

We define a "convention" to be a set of simulation units. These are:

- ``bref``: magnetic field
- ``lref``: length
- ``mref``: mass
- ``nref``: density
- ``qref``: charge
- ``tref``: temperature
- ``vref``: velocity

For example, the GS2 convention has an ``lref`` of
``lref_minor_radius``, and a ``vref`` of ``vref_most_probable``; while
GENE has ``lref_major_radius`` and ``vref_nrl`` respectively.

A unique `SimulationNormalisation` is created for each `Pyro`
instance, and contains a ``dict`` of `ConventionNormalisation`
instances. Each of these conventions contains a full set of units,
that is ``lref``, ``bref``, and so on. The `SimulationNormalisation`
has a default convention whose units can be accessed directly.

Initially, the convention's units refer to the simulation units for
that particular convention, but can be set to physical units::


    >>> norm = SimulationNormalisation("run1234")
    # We can access the units either through a particular convention:
    >>> norm.gs2.lref
    <Unit('lref_minor_radius')>
    # Or through `norm` directly for the default convention
    >>> norm.lref
    <Unit('lref_minor_radius')>
    # Providing a physical reference value changes simulation units
    # to physical units
    >>> norm.set_lref(minor_radius=1.5)
    >>> norm.lref
    <Unit('lref_minor_radius_run1234')>

We use [pint](https://pint.readthedocs.io) (with some local
modifications) for the units. Read their docs to understand how to
work with units generally.

The modifications here are to enable converting units to a
convention. To convert growth rates between GS2 and GENE we could do::

    growth_rate_gene = growth_rate_gs2.to(norms.gene.vref / norms.gene.lref)

Or more succinctly::

    growth_rate_gene = growth_rate_gs2.to(norms.gene)

which converts all (simulation or physical) units to GENE's
normalisation convention.

"""


from contextlib import contextmanager
import copy
from typing import Optional, Dict

import pint

from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometry


def _create_unit_registry() -> pint.UnitRegistry:
    """Create a default pint.UnitRegistry with some common features we need"""

    @contextmanager
    def as_system(self, system):
        """Temporarily change the current system of units"""
        old_system = self.default_system

        if system is None:
            pass
        elif isinstance(system, str):
            self.default_system = system
        else:
            self.default_system = system._system.name
        yield
        self.default_system = old_system

    pint.UnitRegistry.as_system = as_system

    ureg = pint.UnitRegistry()
    ureg._on_redefinition = "ignore"

    class PyroQuantity(ureg.Quantity):
        """Specialisation of `pint.UnitRegistry.Quantity` that expands
        some methods to be aware of pyrokinetics normalisation objects.

        Note that we need to define this class after creating ``ureg``
        so we can inherit from its internal ``Quantity`` class.
        """

        def to_base_units(self, system: Optional[str] = None):
            """Convert Quantity to base units, possibly in a different system"""
            with self._REGISTRY.as_system(system):
                return super().to_base_units()

        def to(self, other=None, *contexts, **ctx_kwargs):
            """Return Quantity rescaled to other units or normalisation"""

            # Converting to a normalisation implies converting to its base units
            if isinstance(other, (ConventionNormalisation, SimulationNormalisation)):
                with self._REGISTRY.context(other.context, *contexts, **ctx_kwargs):
                    return self.to_base_units(other)

            return super().to(other, *contexts, **ctx_kwargs)

    ureg.Quantity = PyroQuantity

    # Enable the Boltzmann context by default so we can always convert
    # eV to Kelvin
    ureg.enable_contexts("boltzmann")

    # IMAS normalises to the actual deuterium mass, so lets add that
    # as a constant
    ureg.define("deuterium_mass = 3.3435837724e-27 kg")

    # We can immediately define reference masses in physical units.
    # WARNING: This might need refactoring to use a [mref] dimension
    # if we start having other possible reference masses
    ureg.define("mref_deuterium = deuterium_mass")
    ureg.define("mref_electron = electron_mass")

    # For each normalisation unit, we create a unique dimension for
    # that unit and convention
    ureg.define("bref_B0 = [bref]")
    ureg.define("lref_minor_radius = [lref]")
    ureg.define("nref_electron = [nref]")
    ureg.define("tref_electron = [tref]")
    ureg.define("vref_nrl = [vref] = ([tref] / [mref])**(0.5)")
    ureg.define("beta_ref_ee_B0 = [beta_ref]")

    # vrefs are related by constant, so we can always define this one
    ureg.define("vref_most_probable = (2**0.5) * vref_nrl")

    # Now we define the "other" normalisation units that require more
    # information, such as bunit_over_B0 or the aspect_ratio
    ureg.define("bref_Bunit = NaN bref_B0")
    ureg.define("lref_major_radius = NaN lref_minor_radius")
    ureg.define("nref_deuterium = NaN nref_electron")
    ureg.define("tref_deuterium = NaN tref_electron")

    # Too many combinations of beta units, this almost certainly won't
    # scale, so just do the only one we know is used for now
    ureg.define("beta_ref_ee_Bunit = NaN beta_ref_ee_B0")

    return ureg


ureg = _create_unit_registry()
"""Default unit registry"""


REFERENCE_CONVENTIONS = {
    "lref": [ureg.lref_major_radius, ureg.lref_minor_radius],
    # For discussion of different v_thermal conventions, see:
    # https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.speeds.thermal_speed.html#thermal-speed-notes
    "vref": [ureg.vref_nrl, ureg.vref_most_probable],
    "bref": [ureg.bref_B0, ureg.bref_Bunit],
    # TODO: handle main_ion convention
    "mref": {"deuterium": ureg.mref_deuterium, "electron": ureg.mref_electron},
    "tref": {"deuterium": ureg.tref_deuterium, "electron": ureg.tref_electron},
    "nref": {"deuterium": ureg.nref_deuterium, "electron": ureg.nref_electron},
}


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

    def __init__(
        self,
        name: str,
        tref_species: str = "electron",
        nref_species: str = "electron",
        mref_species: str = "deuterium",
        vref: ureg.Unit = ureg.vref_nrl,
        lref: ureg.Unit = ureg.lref_minor_radius,
        bref: ureg.Unit = ureg.bref_B0,
    ):
        self.name = name

        if bref not in REFERENCE_CONVENTIONS["bref"]:
            raise ValueError(
                f"Unexpected bref: {bref} (valid options: {REFERENCE_CONVENTIONS['bref']}"
            )
        self.bref = bref

        if lref not in REFERENCE_CONVENTIONS["lref"]:
            raise ValueError(
                f"Unexpected lref: {lref} (valid options: {REFERENCE_CONVENTIONS['lref']}"
            )
        self.lref = lref

        if tref_species not in REFERENCE_CONVENTIONS["tref"]:
            raise ValueError(
                f"Unexpected tref: {tref_species} (valid options: {REFERENCE_CONVENTIONS['tref']}"
            )
        self.tref = REFERENCE_CONVENTIONS["tref"][tref_species]
        self.tref_species = tref_species

        if nref_species not in REFERENCE_CONVENTIONS["nref"]:
            raise ValueError(
                f"Unexpected nref: {nref_species} (valid options: {REFERENCE_CONVENTIONS['nref']}"
            )
        self.nref = REFERENCE_CONVENTIONS["nref"][nref_species]
        self.nref_species = nref_species

        if mref_species not in REFERENCE_CONVENTIONS["mref"]:
            raise ValueError(
                f"Unexpected mref: {mref_species} (valid options: {REFERENCE_CONVENTIONS['mref']}"
            )
        self.mref = REFERENCE_CONVENTIONS["mref"][mref_species]
        self.mref_species = mref_species

        if vref not in REFERENCE_CONVENTIONS["vref"]:
            raise ValueError(
                f"Unexpected vref: {vref} (valid options: {REFERENCE_CONVENTIONS['vref']}"
            )
        self.vref = vref

        # Construct name of beta_ref dimension
        bref_type = str(bref).split("_")[1]
        beta_ref_name = f"beta_ref_{nref_species[0]}{tref_species[0]}_{bref_type}"
        self.beta_ref = getattr(ureg, beta_ref_name)

        self.qref = "elementary_charge"


NORMALISATION_CONVENTIONS = {
    "pyrokinetics": Convention("pyrokinetics"),
    "cgyro": Convention("cgyro", bref=ureg.bref_Bunit),
    "gs2": Convention("gs2", vref=ureg.vref_most_probable),
    "gene": Convention("gene", lref=ureg.lref_major_radius),
    "imas": Convention("imas", vref=ureg.vref_most_probable),
}
"""Particular normalisation conventions"""


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
        # Physical context to convert simulations without physical
        # reference quantities
        self.context = pint.Context(name)

        # Create instances of each convention we know about
        self._conventions: Dict[str, ConventionNormalisation] = {
            name: ConventionNormalisation(convention, self)
            for name, convention in NORMALISATION_CONVENTIONS.items()
        }
        self.default_convention = convention

        # Gives us a nice attribute access for each convention, e.g. norms.gs2
        for name, convention in self._conventions.items():
            setattr(self, name, convention)

        if geometry:
            self.set_bref(geometry)
            self.set_lref(geometry)
        if kinetics:
            self.set_kinetic_references(kinetics, psi_n)

    def __deepcopy__(self, memodict):
        """Copy this instance."""

        # We have to be careful here, we don't want to copy
        # everything, because we need to make sure that units all use
        # the same global unit registry, or pint will break

        # This seems like a really bad way of achieving this, but hey,
        # it works

        new_object = SimulationNormalisation("COPY")
        new_object.name = self.name
        new_object.units = self.units
        new_object._conventions = copy.deepcopy(self._conventions)
        new_object.default_convention = self.default_convention.name

        # This is not clever, should be in ConventionNormalisation.__deepcopy__?
        for name, convention in self._conventions.items():
            new_object._conventions[name]._registry = self.units
            new_object._conventions[name].run_name = convention.run_name
            new_object._conventions[name].context = convention.context
            new_object._conventions[name].bref = convention.bref
            new_object._conventions[name].lref = convention.lref
            new_object._conventions[name].mref = convention.mref
            new_object._conventions[name].nref = convention.nref
            new_object._conventions[name].tref = convention.tref
            new_object._conventions[name].vref = convention.vref

            new_object._conventions[name]._update_system()
            setattr(new_object, name, new_object._conventions[name])

        new_object._system = self._system
        new_object._update_references()

        return new_object

    @property
    def default_convention(self):
        """Change the current convention that the short names refer to"""
        return self._current_convention

    @default_convention.setter
    def default_convention(self, convention):
        self._current_convention = self._conventions[convention]
        self._update_references()

    def _update_references(self):
        """Update all the short names to the current convention's
        actual units"""

        # We've updated some units, which means things like vref need
        # recalculating, so delete what we can from the cache.

        # WARNING: this relies on private details of the unit registry
        for key in list(self.units._cache.root_units.keys()):
            if self.name in key:
                del self.units._cache.root_units[key]

        self.bref = self._current_convention.bref
        self.lref = self._current_convention.lref
        self.mref = self._current_convention.mref
        self.nref = self._current_convention.nref
        self.qref = self._current_convention.qref
        self.tref = self._current_convention.tref
        self.vref = self._current_convention.vref
        self._system = self._current_convention._system

    @property
    def beta(self):
        return self._current_convention.beta

    def set_bref(self, local_geometry: LocalGeometry):
        """Set the magnetic field reference values for all the
        conventions from the local geometry

        FIXME: Can we take just the values we want?"""

        # Simulation units
        self.context.redefine(f"bref_Bunit = {local_geometry.bunit_over_b0} bref_B0")

        # Physical units
        bref_B0_sim = f"bref_B0_{self.name}"
        bref_Bunit_sim = f"bref_Bunit_{self.name}"
        self.units.define(f"{bref_B0_sim} = {local_geometry.B0} tesla")
        bunit = local_geometry.B0 * local_geometry.bunit_over_b0
        self.units.define(f"{bref_Bunit_sim} = {bunit} tesla")

        self.units.define(f"beta_ref_ee_B0_{self.name} = 1 beta_ref_ee_B0")
        self.units.define(
            f"beta_ref_ee_Bunit_{self.name} = {local_geometry.bunit_over_b0}**2 beta_ref_ee_B0"
        )

        bref_B0_sim_unit = getattr(self.units, bref_B0_sim)
        self.context.add_transformation(
            "[bref]",
            bref_B0_sim,
            lambda ureg, x: x.to(ureg.bref_B0).m * bref_B0_sim_unit,
        )

        for convention in self._conventions.values():
            convention.set_bref()
        self._update_references()

    def set_lref(
        self,
        local_geometry: Optional[LocalGeometry] = None,
        minor_radius: Optional[float] = None,
        major_radius: Optional[float] = None,
    ):
        """Set the length reference values for all the conventions
        from the local geometry

        TODO: Input checking
        TODO: Error handling
        TODO: Units on inputs
        """

        if local_geometry:
            minor_radius = local_geometry.a_minor
            major_radius = local_geometry.Rmaj

        # FIXME: We could also set this from minor_radius * aspect_ratio
        major_radius = major_radius or 0.0

        # Simulation unit can be converted with this context
        aspect_ratio = major_radius / minor_radius
        self.context.redefine(f"lref_major_radius = {aspect_ratio} lref_minor_radius")

        # Physical units
        self.units.define(f"lref_minor_radius_{self.name} = {minor_radius} metres")
        self.units.define(f"lref_major_radius_{self.name} = {major_radius} metres")

        for convention in self._conventions.values():
            convention.set_lref()
        self._update_references()

        self.context.add_transformation(
            "[lref]",
            self.pyrokinetics.lref,
            lambda ureg, x: x.to(ureg.lref_minor_radius).m * self.pyrokinetics.lref,
        )

    def set_kinetic_references(self, kinetics: Kinetics, psi_n: float):
        """Set the temperature, density, and mass reference values for
        all the conventions"""

        # Define physical units for each possible reference species
        for species in REFERENCE_CONVENTIONS["tref"]:
            tref = kinetics.species_data[species].get_temp(psi_n)
            self.units.define(f"tref_{species}_{self.name} = {tref} eV")

        for species in REFERENCE_CONVENTIONS["nref"]:
            nref = kinetics.species_data[species].get_dens(psi_n)
            self.units.define(f"nref_{species}_{self.name} = {nref} m**-3")

        for species in REFERENCE_CONVENTIONS["mref"]:
            mref = kinetics.species_data[species].get_mass()
            self.units.define(f"mref_{species}_{self.name} = {mref} kg")

        # We can also define physical vref now
        self.units.define(
            f"vref_nrl_{self.name} = (tref_electron_{self.name} / mref_deuterium_{self.name})**(0.5)"
        )
        self.units.define(
            f"vref_most_probable_{self.name} = (2 ** 0.5) * vref_nrl_{self.name}"
        )

        # Update the individual convention normalisations
        for convention in self._conventions.values():
            convention.set_kinetic_references()
        self._update_references()

        # Transformations between simulation and physical units
        self.context.add_transformation(
            "[tref]",
            self.pyrokinetics.tref,
            lambda ureg, x: x.to(ureg.tref_electron).m * self.pyrokinetics.tref,
        )
        self.context.add_transformation(
            "[nref]",
            self.pyrokinetics.nref,
            lambda ureg, x: x.to(ureg.nref_electron).m * self.pyrokinetics.nref,
        )

        # Transformations for mixed units because pint can't handle
        # them automatically.

        # TODO: Can we fix the pint algorithm to do this automatically?
        pressure_sim = self.units.nref_electron * self.units.tref_electron
        pressure_physical = self.pyrokinetics.nref * self.pyrokinetics.tref
        self.context.add_transformation(
            "[nref] * [tref]",
            pressure_physical,
            lambda ureg, x: x.to(pressure_sim).m * pressure_physical,
        )

        self.context.add_transformation(
            "[vref]",
            self.pyrokinetics.vref,
            lambda ureg, x: x.to(ureg.vref_nrl).m * self.pyrokinetics.vref,
        )

        try:
            lref_physical = self.pyrokinetics.lref
        except pint.UndefinedUnitError:
            # FIXME: We've apparently not set lref yet, so we can't
            # define the following conversions properly, but we need
            # something for now
            lref_physical = self.units.lref_minor_radius

        self.context.add_transformation(
            "[vref] / [lref]",
            self.pyrokinetics.vref / lref_physical,
            lambda ureg, x: x.to(ureg.vref_nrl / ureg.lref_minor_radius).m
            * (self.pyrokinetics.vref / lref_physical),
        )
        self.context.add_transformation(
            "1 / [time]",
            "[vref] / [lref]",
            lambda ureg, x: x.to(self.pyrokinetics.vref / lref_physical).m
            * (ureg.vref_nrl / ureg.lref_minor_radius),
        )


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

    def __init__(
        self,
        convention: Convention,
        parent: SimulationNormalisation,
        definitions: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.convention = convention
        self.name = convention.name
        self.run_name = parent.name
        self.context = parent.context

        self._registry = parent.units
        self._system = parent.units.get_system(
            f"{self.convention.name}_{self.run_name}"
        )

        self.bref = convention.bref
        self.lref = convention.lref
        self.mref = convention.mref
        self.nref = convention.nref
        self.qref = convention.qref
        self.tref = convention.tref
        self.vref = convention.vref
        self.beta_ref = convention.beta_ref

        self._update_system()

    def _update_system(self):
        self._system.base_units = {
            # Physical units
            "tesla": {str(self.bref): 1.0},
            "meter": {str(self.lref): 1.0},
            "gram": {str(self.mref): 1.0},
            "kelvin": {str(self.tref): 1.0},
            "second": {str(self.lref): 1.0, str(self.vref): -1.0},
            "ampere": {
                str(self.qref): 1.0,
                str(self.lref): -1.0,
                str(self.vref): 1.0,
            },
            # Simulation units
            "bref_B0": {str(self.bref): 1.0},
            "lref_minor_radius": {str(self.lref): 1.0},
            "mref_deuterium": {str(self.mref): 1.0},
            "nref_electron": {str(self.nref): 1.0},
            "tref_electron": {str(self.tref): 1.0},
            "vref_nrl": {str(self.vref): 1.0},
            "beta_ref_ee_B0": {str(self.beta_ref): 1.0},
        }

    @property
    def beta(self):
        try:
            return (
                2 * self._registry.mu0 * self.nref * self.tref / (self.bref**2)
            ).to_base_units(self) * self.beta_ref
        except pint.DimensionalityError:
            # We get a dimensionality error if we've not set
            # nref/tref/bref, so we can't compute the beta.
            return 0.0 * self._registry.dimensionless

    def set_bref(self):
        self.bref = getattr(self._registry, f"{self.convention.bref}_{self.run_name}")
        self.beta_ref = getattr(
            self._registry, f"{self.convention.beta_ref}_{self.run_name}"
        )
        self._update_system()

    def set_lref(self):
        self.lref = getattr(self._registry, f"{self.convention.lref}_{self.run_name}")
        self._update_system()

    def set_kinetic_references(self):
        self.tref = getattr(self._registry, f"{self.convention.tref}_{self.run_name}")
        self.mref = getattr(self._registry, f"{self.convention.mref}_{self.run_name}")
        self.nref = getattr(self._registry, f"{self.convention.nref}_{self.run_name}")
        self.vref = getattr(self._registry, f"{self.convention.vref}_{self.run_name}")
        self._update_system()
