r"""Classes for working with different conventions for normalisation.

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
- ``lref``: Equilibrium length
- ``mref``: mass
- ``nref``: density
- ``qref``: charge
- ``tref``: temperature
- ``vref``: velocity
- ``rhoref``: gyroradius

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

.. warning::
    ``bref`` is not a fundamental dimension, so converting magnetic
    fields needs to be done directly:

    .. code-block::

       # Wrong! Units will be mref * vref / qref / lref
       B0_cgyro = B0_gene.to(norms.cgyro)
       # Right! Units will be bref
       B0_cgyro = B0_gene.to(norms.cgyro.bref)

``beta``
~~~~~~~~

The magnetic :math:`\beta_N` is a dimensionless quantity defined by:

.. math::

    \beta_N = \frac{2 \mu_0 n_{ref} T_{ref}}{B_{ref}^2}

When we have all of ``nref, tref, bref`` we can compute ``beta_ref``,
and we only need to track which convention it was defined in. We do so
by giving it units in another arbitrary dimension, ``[beta_ref]``,
with distinct units for each convention. We also don't need to keep
track of simulation and physical units separately, as these are always
directly convertable. ``[beta_ref]`` is essentially just another name
for "percent", but gives us a mechanism to convert between
normalisations.

"""

import copy
from typing import Dict, Optional, Union
from typing_extensions import Self

import pint

from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometry
from pyrokinetics.units import Normalisation, PyroNormalisationError, PyroQuantity, ureg

REFERENCE_CONVENTIONS = {
    "lref": [ureg.lref_major_radius, ureg.lref_minor_radius],
    # For discussion of different v_thermal conventions, see:
    # https://docs.plasmapy.org/en/stable/api/plasmapy.formulary.speeds.thermal_speed.html#thermal-speed-notes
    "vref": [ureg.vref_nrl, ureg.vref_most_probable],
    "rhoref": [ureg.rhoref_pyro, ureg.rhoref_unit, ureg.rhoref_gs2],
    "bref": [ureg.bref_B0, ureg.bref_Bunit],
    # TODO: handle main_ion convention
    "mref": {
        "deuterium": ureg.mref_deuterium,
        "electron": ureg.mref_electron,
        "hydrogen": ureg.mref_hydrogen,
        "tritium": ureg.mref_tritium,
    },
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
    rhoref_multiplier:
       gyroradius multiplier
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
        rhoref: ureg.Unit = ureg.rhoref_pyro,
        lref: ureg.Unit = ureg.lref_minor_radius,
        bref: ureg.Unit = ureg.bref_B0,
        betaref: ureg.Unit = None,
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

        if rhoref not in REFERENCE_CONVENTIONS["rhoref"]:
            raise ValueError(
                f"Unexpected rhoref: {rhoref} (valid options: {REFERENCE_CONVENTIONS['rhoref']}"
            )
        self.rhoref = rhoref

        if betaref is None:
            # Construct name of beta_ref dimension
            bref_type = str(bref).split("_")[1]
            beta_ref_name = f"beta_ref_{nref_species[0]}{tref_species[0]}_{bref_type}"
            self.beta_ref = getattr(ureg, beta_ref_name)
        else:
            self.beta_ref = betaref

        self.qref = ureg.elementary_charge

    def __repr__(self):
        return (
            f"Convention(\n"
            f"    name = {self.name},\n"
            f"    tref_species = {self.tref_species},\n"
            f"    nref_species = {self.nref_species},\n"
            f"    mref_species = {self.mref_species},\n"
            f"    vref = {self.vref},\n"
            f"    rhoref = {self.rhoref},\n"
            f"    lref = {self.lref},\n"
            f"    bref = {self.bref},\n"
            f"    betaref = {self.beta_ref}\n"
            f")"
        )


NORMALISATION_CONVENTIONS = {
    "pyrokinetics": Convention("pyrokinetics"),
    "cgyro": Convention("cgyro", bref=ureg.bref_Bunit, rhoref=ureg.rhoref_unit),
    "gs2": Convention("gs2", vref=ureg.vref_most_probable, rhoref=ureg.rhoref_gs2),
    "stella": Convention(
        "stella", vref=ureg.vref_most_probable, rhoref=ureg.rhoref_gs2
    ),
    "gene": Convention("gene", lref=ureg.lref_major_radius, rhoref=ureg.rhoref_pyro),
    "gkw": Convention(
        "gkw",
        lref=ureg.lref_major_radius,
        vref=ureg.vref_most_probable,
        rhoref=ureg.rhoref_gs2,
    ),
    "imas": Convention(
        "imas",
        vref=ureg.vref_most_probable,
        rhoref=ureg.rhoref_gs2,
        lref=ureg.lref_major_radius,
    ),
    "tglf": Convention("tglf", bref=ureg.bref_Bunit, rhoref=ureg.rhoref_unit),
}
"""Particular normalisation conventions"""


class SimulationNormalisation(Normalisation):
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
        self.units = registry
        self.name = name
        # Physical context to convert simulations without physical
        # reference quantities
        self.context = pint.Context(self.name)

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
            self.set_ref_ratios(geometry)
        if kinetics:
            self.set_kinetic_references(kinetics, psi_n)
        if geometry and kinetics:
            self.set_rhoref(geometry)

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
        new_object._conventions = copy.deepcopy(self._conventions, memodict)
        new_object.default_convention = self.default_convention.name

        for name in self._conventions:
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

    def add_convention_normalisation(self, name=None, convention_dict=None):
        """

        Parameters
        ----------
        name : str
            Name of new convention to add
        convention_dict : dict
            Dictionary of refererence species and locations

        Returns
        -------

        """
        # Create instances of each convention we know about

        te = convention_dict["te"]
        ne = convention_dict["ne"]
        rgeo_rmaj = convention_dict["rgeo_rmaj"]

        beta_ref_name = f"beta_ref_{convention_dict['nref_species'][0]}{convention_dict['tref_species'][0]}_{convention_dict['bref']}"

        if beta_ref_name not in self.units:
            self.define(
                f"{beta_ref_name} = {ne} * {te} / {rgeo_rmaj ** 2} beta_ref_ee_B0",
                units=True,
            )

        if rgeo_rmaj != 1.0:
            self.define(f"bref_Bgeo = {rgeo_rmaj}**-1 bref_B0", units=True)
            REFERENCE_CONVENTIONS["bref"].append(self.units.bref_Bgeo)

        if ne != 1.0:
            self.define(
                f"nref_{convention_dict['nref_species']} = {ne ** -1} nref_electron",
                units=True,
            )

        if te != 1.0:
            self.define(
                f"tref_{convention_dict['tref_species']} = {te ** -1} tref_electron",
                units=True,
            )

        md = (
            (
                1.0
                * self.units.mref_deuterium
                / getattr(self.units, f"mref_{convention_dict['mref_species']}")
            )
            .to_base_units()
            .m
        )

        vref_multiplier = (md / te) ** 0.5
        rho_ref_multiplier = vref_multiplier * rgeo_rmaj / md

        if te != 1.0 or md != 1.0:
            vref_base = f"vref_{convention_dict['vref']}"
            vref_new = f"{convention_dict['vref']}_{convention_dict['tref_species'][0]}_{convention_dict['mref_species'][0]}"
            self.define(
                f"vref_{vref_new} = {vref_multiplier} {vref_base}",
                units=True,
            )
            REFERENCE_CONVENTIONS["vref"].append(
                getattr(self.units, f"vref_{vref_new}")
            )
            convention_dict["vref"] = vref_new

        if te != 1.0 or md != 1.0 or rgeo_rmaj != 1.0:
            self.define(
                f"rhoref_custom = {rho_ref_multiplier} rhoref_{convention_dict['rhoref']}",
                units=True,
            )
            REFERENCE_CONVENTIONS["rhoref"].append(self.units.rhoref_custom)

            convention_dict["rhoref"] = "custom"

        convention_dict["rhoref"] = getattr(
            self.units, f"rhoref_{convention_dict['rhoref']}"
        )
        convention_dict["bref"] = getattr(self.units, f"bref_{convention_dict['bref']}")
        convention_dict["lref"] = getattr(self.units, f"lref_{convention_dict['lref']}")
        convention_dict["vref"] = getattr(self.units, f"vref_{convention_dict['vref']}")

        ref_keys = [
            "bref",
            "lref",
            "vref",
            "tref_species",
            "mref_species",
            "nref_species",
            "betaref",
            "rhoref",
        ]
        convention_dict = {k: v for k, v in convention_dict.items() if k in ref_keys}

        convention = Convention(name=name, **convention_dict)

        self._conventions[name] = ConventionNormalisation(convention, self)
        setattr(self, name, self._conventions[name])

        self.units._build_cache()

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
        self.rhoref = self._current_convention.rhoref

        self._system = self._current_convention._system

    def define(self, definition: str, context=False, units=False):
        r"""Defines a new units and adds it to the Context"""

        if units:
            self.units.define(definition)

        if context:
            self.context.redefine(definition)

    @property
    def beta(self):
        r"""The magnetic :math:`\beta_N` is a dimensionless quantity defined by:

        .. math::
            \beta_N = \frac{2 \mu_0 n_{ref} T_{ref}}{B_{ref}^2}

        """
        return self._current_convention.beta

    @property
    def beta_ref(self):
        r"""The magnetic :math:`\beta_N` is a dimensionless quantity defined by:

        .. math::
            \beta_N = \frac{2 \mu_0 n_{ref} T_{ref}}{B_{ref}^2}

        """
        return self._current_convention.beta_ref

    def set_bref(self, local_geometry: LocalGeometry):
        """Set the magnetic field reference values for all the
        conventions from the local geometry

        FIXME: Can we take just the values we want?"""
        # Simulation units
        self.define(
            f"bref_Bunit = {local_geometry.bunit_over_b0.m} bref_B0", context=True
        )

        # Physical units
        bref_B0_sim = f"bref_B0_{self.name}"
        bref_Bunit_sim = f"bref_Bunit_{self.name}"
        self.define(f"{bref_B0_sim} = {local_geometry.B0}", units=True)
        bunit = local_geometry.B0 * local_geometry.bunit_over_b0
        self.define(f"{bref_Bunit_sim} = {bunit}", units=True)

        self.define(
            f"beta_ref_ee_Bunit = {local_geometry.bunit_over_b0.m}**2 beta_ref_ee_B0",
            context=True,
        )

        bref_B0_sim_unit = getattr(self.units, bref_B0_sim)
        self.context.add_transformation(
            "[bref]",
            bref_B0_sim_unit.dimensionality,
            lambda ureg, x: x.to(ureg.bref_B0).m * bref_B0_sim_unit,
        )

        for convention in self._conventions.values():
            convention.set_bref()
        self._update_references()

        self.units._build_cache()

    def set_lref(
        self,
        local_geometry: Optional[LocalGeometry] = None,
        minor_radius: Optional[PyroQuantity] = None,
        major_radius: Optional[PyroQuantity] = None,
    ):
        """Set the length reference values for all the conventions
        from the local geometry

        * TODO: Input checking
        * TODO: Error handling
        * TODO: Units on inputs
        """

        if local_geometry:
            minor_radius = local_geometry.a_minor
            aspect_ratio = local_geometry.Rmaj / local_geometry.a_minor
        elif minor_radius and major_radius:
            aspect_ratio = (major_radius / minor_radius).to_base_units()
        else:
            aspect_ratio = 0.0 * self.units.dimensionless

        # Simulation unit can be converted with this context
        if minor_radius is not None and aspect_ratio is not None:
            major_radius = aspect_ratio * minor_radius
        else:
            major_radius = 0.0 * self.units.meter

        self.define(
            f"lref_major_radius = {aspect_ratio.m} lref_minor_radius", context=True
        )

        # Physical units
        if minor_radius is not None:
            self.define(f"lref_minor_radius_{self.name} = {minor_radius}", units=True)

        if major_radius is not None:
            self.define(f"lref_major_radius_{self.name} = {major_radius}", units=True)

        for convention in self._conventions.values():
            convention.set_lref()
        self._update_references()

        self.context.add_transformation(
            "[lref]",
            self.pyrokinetics.lref.dimensionality,
            lambda ureg, x: x.to(ureg.lref_minor_radius).m * self.pyrokinetics.lref,
        )

        self.units._build_cache()

    def set_rhoref(
        self,
        local_geometry: Optional[LocalGeometry] = None,
    ):
        """Set the gyroradius reference values for all the conventions
        from the local geometry and kinetics

        """
        if local_geometry:
            bunit_over_b0 = local_geometry.bunit_over_b0.m

        self.define(
            f"rhoref_pyro_{self.name} = {self.vref} / ({self.bref} / {self.mref} * qref)",
            units=True,
        )

        self.define(
            f"rhoref_gs2_{self.name} = (2 ** 0.5) * rhoref_pyro_{self.name}", units=True
        )

        self.define(
            f"rhoref_unit_{self.name} = {bunit_over_b0}**-1 * rhoref_pyro_{self.name}",
            units=True,
        )

        if "rhoref_custom" in self.units:
            self.define(
                f"rhoref_custom_{self.name} = rhoref_custom",
                units=True,
            )

        # Update the individual convention normalisations
        for convention in self._conventions.values():
            convention.set_rhoref()

        self._update_references()

        self.context.add_transformation(
            "[rhoref]",
            self.pyrokinetics.rhoref.dimensionality,
            lambda ureg, x: x.to(ureg.rhoref_pyro).m * self.pyrokinetics.rhoref,
        )

        self.units._build_cache()

    def set_ref_ratios(
        self,
        local_geometry: Optional[LocalGeometry] = None,
        aspect_ratio: Optional[float] = None,
    ):
        """Set the ratio of B0/Bunit and major_radius/minor_radius for normalised data

        * TODO: Input checking
        * TODO: Error handling
        * TODO: Units on inputs
        """

        # Simulation unit can be converted with this context
        if local_geometry:
            try:
                aspect_ratio = local_geometry.Rmaj.to(
                    self.pyrokinetics.lref, self.context
                ).m
                self.define(
                    f"lref_major_radius = {aspect_ratio} lref_minor_radius",
                    context=True,
                )
            except (PyroNormalisationError, pint.DimensionalityError):
                raise ValueError(
                    "Cannot determined ratio of R_major / a_minor. "
                    "Please set directly using"
                    " `pyro.norms.set_lref(aspect_ratio=aspect_ratio)`"
                )

            self.define(
                f"bref_Bunit = {local_geometry.bunit_over_b0.m} bref_B0", context=True
            )

            self.define(
                f"beta_ref_ee_Bunit = {local_geometry.bunit_over_b0.m}**2 beta_ref_ee_B0",
                context=True,
            )

            self.define(
                f"rhoref_unit ={local_geometry.bunit_over_b0.m}**-1 rhoref_pyro",
                context=True,
            )
        elif aspect_ratio:
            self.define(
                f"lref_major_radius = {aspect_ratio} lref_minor_radius", context=True
            )
        else:
            raise ValueError(
                "Need either LocalGeometry or aspect_ratio when setting reference ratios"
            )

        self._update_references()

        self.units._build_cache()

    def set_kinetic_references(self, kinetics: Kinetics, psi_n: float):
        """Set the temperature, density, and mass reference values for
        all the conventions"""

        # Define physical units for each possible reference species
        for species in REFERENCE_CONVENTIONS["tref"]:
            tref = kinetics.species_data[species].get_temp(psi_n)
            self.define(f"tref_{species}_{self.name} = {tref}", units=True)

        for species in REFERENCE_CONVENTIONS["nref"]:
            nref = kinetics.species_data[species].get_dens(psi_n)
            self.define(f"nref_{species}_{self.name} = {nref}", units=True)

        for species in REFERENCE_CONVENTIONS["mref"]:
            if species in kinetics.species_data:
                mref = kinetics.species_data[species].get_mass()
                self.define(f"mref_{species}_{self.name} = {mref}", units=True)

        # We can also define physical vref now
        self.define(
            f"vref_nrl_{self.name} = (tref_electron_{self.name} / mref_deuterium_{self.name})**(0.5)",
            units=True,
        )
        self.define(
            f"vref_most_probable_{self.name} = (2 ** 0.5) * vref_nrl_{self.name}",
            units=True,
        )

        # Update the individual convention normalisations
        for convention in self._conventions.values():
            convention.set_kinetic_references()
        self._update_references()

        # Transformations between simulation and physical units
        self.context.add_transformation(
            "[tref]",
            self.pyrokinetics.tref.dimensionality,
            lambda ureg, x: x.to(ureg.tref_electron).m * self.pyrokinetics.tref,
        )
        self.context.add_transformation(
            "[nref]",
            self.pyrokinetics.nref.dimensionality,
            lambda ureg, x: x.to(ureg.nref_electron).m * self.pyrokinetics.nref,
        )

        # Transformations for mixed units because pint can't handle
        # them automatically.
        self.context.add_transformation(
            "[vref]",
            self.pyrokinetics.vref.dimensionality,
            lambda ureg, x: x.to(ureg.vref_nrl).m * self.pyrokinetics.vref,
        )

        self.units._build_cache()

    def set_all_references(
        self,
        pyro,
        tref_electron=None,
        nref_electron=None,
        bref_B0=None,
        lref_minor_radius=None,
        lref_major_radius=None,
    ):
        self.define(f"tref_electron_{self.name} = {tref_electron}", units=True)
        self.define(f"nref_electron_{self.name} = {nref_electron}", units=True)

        self.define(f"mref_deuterium_{self.name} = mref_deuterium", units=True)

        if lref_minor_radius and lref_major_radius:
            if (
                lref_major_radius
                != pyro.local_geometry.Rmaj.to(self.pyrokinetics.lref, self.context).m
                * lref_minor_radius
            ):
                raise ValueError(
                    "Specified major radius and minor radius do not match, please check the data"
                )
        elif lref_minor_radius:
            lref_major_radius = lref_minor_radius * pyro.local_geometry.Rmaj.to(
                self.pyrokinetics.lref, self.context
            )
        elif lref_major_radius:
            lref_minor_radius = lref_major_radius / pyro.local_geometry.Rmaj.to(
                self.gene.lref
            )

        self.define(f"lref_minor_radius_{self.name} = {lref_minor_radius}", units=True)
        self.define(f"lref_major_radius_{self.name} = {lref_major_radius}", units=True)

        # Physical units
        bunit = bref_B0 * pyro.local_geometry.bunit_over_b0.m
        self.define(f"bref_B0_{self.name} = {bref_B0}", units=True)
        self.define(f"bref_Bunit_{self.name} = {bunit}", units=True)

        self.define(
            f"beta_ref_ee_Bunit = {pyro.local_geometry.bunit_over_b0.m}**2 beta_ref_ee_B0",
            context=True,
        )

        self.define(
            f"vref_nrl_{self.name} = (tref_electron_{self.name} / mref_deuterium_{self.name})**(0.5)",
            units=True,
        )
        self.define(
            f"vref_most_probable_{self.name} = (2 ** 0.5) * vref_nrl_{self.name}",
            units=True,
        )

        self.define(
            f"rhoref_pyro_{self.name} = vref_nrl_{self.name} / (bref_B0_{self.name} / mref_deuterium_{self.name} * qref)",
            units=True,
        )

        self.define(
            f"rhoref_gs2_{self.name} = (2 ** 0.5) * rhoref_pyro_{self.name}", units=True
        )

        self.define(
            f"rhoref_unit_{self.name} = {pyro.local_geometry.bunit_over_b0.m}**-1 * rhoref_pyro_{self.name}",
            units=True,
        )

        # Update the individual convention normalisations
        for convention in self._conventions.values():
            convention.set_all_references()
        self._update_references()

        # Transformations between simulation and physical units
        self.context.add_transformation(
            "[tref]",
            self.pyrokinetics.tref.dimensionality,
            lambda ureg, x: x.to(ureg.tref_electron).m * self.pyrokinetics.tref,
        )

        self.context.add_transformation(
            "[lref]",
            self.pyrokinetics.lref.dimensionality,
            lambda ureg, x: x.to(ureg.lref_minor_radius).m * self.pyrokinetics.lref,
        )

        self.context.add_transformation(
            "[nref]",
            self.pyrokinetics.nref.dimensionality,
            lambda ureg, x: x.to(ureg.nref_electron).m * self.pyrokinetics.nref,
        )

        self.context.add_transformation(
            "[bref]",
            self.pyrokinetics.bref.dimensionality,
            lambda ureg, x: x.to(ureg.bref_B0).m * self.pyrokinetics.bref,
        )

        self.context.add_transformation(
            "[vref]",
            self.pyrokinetics.vref.dimensionality,
            lambda ureg, x: x.to(ureg.vref_nrl).m * self.pyrokinetics.vref,
        )

        self.context.add_transformation(
            "[rhoref]",
            self.pyrokinetics.rhoref.dimensionality,
            lambda ureg, x: x.to(ureg.rhoref_pyro).m * self.pyrokinetics.rhoref,
        )

        self.units._build_cache()


class ConventionNormalisation(Normalisation):
    """A concrete set of reference values/normalisations.

    You should call `set_lref`, `set_bref` and then
    `set_kinetic_references` (in that order) before attempting to use
    most of these units

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
        self.rhoref = convention.rhoref

        self.beta_ref = convention.beta_ref

        self._update_system()

    def __deepcopy__(self, memodict):
        """Overrides deepcopy behaviour to perform regular copy of the Pint registry."""
        new_obj = object.__new__(type(self))
        for k, v in self.__dict__.items():
            if k == "_registry" or k == "_system":
                continue
            new_obj.__dict__[k] = copy.deepcopy(v, memodict)
        new_obj._registry = self._registry
        new_obj._system = self._registry.get_system(
            f"{self.convention.name}_{self.run_name}"
        )
        new_obj._update_system()
        return new_obj

    def __repr__(self):
        return (
            f"ConventionNormalisation(\n"
            f"    name = {self.name},\n"
            f"    tref = {self.tref},\n"
            f"    nref = {self.nref},\n"
            f"    mref = {self.mref},\n"
            f"    vref = {self.vref},\n"
            f"    rhoref = {self.rhoref},\n"
            f"    lref = {self.lref},\n"
            f"    bref = {self.bref},\n"
            f"    betaref = {self.beta_ref}\n"
            f")"
        )

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
            "rhoref_pyro": {str(self.rhoref): 1.0},
            "beta_ref_ee_B0": {str(self.beta_ref): 1.0},
        }

    @property
    def references(self):
        return {
            "bref": self.bref,
            "lref": self.lref,
            "mref": self.mref,
            "nref": self.nref,
            "qref": self.qref,
            "tref": self.tref,
            "vref": self.vref,
            "rhoref": self.rhoref,
        }

    @property
    def beta(self):
        """Returns the magnetic beta if all the reference quantites
        are set, otherwise zero

        """
        try:
            return (
                2 * self._registry.mu0 * self.nref * self.tref / (self.bref**2)
            ).to_base_units(self) * self.beta_ref
        except pint.DimensionalityError:
            # We get a dimensionality error if we've not set
            # nref/tref/bref, so we can't compute the beta.
            return 0.0 * self._registry.dimensionless

    def set_bref(self):
        """Set the reference magnetic field to the physical value"""
        self.bref = getattr(self._registry, f"{self.convention.bref}_{self.run_name}")
        self._update_system()

    def set_lref(self):
        """Set the reference length to the physical value"""
        self.lref = getattr(self._registry, f"{self.convention.lref}_{self.run_name}")
        self._update_system()

    def set_kinetic_references(self):
        """Set the reference temperature, density, mass, velocity to the physical value"""
        self.tref = getattr(self._registry, f"{self.convention.tref}_{self.run_name}")
        self.mref = getattr(self._registry, f"{self.convention.mref}_{self.run_name}")
        self.nref = getattr(self._registry, f"{self.convention.nref}_{self.run_name}")
        self.vref = getattr(self._registry, f"{self.convention.vref}_{self.run_name}")
        self._update_system()

    def set_rhoref(self):
        self.rhoref = getattr(
            self._registry, f"{self.convention.rhoref}_{self.run_name}"
        )

        self._update_system()

    def set_all_references(self):
        """Set reference value manually"""
        self.tref = getattr(self._registry, f"{self.convention.tref}_{self.run_name}")
        self.mref = getattr(self._registry, f"{self.convention.mref}_{self.run_name}")
        self.nref = getattr(self._registry, f"{self.convention.nref}_{self.run_name}")
        self.vref = getattr(self._registry, f"{self.convention.vref}_{self.run_name}")
        self.lref = getattr(self._registry, f"{self.convention.lref}_{self.run_name}")
        self.bref = getattr(self._registry, f"{self.convention.bref}_{self.run_name}")
        self.rhoref = getattr(
            self._registry, f"{self.convention.rhoref}_{self.run_name}"
        )
        self._update_system()


def convert_dict(data: Dict, norm: ConventionNormalisation) -> Dict:
    """Copy data into a new dict, converting any quantities to other normalisation"""

    new_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            if isinstance(value[0], norm._registry.Quantity):
                try:
                    value = [v.to(norm).magnitude for v in value]
                except (PyroNormalisationError, pint.DimensionalityError) as err:
                    raise ValueError(
                        f"Couldn't convert '{key}' ({value}) to {norm.name} normalisation. "
                        "This is probably because it did not contain physical reference values. "
                        "To fix this, please add a geometry and/or kinetic file to your "
                        "`Pyro` object."
                    ) from err

        if hasattr(value, "units"):
            try:
                value = value.to(norm).magnitude
            except (PyroNormalisationError, pint.DimensionalityError) as err:
                raise ValueError(
                    f"Couldn't convert '{key}' ({value}) to {norm.name} normalisation. "
                    "This is probably because it did not contain physical reference values. "
                    "To fix this, please add a geometry and/or kinetic file to your "
                    "`Pyro` object."
                ) from err

        new_data[key] = value

    return new_data


QuantityT = Union[float, PyroQuantity]


class ConstNormalisation(Normalisation):

    def __init__(
        self,
        name: str,
        B0: QuantityT,
        bunit_over_b0: QuantityT,
        minor_radius: QuantityT,
        aspect_ratio: QuantityT,
        electron_temperature: QuantityT,
        deuterium_temperature: QuantityT,
        electron_density: QuantityT,
        deuterium_density: QuantityT,
        electron_mass: QuantityT,
        hydrogen_mass: QuantityT,
        deuterium_mass: QuantityT,
        tritium_mass: QuantityT,
        convention: str = "pyrokinetics",
        registry: pint.UnitRegistry = ureg,
    ):
        """Holds the normalisations for a given simulation.

        Works similarly to :class:`SimulationNormalisation`, but rather than
        tracking normalisations for all known conventions, it only knows a
        single convention.

        Does not permit redefinition of units after initialisation.
        """
        self.name = name
        self.units = registry
        self.context = pint.Context(self.name)
        self.convention = NORMALISATION_CONVENTIONS[convention]

        self._set_magnetic_references(B0, bunit_over_b0)
        self._set_length_references(minor_radius, aspect_ratio)
        self._set_temperature_references(electron_temperature, deuterium_temperature)
        self._set_density_references(electron_density, deuterium_density)
        self._set_mass_references(
            electron_mass, hydrogen_mass, deuterium_mass, tritium_mass
        )
        self._set_velocity_references()
        self._set_charge_references()
        self._set_rho_references(bunit_over_b0)

        # TODO Do we need to further set up beta_ref here?
        self.beta_ref = self.convention.beta_ref

        # Finish setting up
        self._update_system()

    def __repr__(self):
        return (
            f"ConstNormalisation(\n"
            f"    name = {self.name},\n"
            f"    tref = {self.tref},\n"
            f"    nref = {self.nref},\n"
            f"    mref = {self.mref},\n"
            f"    vref = {self.vref},\n"
            f"    rhoref = {self.rhoref},\n"
            f"    lref = {self.lref},\n"
            f"    bref = {self.bref},\n"
            f"    betaref = {self.beta_ref}\n"
            f")"
        )

    def _has_physical_units(self, val: QuantityT) -> bool:
        return self.name in str(self.units.Quantity(val))

    @property
    def beta(self):
        r"""The magnetic :math:`\beta_N` is a dimensionless quantity defined by:

        .. math::
            \beta_N = \frac{2 \mu_0 n_{ref} T_{ref}}{B_{ref}^2}

        """

        # TODO Not sure if this is correct!
        #      beta_ref is set from convention, not from any real units

        try:
            return (
                2 * self.units.mu0 * self.nref * self.tref / (self.bref**2)
            ).to_base_units(self) * self.beta_ref
        except pint.DimensionalityError:
            # We get a dimensionality error if we've not set
            # nref/tref/bref, so we can't compute the beta.
            return 0.0 * self.units.dimensionless

    @property
    def references(self):
        return {
            "bref": self.bref,
            "lref": self.lref,
            "mref": self.mref,
            "nref": self.nref,
            "qref": self.qref,
            "tref": self.tref,
            "vref": self.vref,
            "rhoref": self.rhoref,
        }

    def _set_magnetic_references(self, B0: QuantityT, bunit_over_b0: QuantityT) -> None:
        B0 = self.units.Quantity(B0)
        bunit_over_b0 = self.units.Quantity(bunit_over_b0).to(self.units.dimensionless)

        # Simulation unit ratios
        self.context.redefine(f"bref_Bunit = {bunit_over_b0.m} bref_B0")

        self.context.redefine(
            f"beta_ref_ee_Bunit = {bunit_over_b0.m}**2 beta_ref_ee_B0"
        )

        # Physical units
        try:
            # Raises DimensionalityError if not physical units
            B0 = B0.to(self.units.tesla)

            bref_B0_name = f"bref_B0_{self.name}"
            bref_Bunit_name = f"bref_Bunit_{self.name}"
            self.units.define(f"{bref_B0_name} = {B0}")
            bunit = B0 * bunit_over_b0
            self.units.define(f"{bref_Bunit_name} = {bunit}")

            bref_B0_unit = getattr(self.units, bref_B0_name)
            self.context.add_transformation(
                "[bref]",
                bref_B0_unit.dimensionality,
                lambda ureg, x: x.to(ureg.bref_B0).m * bref_B0_unit,
            )

            self.bref = getattr(self.units, f"{self.convention.bref}_{self.name}")
        except pint.DimensionalityError:
            self.bref = self.convention.bref

    def _set_length_references(
        self, minor_radius: QuantityT, aspect_ratio: QuantityT
    ) -> None:
        minor_radius = self.units.Quantity(minor_radius)
        aspect_ratio = self.units.Quantity(aspect_ratio).to(self.units.dimensionless)

        # Simulation unit ratio
        self.context.redefine(f"lref_major_radius = {aspect_ratio.m} lref_minor_radius")

        # Physical units
        try:
            # Raises DimensionalityError if not physical units
            minor_radius = minor_radius.to(self.units.m)

            major_radius_name = f"lref_major_radius_{self.name}"
            minor_radius_name = f"lref_minor_radius_{self.name}"
            self.units.define(f"{major_radius_name} = {minor_radius * aspect_ratio}")
            self.units.define(f"{minor_radius_name} = {minor_radius}")

            minor_radius_unit = getattr(self.units, minor_radius_name)
            self.context.add_transformation(
                "[lref]",
                minor_radius_unit.dimensionality,
                lambda ureg, x: x.to(ureg.lref_minor_radius).m * minor_radius_unit,
            )

            self.lref = getattr(self.units, f"{self.convention.lref}_{self.name}")
        except pint.DimensionalityError:
            self.lref = self.convention.lref

    def _set_temperature_references(
        self, electron_temp: QuantityT, deuterium_temp: QuantityT
    ) -> None:
        electron_temp = self.units.Quantity(electron_temp)
        deuterium_temp = self.units.Quantity(deuterium_temp)

        # Simulation unit ratio
        ratio = (deuterium_temp / electron_temp).to(self.units.dimensionless)
        self.context.redefine(f"tref_deuterium = {ratio.m} tref_electron")

        # Physical units
        try:
            electron_temp = electron_temp.to(self.units.kelvin)
            deuterium_temp = deuterium_temp.to(self.units.kelvin)

            electron_name = f"tref_electron_{self.name}"
            deuterium_name = f"tref_deuterium_{self.name}"
            self.units.define(f"{electron_name} = {electron_temp}")
            self.units.define(f"{deuterium_name} = {deuterium_temp}")

            tref_unit = getattr(self.units, electron_name)
            self.context.add_transformation(
                "[tref]",
                tref_unit.dimensionality,
                lambda ureg, x: x.to(ureg.tref_electron).m * tref_unit,
            )
            self.tref = getattr(self.units, f"{self.convention.tref}_{self.name}")
        except pint.DimensionalityError:
            self.tref = self.convention.tref

    def _set_density_references(
        self, electron_density: QuantityT, deuterium_density: QuantityT
    ) -> None:
        electron_density = self.units.Quantity(electron_density)
        deuterium_density = self.units.Quantity(deuterium_density)

        # Simulation unit ratio
        ratio = (deuterium_density / electron_density).to(self.units.dimensionless)
        self.context.redefine(f"nref_deuterium = {ratio.m} nref_electron")

        # Physical units
        try:
            electron_density = electron_density.to(self.units.meter**-3)
            deuterium_density = deuterium_density.to(self.units.meter**-3)

            electron_name = f"nref_electron_{self.name}"
            deuterium_name = f"nref_deuterium_{self.name}"
            self.units.define(f"{electron_name} = {electron_density}")
            self.units.define(f"{deuterium_name} = {deuterium_density}")

            nref_unit = getattr(self.units, electron_name)
            self.context.add_transformation(
                "[nref]",
                nref_unit.dimensionality,
                lambda ureg, x: x.to(ureg.nref_electron).m * nref_unit,
            )

            self.nref = getattr(self.units, f"{self.convention.nref}_{self.name}")
        except pint.DimensionalityError:
            self.nref = self.convention.nref

    def _set_mass_references(
        self,
        electron_mass: QuantityT,
        hydrogen_mass: QuantityT,
        deuterium_mass: QuantityT,
        tritium_mass: QuantityT,
    ) -> None:
        electron_mass = self.units.Quantity(electron_mass)
        hydrogen_mass = self.units.Quantity(hydrogen_mass)
        deuterium_mass = self.units.Quantity(deuterium_mass)
        tritium_mass = self.units.Quantity(tritium_mass)

        # Ratios between mass types already defined elsewhere

        try:
            electron_mass = electron_mass.to(self.units.gram)
            hydrogen_mass = hydrogen_mass.to(self.units.gram)
            deuterium_mass = deuterium_mass.to(self.units.gram)
            tritium_mass = tritium_mass.to(self.units.gram)

            electron_name = f"mref_electron_{self.name}"
            hydrogen_name = f"mref_hydrogen_{self.name}"
            deuterium_name = f"mref_deuterium_{self.name}"
            tritium_name = f"mref_tritium_{self.name}"
            self.units.define(f"{electron_name} = {electron_mass}")
            self.units.define(f"{hydrogen_name} = {hydrogen_mass}")
            self.units.define(f"{deuterium_name} = {deuterium_mass}")
            self.units.define(f"{tritium_name} = {tritium_mass}")

            self.mref = getattr(self.units, f"{self.convention.mref}_{self.name}")
        except pint.DimensionalityError:
            self.mref = self.convention.mref

    def _set_velocity_references(self) -> None:
        if self._has_physical_units(self.tref) and self._has_physical_units(self.mref):
            self.units.define(
                f"vref_nrl_{self.name} = (tref_electron_{self.name} "
                f"/ mref_deuterium_{self.name})**(0.5)"
            )
            self.units.define(
                f"vref_most_probable_{self.name} "
                f"= (2 ** 0.5) * vref_nrl_{self.name}"
            )

            vref_unit = getattr(self.units, f"vref_nrl_{self.name}")

            # Transformations for mixed units needed because pint can't handle
            # them automatically.
            self.context.add_transformation(
                "[vref]",
                vref_unit.dimensionality,
                lambda ureg, x: x.to(ureg.vref_nrl).m * vref_unit,
            )

            self.vref = getattr(self.units, f"{self.convention.vref}_{self.name}")
        else:
            self.vref = self.convention.vref

    def _set_charge_references(self) -> None:
        self.qref = self.convention.qref  # Assume this is the same in all cases

    def _set_rho_references(self, bunit_over_b0: QuantityT) -> None:
        bunit_over_b0 = self.units.Quantity(bunit_over_b0).to(self.units.dimensionless)

        # Simulation unit ratio
        self.context.redefine(f"rhoref_unit ={bunit_over_b0.m}**-1 rhoref_pyro")

        # Physical units
        if all(self._has_physical_units(x) for x in (self.vref, self.bref, self.mref)):
            self.units.define(
                f"rhoref_pyro_{self.name} = {self.vref} "
                f"/ ({self.bref} / {self.mref} * qref)"
            )

            self.units.define(
                f"rhoref_gs2_{self.name} = (2 ** 0.5) * rhoref_pyro_{self.name}"
            )

            self.units.define(
                f"rhoref_unit_{self.name} = {bunit_over_b0.m}**-1 "
                f"* rhoref_pyro_{self.name}",
            )

            if "rhoref_custom" in self.units:
                # TODO This needs fixing
                self.units.define(
                    f"rhoref_custom_{self.name} = rhoref_custom",
                )

            rhoref_unit = getattr(self.units, f"rhoref_pyro_{self.name}")
            self.context.add_transformation(
                "[rhoref]",
                rhoref_unit.dimensionality,
                lambda ureg, x: x.to(ureg.rhoref_pyro).m * rhoref_unit,
            )
            self.rhoref = getattr(self.units, f"{self.convention.rhoref}_{self.name}")
        else:
            self.rhoref = self.convention.rhoref

    def _update_system(self) -> None:
        self._system = self.units.get_system(f"{self.convention.name}_{self.name}")
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
            "rhoref_pyro": {str(self.rhoref): 1.0},
            "beta_ref_ee_B0": {str(self.beta_ref): 1.0},
        }

    def _mutate(
        self,
        name: Optional[str] = None,
        convention: Optional[str] = None,
        B0: Optional[QuantityT] = None,
        bunit_over_b0: Optional[QuantityT] = None,
        minor_radius: Optional[QuantityT] = None,
        aspect_ratio: Optional[QuantityT] = None,
        electron_temp: Optional[QuantityT] = None,
        deuterium_temp: Optional[QuantityT] = None,
        electron_dens: Optional[QuantityT] = None,
        deuterium_dens: Optional[QuantityT] = None,
        electron_mass: Optional[QuantityT] = None,
        hydrogen_mass: Optional[QuantityT] = None,
        deuterium_mass: Optional[QuantityT] = None,
        tritium_mass: Optional[QuantityT] = None,
    ) -> Self:
        """Creates a new instance with modified parameters.

        Not recommended for direct use, as the wrong choice of optional
        arguments results in failure to build a new instance. This is used as
        the base function for multiple other methods that generate modified
        variants of a given normalisation, and those should be used instead.

        If a new name is not supplied, all parameters except ``convention``
        are ignored. This produces a shallow copy that references the same
        units system, though perhaps in a different convention.

        If a new name is supplied, this generates a new normalisation with
        a new set of physical units, if any have been set. Any parameters
        that are not set to ``None`` will overwrite those of ``self``.

        TODO complete docs

        Parameters
        ----------
        name
            If provided, the new convention will be renamed, and all physical
            units will be redefined.
        convention
            The choice of normalisation convention to use.
        """

        if name is None or name == self.name:
            # Both normalisations are equivalent, only need a shallow copy with
            # minor changes. Reuse existing definitions of physical units.
            other = copy.copy(self)
            other.convention = NORMALISATION_CONVENTIONS[convention]
            for unit_name, unit in self.references.items():
                other_sim = getattr(other.convention, unit_name)
                if self._has_physical_units(unit):
                    other_phys = getattr(self.units, f"{other_sim}_{self.name}")
                    setattr(other, unit_name, other_phys)
                else:
                    setattr(other, unit_name, other_sim)
            other._update_system()
            return other
        else:
            # Need brand new object.
            # Recreate inputs from self, unless they are given as arguments
            # to this method.
            geometry_args = (B0, bunit_over_b0, minor_radius, aspect_ratio)
            species_args = (
                electron_temp,
                deuterium_temp,
                electron_dens,
                deuterium_dens,
                electron_mass,
                hydrogen_mass,
                deuterium_mass,
                tritium_mass,
            )
            geometry_is_none = [x is None for x in geometry_args]
            species_is_none = [x is None for x in species_args]

            if all(geometry_is_none):
                B0_unit = getattr(
                    self.units, f"bref_B0_{self.name}", self.units.bref_B0
                )
                bunit_unit = getattr(
                    self.units, f"bref_Bunit_{self.name}", self.units.bref_Bunit
                )
                minor_unit = getattr(
                    self.units,
                    f"lref_minor_radius_{self.name}",
                    self.units.lref_minor_radius,
                )
                major_unit = getattr(
                    self.units,
                    f"lref_major_radius_{self.name}",
                    self.units.lref_major_radius,
                )

                B0 = 1.0 * B0_unit
                bunit_over_b0 = ((1.0 * bunit_unit) / (1.0 * B0_unit)).to(
                    self.units.dimensionless, self.context
                )
                minor_radius = 1.0 * minor_unit
                aspect_ratio = ((1.0 * major_unit) / (1.0 * minor_unit)).to(
                    self.units.dimensionless, self.context
                )
            elif any(geometry_is_none):
                raise ValueError("Either set all geometry params or none of them")

            if all(species_is_none):
                # Recreate all temperatures/densities/masses, and ensure they're
                # in the same units
                electron_temp_unit = getattr(
                    self.units, f"tref_electron_{self.name}", self.units.tref_electron
                )
                deuterium_temp_unit = getattr(
                    self.units, f"tref_deuterium_{self.name}", self.units.tref_deuterium
                )
                electron_dens_unit = getattr(
                    self.units, f"nref_electron_{self.name}", self.units.nref_electron
                )
                deuterium_dens_unit = getattr(
                    self.units, f"nref_deuterium_{self.name}", self.units.nref_deuterium
                )
                electron_mass_unit = getattr(
                    self.units, f"mref_electron_{self.name}", self.units.mref_electron
                )
                hydrogen_mass_unit = getattr(
                    self.units, f"mref_hydrogen_{self.name}", self.units.mref_hydrogen
                )
                deuterium_mass_unit = getattr(
                    self.units, f"mref_deuterium_{self.name}", self.units.mref_deuterium
                )
                tritium_mass_unit = getattr(
                    self.units, f"mref_tritium_{self.name}", self.units.mref_tritium
                )
                electron_temp = (1.0 * electron_temp_unit).to(self.tref, self.context)
                deuterium_temp = (1.0 * deuterium_temp_unit).to(self.tref, self.context)
                electron_dens = (1.0 * electron_dens_unit).to(self.nref, self.context)
                deuterium_dens = (1.0 * deuterium_dens_unit).to(self.nref, self.context)
                electron_mass = (1.0 * electron_mass_unit).to(self.mref, self.context)
                hydrogen_mass = (1.0 * hydrogen_mass_unit).to(self.mref, self.context)
                deuterium_mass = (1.0 * deuterium_mass_unit).to(self.mref, self.context)
                tritium_mass = (1.0 * tritium_mass_unit).to(self.mref, self.context)
            elif any(species_is_none):
                raise ValueError("Either set all species params or none of them")

            return self.__class__(
                name=name,
                convention=self.convention.name if convention is None else convention,
                registry=self.units,
                B0=B0,
                bunit_over_b0=bunit_over_b0,
                minor_radius=minor_radius,
                aspect_ratio=aspect_ratio,
                electron_temperature=electron_temp,
                deuterium_temperature=deuterium_temp,
                electron_density=electron_dens,
                deuterium_density=deuterium_dens,
                electron_mass=electron_mass,
                hydrogen_mass=hydrogen_mass,
                deuterium_mass=deuterium_mass,
                tritium_mass=tritium_mass,
            )

    def with_convention(self, convention: str, name: Optional[str] = None) -> Self:
        """Convert to a new gyrokinetics convention.

        Keeps all ratios between different units the same.

        If ``name`` is ``None``, returns a shallow copy set to the new
        convention. If ``name`` is set, recreates all physical units and unit
        ratios.

        Examples
        --------

        >>> norm = ConstNormalisation(
        ...     "foo", minor_radius=2.0 * ureg.m, major_radius=5.0 * ureg.m
        ... )
        >>> print(norm.lref)
        lref_minor_radius_foo
        >>> # Make equivalent ConstNormalisation with different references
        >>> gene_norm = norm.to_convention("gene")
        >>> print(gene_norm.lref)
        lref_major_radius_foo
        >>> # Make new ConstNormalisation, redefining physical units
        >>> gene_norm_2 = norm.to_convention("gene", name="bar")
        >>> print(gene_norm_2.lref)
        lref_major_radius_bar
        """
        return self._mutate(convention=convention, name=name)

    def with_geometry_params(
        self,
        name: str,
        B0: QuantityT,
        bunit_over_b0: QuantityT,
        minor_radius: QuantityT,
        aspect_ratio: QuantityT,
    ) -> Self:
        """Duplicate self with new geometrical units."""
        return self._mutate(
            name=name,
            B0=B0,
            bunit_over_b0=bunit_over_b0,
            minor_radius=minor_radius,
            aspect_ratio=aspect_ratio,
        )

    def with_species_params(
        self,
        name: str,
        electron_temperature: QuantityT,
        deuterium_temperature: QuantityT,
        electron_density: QuantityT,
        deuterium_density: QuantityT,
        electron_mass: QuantityT,
        hydrogen_mass: QuantityT,
        deuterium_mass: QuantityT,
        tritium_mass: QuantityT,
    ) -> Self:
        """Duplicate self with new geometrical units."""
        return self._mutate(
            name=name,
            electron_temp=electron_temperature,
            deuterium_temp=deuterium_temperature,
            electron_dens=electron_density,
            deuterium_dens=deuterium_density,
            electron_mass=electron_mass,
            hydrogen_mass=hydrogen_mass,
            deuterium_mass=deuterium_mass,
            tritium_mass=tritium_mass,
        )

    def __getattr__(self, key: str):
        """Creates an equivalent instance with the given convention

        Examples
        --------

        >>> # Using norms associated with GS2, convert a quantity to GENE
        >>> growth_rate_gene = growth_rate_gs2.to(norm_gs2.gene)
        """
        if key in NORMALISATION_CONVENTIONS:
            return self.with_convention(key)
        raise AttributeError(key)
