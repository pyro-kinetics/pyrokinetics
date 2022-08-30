from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import pint

from pyrokinetics.constants import electron_charge, mu0
from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometry


@dataclass
class NormalisationConvention:
    """The set of normalising quantities for a given normalisation convention

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
    context: pint.Context = field(init=False)

    def __post_init__(self):
        self.context = pint.Context(self.name)


NORMALISATION_CONVENTIONS = {
    "pyrokinetics": NormalisationConvention("pyrokinetics"),
    "cgyro": NormalisationConvention("cgyro", bref_type="Bunit"),
    "gs2": NormalisationConvention("gs2", vref_multiplier=np.sqrt(2)),
    "gene": NormalisationConvention("gene", lref_type="major_radius"),
    "gkdb": NormalisationConvention("gkdb", vref_multiplier=np.sqrt(2)),
    "imas": NormalisationConvention("imas", vref_multiplier=np.sqrt(2)),
}
"""Particular normalisation conventions"""


def _create_unit_registry(conventions):
    ureg = pint.UnitRegistry()
    ureg.enable_contexts("boltzmann")

    REF_DEFS = {
        "deuterium_mass": {"def": "3.3435837724e-27 kg"},
        "bref": {"def": "nan tesla", "base": "tesla"},
        "lref": {"def": "nan metres", "base": "meter"},
        "mref": {"def": "deuterium_mass", "base": "gram"},
        "nref": {"def": "nan m**-3"},
        "qref": {"def": "elementary_charge"},
        "tref": {"def": "nan eV", "base": "kelvin"},
        "vref": {"def": "(tref / mref)**(0.5)"},
        "beta": {"def": "2 * mu0 * nref * tref / bref**2"},
        "rhoref": {"def": "mref * vref / qref / bref"},
    }

    for unit, definition in REF_DEFS.items():
        ureg.define(f"{unit} = {definition['def']}")

    for name, convention in conventions.items():
        group = ureg.get_group(name)
        group.add_units(*REF_DEFS.keys())
        system = ureg.get_system(name)
        system.add_groups(name)

        for unit, definition in REF_DEFS.items():
            convention_unit = f"{name}_{unit}"

            unit_def = definition["def"]
            for unit_name in list(REF_DEFS.keys()):
                unit_def = unit_def.replace(unit_name, f"{name}_{unit_name}")

            if unit == "vref":
                unit_def = f"{convention.vref_multiplier} * {unit_def}"

            ureg.define(f"{convention_unit} = {unit_def}")
            convention.context.redefine(f"{unit} = {convention_unit}")

            if "base" in definition:
                system.base_units[definition["base"]] = {convention_unit: 1.0}

        ureg.add_context(convention.context)

    return ureg


def set_reference_quantities_from_local_geometry(
    ureg: pint.UnitRegistry, local_geometry: LocalGeometry
):
    """Create a `Normalisation` using local normalising field from `LocalGeometry` Object."""

    BREF_TYPES = {
        "B0": local_geometry.B0,
        "Bunit": local_geometry.B0 * local_geometry.bunit_over_b0,
    }
    LREF_TYPES = {
        "minor_radius": local_geometry.a_minor,
        "major_radius": local_geometry.Rmaj,
    }

    for name, convention in NORMALISATION_CONVENTIONS.items():
        if convention.bref_type not in BREF_TYPES:
            raise ValueError(
                f"Unrecognised bref_type: got '{convention.bref_type}', expected one of {list(BREF_TYPES.keys())}"
            )

        if convention.lref_type not in LREF_TYPES:
            raise ValueError(
                f"Unrecognised lref_type: got '{convention.lref_type}', expected one of {list(LREF_TYPES.keys())}"
            )

        bref = BREF_TYPES[convention.bref_type]
        ureg.define(f"{name}_bref = {bref} tesla")
        lref = LREF_TYPES[convention.lref_type]
        ureg.define(f"{name}_lref = {lref} metres")


def set_reference_quantities_from_kinetics(
    ureg: pint.UnitRegistry, kinetics: Kinetics, psi_n: float
):
    """Create a `Normalisation` using local normalising species data from kinetics object"""

    for name, convention in NORMALISATION_CONVENTIONS.items():
        tref = kinetics.species_data[convention.tref_species].get_temp(psi_n)
        nref = kinetics.species_data[convention.nref_species].get_dens(psi_n)
        mref = kinetics.species_data[convention.mref_species].get_mass()

        ureg.define(f"{name}_tref = {tref} eV")
        ureg.define(f"{name}_nref = {nref} m**-3")
        ureg.define(f"{name}_mref = {mref} kg")


class Normalisation:
    """A concrete set of normalisation parameters following a given convention

    Attributes
    ----------
    tref:
        Reference temperature
    nref:
        Reference density
    mref:
        Reference mass
    vref:
        Reference velocity
    lref:
        Reference length scale
    bref:
        Reference magnetic field
    """

    def __init__(
        self,
        convention: Optional[int] = "pyrokinetics",
        tref: Optional[float] = None,
        nref: Optional[float] = None,
        mref: Optional[float] = None,
        vref: Optional[float] = None,
        lref: Optional[float] = None,
        bref: Optional[float] = None,
        beta: Optional[float] = None,
        rhoref: Optional[float] = None,
    ):

        self.nocos = self.choose_convention(convention)
        self.tref = tref
        self.nref = nref
        self.mref = mref
        self.vref = vref
        self.lref = lref
        self.bref = bref
        self.beta = self._calculate_beta()
        self.rhoref = self._calculate_rhoref()

    def __repr__(self):
        return (
            f"Normalisation(nocos='{self.nocos.name}', "
            f"tref={self.tref}, "
            f"nref={self.nref}, "
            f"mref={self.mref}, "
            f"vref={self.vref}, "
            f"lref={self.lref}, "
            f"bref={self.bref}, "
            f"beta={self.beta}, "
            f"rhoref={self.rhoref}"
            ")"
        )

    @staticmethod
    def choose_convention(convention: str = "pyrokinetics"):
        """Set normalisation convention"""
        try:
            return NORMALISATION_CONVENTIONS[convention]
        except KeyError:
            raise NotImplementedError(f"NOCOS value {convention} not yet supported")

    def _calculate_beta(self):
        """Return beta from normalised value"""

        if self.bref is None:
            return None

        if self.nref is None:
            return 1.0 / self.bref**2

        return self.nref * self.tref * electron_charge / (self.bref**2 / (2 * mu0))

    def _calculate_rhoref(self):
        """Return reference Larmor radius"""

        if self.vref is None or self.bref is None:
            return None

        return self.mref * self.vref / electron_charge / self.bref

    @classmethod
    def from_kinetics(
        cls,
        kinetics: Kinetics,
        psi_n: float,
        convention: str = "pyrokinetics",
        lref: Optional[float] = None,
        bref: Optional[float] = None,
    ):
        """Create a `Normalisation` using local normalising species data from kinetics object"""

        nocos = cls.choose_convention(convention)

        tref = kinetics.species_data[nocos.tref_species].get_temp(psi_n)
        nref = kinetics.species_data[nocos.nref_species].get_dens(psi_n)
        mref = kinetics.species_data[nocos.mref_species].get_mass()
        vref = np.sqrt(electron_charge * tref / mref) * nocos.vref_multiplier

        return cls(
            convention, tref=tref, nref=nref, mref=mref, vref=vref, lref=lref, bref=bref
        )

    @classmethod
    def from_local_geometry(
        cls, local_geometry: LocalGeometry, convention: str = "pyrokinetics", **kwargs
    ):
        """Create a `Normalisation` using local normalising field from `LocalGeometry` Object.

        This really only sets `bref`, and you'll likely need to pass that into `from_kinetics`
        """

        nocos = cls.choose_convention(convention)

        if nocos.bref_type == "B0":
            bref = local_geometry.B0
        elif nocos.bref_type == "Bunit":
            bref = local_geometry.B0 * local_geometry.bunit_over_b0
        else:
            raise ValueError(f"bref_type : {nocos.bref_type} is not recognised")

        return cls(convention, bref=bref, **kwargs)
