from dataclasses import dataclass
from typing import Optional
from pyrokinetics.constants import electron_charge, pi
from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometry
import numpy as np


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


NORMALISATION_CONVENTIONS = {
    "pyrokinetics": NormalisationConvention("pyrokinetics"),
    "cgyro": NormalisationConvention("cgyro", bref_type="Bunit"),
    "gs2": NormalisationConvention("gs2", vref_multiplier=np.sqrt(2)),
    "gene": NormalisationConvention("gene", lref_type="major_radius"),
}
"""Particular normalisation conventions"""


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

        return self.nref * self.tref * electron_charge / self.bref**2 * 8 * pi * 1e-7

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
