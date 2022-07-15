from dataclasses import dataclass
from typing import Optional
from pyrokinetics.constants import electron_charge, pi
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


class LocalNorm:
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
        nocos: Optional[int] = "pyrokinetics",
        tref: Optional[str] = None,
        nref: Optional[str] = None,
        mref: Optional[str] = None,
        vref: Optional[str] = None,
        lref: Optional[str] = None,
        bref: Optional[str] = None,
    ):

        self.nocos = nocos
        self.tref = tref
        self.nref = nref
        self.mref = mref
        self.vref = vref
        self.lref = lref
        self.bref = bref

    @property
    def nocos(self) -> NormalisationConvention:
        return self._nocos

    @nocos.setter
    def nocos(self, value: str = "pyrokinetics"):
        """Set normalisation convention"""
        try:
            self._nocos = NORMALISATION_CONVENTIONS[value]
        except KeyError:
            raise NotImplementedError(f"NOCOS value {value} not yet supported")

    def update_derived_values(self):
        """
        Updates dervived quantities
        beta
        rho_ref
        Returns
        -------

        """

        self.calculate_beta()
        self.calculate_rhoref()

    def calculate_beta(self):
        """
        Calculates beta from normalised value
        Returns
        -------
        self.beta
        """

        if self.bref is not None:
            if self.nref is not None:
                self.beta = (
                    self.nref
                    * self.tref
                    * electron_charge
                    / self.bref**2
                    * 8
                    * pi
                    * 1e-7
                )
            else:
                self.beta = 1 / self.bref**2
        else:
            self.beta = None

    def calculate_rhoref(self):
        """
        Calculates reference Larmor radius
        Returns
        -------

        """

        if self.vref is not None and self.bref is not None:
            self.rhoref = self.mref * self.vref / electron_charge / self.bref
        else:
            self.rhoref = None

    def from_kinetics(
        self, kinetics, psi_n=None, tref=None, nref=None, vref=None, mref=None
    ):
        """
        Loads local normalising species data from kinetics object

        """

        if psi_n is None:
            raise ValueError("Need value of psi_n")

        if tref is None:
            tref = kinetics.species_data[self.nocos.tref_species].get_temp(psi_n)

        if nref is None:
            nref = kinetics.species_data[self.nocos.nref_species].get_dens(psi_n)

        if mref is None:
            mref = kinetics.species_data[self.nocos.mref_species].get_mass()

        if vref is None:
            vref = np.sqrt(electron_charge * tref / mref) * self.nocos.vref_multiplier

        self.tref = tref
        self.nref = nref
        self.mref = mref
        self.vref = vref

        self.update_derived_values()

    def from_local_geometry(self, local_geometry):
        """
        Loads local normalising field from LocalGeometry Object

        """

        if self.nocos.bref_type == "B0":
            self.bref = local_geometry.B0
        elif self.nocos.bref_type == "Bunit":
            self.bref = local_geometry.B0 * local_geometry.bunit_over_b0
        else:
            raise ValueError(f"bref_type : {self.nocos.bref_type} is not recognised")

        self.update_derived_values()
