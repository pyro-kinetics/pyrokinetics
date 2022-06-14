from .NOCOS import Nocos
from typing import Optional
from pyrokinetics.constants import electron_charge, pi
import numpy as np


class LocalNorm:
    """
    Dictionary of local beta - only parameter that needs both equilibrium and kinetic profiles

    """

    def __init__(
        self,
        nocos: Optional[int] = 1,
        tref: Optional["str"] = None,
        nref: Optional["str"] = None,
        mref: Optional["str"] = None,
        vref: Optional["str"] = None,
        lref: Optional["str"] = None,
        bref: Optional["str"] = None,
    ):

        self.nocos = nocos
        self.tref = tref
        self.nref = nref
        self.mref = mref
        self.vref = vref
        self.lref = lref
        self.bref = bref

    @property
    def nocos(self) -> Nocos:
        return self._nocos

    @nocos.setter
    def nocos(self, value: Optional[int]):
        """
        Set NOCOs value given a specific integer
        """
        if value is None:
            self._nocos = Nocos(1)
            return
        else:
            try:
                self._nocos = Nocos(value)
            except ValueError:
                raise NotImplementedError(f"NOCOS value {value} not yet supported")

    def from_dict(self, norms_dict, **kwargs):
        """
        Reads local norms parameters from a dictionary

        """

        if isinstance(norms_dict, dict):
            sort_norms_dict = sorted(norms_dict.items())

            super(LocalNorm, self).__init__(*sort_norms_dict, **kwargs)

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
        self,
        kinetics,
        psi_n=None,
        tref=None,
        nref=None,
        vref=None,
        mref=None,
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
            vref = np.sqrt(electron_charge * tref / mref) * self.nocos.vref_multplier

        self.tref = tref
        self.nref = nref
        self.mref = mref
        self.vref = vref

        self.update_derived_values()

    def from_local_geometry(
        self,
        local_geometry,
    ):
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
