from typing import Optional
import numpy as np


class Nocos:
    """
    Class defining the normalising quantities
    """

    def __init__(
        self,
        nocos: Optional[int] = None,
        tref_species: Optional["str"] = None,
        nref_species: Optional["str"] = None,
        mref_species: Optional["str"] = None,
        vref_multiplier: Optional["str"] = None,
        lref_type: Optional["str"] = None,
        bref_type: Optional["str"] = None,
    ):

        if nocos is not None:
            self.nocos = nocos

        if tref_species is not None:
            self._nocos = -1
            self.tref_species = tref_species

        if nref_species is not None:
            self._nocos = -1
            self.nef_species = nref_species

        if mref_species is not None:
            self._nocos = -1
            self.mref_species = mref_species

        if vref_multiplier is not None:
            self._nocos = -1
            self.vref_multplier = vref_multiplier

        if lref_type is not None:
            self._nocos = -1
            self.lref_type = lref_type

        if bref_type is not None:
            self._nocos = -1
            self.bref_type = bref_type

    @property
    def nocos(self):
        return self._nocos

    @nocos.setter
    def nocos(self, nocos_number):

        # Pyrokinetics default
        if nocos_number == 1:
            self._nocos = nocos_number
            self.tref_species = "electron"
            self.nref_species = "electron"
            self.mref_species = "deuterium"
            self.vref_multplier = 1.0
            self.lref_type = "minor_radius"
            self.bref_type = "B0"

        # CGYRO default
        elif nocos_number == 2:
            self._nocos = nocos_number
            self.tref_species = "electron"
            self.nref_species = "electron"
            self.mref_species = "deuterium"
            self.vref_multplier = 1.0
            self.lref_type = "minor_radius"
            self.bref_type = "Bunit"

        # GS2 default
        elif nocos_number == 3:
            self._nocos = nocos_number
            self.tref_species = "electron"
            self.nref_species = "electron"
            self.mref_species = "deuterium"
            self.vref_multplier = np.sqrt(2)
            self.lref_type = "minor_radius"
            self.bref_type = "B0"

        # GENE default
        elif nocos_number == 4:
            self._nocos = nocos_number
            self.tref_species = "electron"
            self.nref_species = "electron"
            self.mref_species = "deuterium"
            self.vref_multplier = 1.0
            self.lref_type = "major_radius"
            self.bref_type = "B0"

        # Using bespoke NOCOS
        elif nocos_number == -1:
            self._nocos = nocos_number

        else:
            raise NotImplementedError(
                f"NOCOS convention {nocos_number} not yet implemented. Try add_cocos method"
            )

    def add_cocos(self, data_dict: dict):
        """
        Create bespoke NOCOS convention
        """

        self._nocos = -1
        self.tref_species = data_dict["tref_species"]
        self.nref_species = data_dict["nref_species"]
        self.mref_species = data_dict["mref_species"]
        self.vref_multplier = data_dict["vref_multiplier"]
        self.lref_type = data_dict["lref_type"]
        self.bref_type = data_dict["bref_type"]
