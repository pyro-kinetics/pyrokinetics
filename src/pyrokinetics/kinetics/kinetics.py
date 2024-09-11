from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from cleverdict import CleverDict

from ..file_utils import ReadableFromFile
from ..typing import PathLike


class Kinetics(ReadableFromFile):
    """
    Contains all the kinetic data in the form of Species objects.
    Data can be accessed via `species_data`, which is a CleverDict with each
    key being a species name. For example, electron data can be accessed via a call
    to ``kinetics.species_data["electron"]`` or ``kinetics.species_data.electron``.

    Each Species is provided with:

    - psi_n: ArrayLike       [units] dimensionless
        1D array of normalised poloidal flux for each flux surface where data is defined
    - r/a: ArrayLike         [units] dimensionless
        1D array of normalised minor radius for each flux surface. This is needed for derivatives w.r.t rho (r/a)
    - Charge: Int      [units] elementary_charge
        Charge of each species
    - Mass: ArrayLike        [units] kg
        Mass of each species
    - Temperature: ArrayLike [units] eV
        1D array of the species temperature profile
    - Density: ArrayLike     [units] meter**-3
        1D array of the species density profile
    - Rotation: ArrayLike    [units] /second
        1D array of the species rotation profile

    Parameters
    ----------
    kinetics_type: str, default None
        Name of the kinetics input type, such as "SCENE", "JETTO", etc.
    **kwargs
        Used to pass in species data.
    """

    def __init__(self, kinetics_type: str, **kwargs):
        self.kinetics_type = kinetics_type
        self.species_data = CleverDict(**kwargs)
        """``CleverDict`` containing kinetics info for each species. May include
        entries such as 'electron' and 'deuterium'"""

    @property
    def kinetics_type(self):
        """Stored reference of the last kinetics type. May be inferred"""
        return self._kinetics_type

    @kinetics_type.setter
    def kinetics_type(self, value):
        if value not in self.supported_file_types():
            raise ValueError(f"Kinetics type {value} is not currently supported.")
        self._kinetics_type = value

    @property
    def nspec(self):
        """Number of species"""
        return len(self.species_data)

    @property
    def species_names(self):
        """Names of each species"""
        return self.species_data.keys()

    def __deepcopy__(self, memodict):
        """
        Allows for deepcopy of a Kinetics object

        Returns
        -------
        Copy of kinetics object
        """
        # Create new object without calling __init__
        new_kinetics = Kinetics.__new__(Kinetics)
        # Deep copy each member besides species_data
        for key, value in self.__dict__.items():
            if key != "species_data":
                setattr(new_kinetics, key, value)
        # Build new species_data dict and populate one element at a time
        # (Note: we're not deepcopying Species. Species should have a __deepcopy__)
        new_kinetics.species_data = CleverDict()
        for name, species in self.species_data.items():
            new_kinetics.species_data[name] = species
        return new_kinetics

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
        x_grid: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        r"""
        Plot a quantity defined on the :math:`\psi` grid.

        Parameters
        ----------
        quantity: str
            Name of the quantity to plot. Must be defined over the grid ``psi``.
        ax: Optional[plt.Axes]
            Axes object on which to plot. If not provided, a new figure is created.
        show: bool, default False
            Immediately show Figure after creation.
        x_grid: Optional[str], default None
            Radial grid to plot against. Options are psi_n (default) and r/a
        **kwargs
            Additional arguments to pass to Matplotlib's ``plot`` call.

        Returns
        -------
        plt.Axes
            The Axes object created after plotting.

        Raises
        ------
        ValueError
            If ``quantity`` is not a quantity defined over the :math:`\psi` grid,
            or is not the name of an Equilibrium quantity.
        """
        import matplotlib.pyplot as plt

        psi_n = np.linspace(0, 1.0, 100)

        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))

        if x_grid in [None, "psi_n"]:
            x_label = r"$\psi_{N}$"
            x_grid = psi_n
        elif x_grid == "r/a":
            x_label = r"$r/a$"
            x_grid = self.species_data[list(self.species_names)[0]].get_rho(psi_n)
        else:
            x_label = ""
            x_grid = psi_n

        for species in self.species_names:
            ax[0].plot(
                x_grid.m,
                self.species_data[species].get_dens(psi_n).to("meter**-3").m,
                label=species,
            )
            ax[1].plot(
                x_grid.m,
                self.species_data[species].get_temp(psi_n).to("keV").m,
                label=species,
            )
            ax[2].plot(
                x_grid.m,
                self.species_data[species]
                .get_angular_velocity(psi_n)
                .to("second**-1")
                .m,
                label=species,
            )

        if x_label != "":
            ax[0].set_xlabel(x_label)
            ax[1].set_xlabel(x_label)
            ax[2].set_xlabel(x_label)

        ax[0].set_ylabel("$m^{-3}$")
        ax[1].set_ylabel("$keV$")
        ax[2].set_ylabel("$s^{-1}$")

        ax[0].legend()
        ax[0].grid()
        ax[0].set_ylim(bottom=0.0)
        ax[0].set_title("Density")

        ax[1].grid()
        ax[1].set_ylim(bottom=0.0)
        ax[1].set_title("Temperature")

        ax[2].grid()
        ax[2].set_title("Angular frequency")
        fig.tight_layout()

        if show:
            plt.show()

        return ax


def read_kinetics(
    path: PathLike, file_type: Optional[str] = None, **kwargs
) -> Kinetics:
    r"""A plain-function alternative to ``Kinetics.from_file``."""
    return Kinetics.from_file(path, file_type=file_type, **kwargs)


def supported_kinetics_types() -> List[str]:
    r"""A plain-function alternative to ``Kinetics.supported_file_types``."""
    return Kinetics.supported_file_types()
