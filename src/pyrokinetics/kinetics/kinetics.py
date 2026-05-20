from __future__ import annotations  # noqa

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from cleverdict import CleverDict

from ..file_utils import ReadableFromFile
from ..typing import PathLike
from ..units import UnitSpline  # local import to avoid changing top-level imports
from ..units import ureg as units

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


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

    def get_total_pressure(
        self,
        psi_n=None,
        *,
        exclude_species=None,
        exclude_fast: bool = False,
    ):
        """
        Total pressure-like quantity p_tot(psi_n) = sum_s p_s(psi_n),
        with optional exclusion of selected species.
        """
        if psi_n is None:
            psi_n = np.linspace(0, 1.0, 100) * units.dimensionless
        elif not hasattr(psi_n, "units"):
            psi_n = np.asarray(psi_n, dtype=float) * units.dimensionless

        if exclude_species is None:
            exclude_species = set()
        else:
            exclude_species = set(exclude_species)

        if exclude_fast:
            # conservative heuristic: anything with '_fast' or common fast-ion labels
            for name in self.species_data.keys():
                if name.endswith("_fast") or name in ("alpha", "beam", "fast", "nbi"):
                    exclude_species.add(name)

        total = 0.0
        for name, s in self.species_data.items():
            if name in exclude_species:
                continue
            total = total + s.get_pressure(psi_n)

        return total

    def p_prime(
        self,
        psi_n=None,
        *,
        eq=None,
        exclude_species=None,
        exclude_fast: bool = False,
        method: str = "spline",
    ):
        """
        Return dp/dpsi (physical poloidal flux), consistent with Equilibrium.p_prime.

        Notes
        -----
        - Requires an Equilibrium object to map psi_n -> psi via eq.psi(...), which is affine:
            psi(psi_n) = psi_axis + psi_n * (psi_lcfs - psi_axis)
          so dpsi/dpsi_n = (psi_lcfs - psi_axis).
        - If no eq is supplied,  error is raised.

        Parameters
        ----------
        psi_n : array-like or pint quantity, optional
            Normalised poloidal flux coordinate(s). If None, uses 100 points in [0,1].
        eq : Equilibrium (required)
            Equilibrium used to define psi(psi_n) and dpsi/dpsi_n.
        exclude_species : bool
            List of species to be excluded form the total pressure calculation.
        exclude_fast : bool
            If true excludes species that have 'alpha' or '_fast' in their name.
        method : {"spline", "gradient"}
            How to compute dp/dpsi_n before converting to dp/dpsi.

        Returns
        -------
        pint quantity array
            dp/dpsi with units of Pa / Wb (or Pa / psi_unit used by Equilibrium internally).
        """

        if eq is None:
            eq = getattr(self, "eq", None)

        if eq is None:
            raise ValueError(
                "Kinetics.p_prime requires an Equilibrium 'eq' to define dp/dpsi. "
                "No equilibrium supplied. (Use get_total_pressure() or implement dp/dpsi_n instead.)"
            )

        # Build psi_n with units
        if psi_n is None:
            psi_n = np.linspace(0.0, 1.0, 100) * units.dimensionless
        elif not hasattr(psi_n, "units"):
            psi_n = np.asarray(psi_n, dtype=float) * units.dimensionless

        # Total pressure p(psi_n)
        p = self.get_total_pressure(
            psi_n, exclude_species=exclude_species, exclude_fast=exclude_fast
        )

   
        sp = UnitSpline(psi_n, p)
        dp_dpsin = sp(psi_n, derivative=1)
   
        psi = eq.psi(psi_n)

        # Convert dp/dpsi_n -> dp/dpsi using equilibrium affine mapping
        _spline = UnitSpline(psi_n, psi)
        dpsi_dpsin = _spline(psi_n, derivative=1)  # should be (eq.psi_lcfs - eq.psi_axis)  # constant
        return dp_dpsin / dpsi_dpsin

    @staticmethod
    def Z_profile(species, round_charge, psi_q):
        z = species.get_charge(psi_q).to("elementary_charge").m
        z = np.abs(z)
        if round_charge:
            z = np.rint(z)
        return z.astype(float)

    def enforce_quasineutrality(
        self,
        *,
        adjust_species: str = "deuterium",
        psi=None,
        npsi: int = 101,
        floor: float = 0.0,
        round_charge: bool = False,
    ):
        """
        Enforce quasineutrality globally by adjusting one species density profile.

        Uses Z(psi) from Species.get_charge(psi). By default it does NOT force rounding,
        so ψ-dependent / fractional charge is preserved. If round_charge=True, applies
        pointwise integer rounding after converting to e (elementary_charge).

        ne(psi) = sum_{ions} Z_i(psi) n_i(psi)
        """
        sp = self.species_data

        if adjust_species not in sp:
            raise ValueError(f"{adjust_species} not found in species_data")
        if "electron" not in sp:
            raise ValueError(
                "Cannot enforce quasineutrality: 'electron' not in species_data"
            )

        if psi is None:
            psi = np.linspace(0.0, 1.0, npsi)
        else:
            psi = np.asarray(psi, dtype=float)

        psi_q = psi * units.dimensionless

        ne = sp["electron"].get_dens(psi_q).to("meter**-3").m

        charge_sum_other = np.zeros_like(ne)
        for name in list(sp.keys()):  # species_names is derived from dict keys
            if name in ("electron", adjust_species):
                continue
            s = sp[name]
            Zi = self.Z_profile(s, round_charge, psi_q)
            ni = s.get_dens(psi_q).to("meter**-3").m
            charge_sum_other += Zi * ni

        Z_adj = self.Z_profile(sp[adjust_species], round_charge, psi_q)
        if np.any(Z_adj == 0.0):
            raise ValueError(
                f"Adjusted species '{adjust_species}' has Z=0 at some psi points."
            )

        n_adj_new = (ne - charge_sum_other) / Z_adj

        if floor is not None:
            n_adj_new = np.maximum(n_adj_new, float(floor))

        sp[adjust_species].dens = sp[adjust_species].dens.__class__(
            psi_q, n_adj_new * units("meter**-3")
        )

        return psi, n_adj_new

    def merge_species_global(
        self,
        *,
        base_species: str,
        merge_species,
        psi=None,
        npsi: int = 101,
        dens_floor: float = 0.0,
        remove_merged: bool = True,
        enforce_qn: bool = False,
        keep_base_species_z: bool = True,
        keep_base_species_mass: bool = True,
        round_charge: bool = False,
    ):
        """
        Global merge analogue of LocalSpecies.merge_species(), implemented for global Species.

        Uses profile charge Z(psi) := |q(psi)|/e from Species.get_charge(psi).
        Default round_charge=False preserves ψ-dependent/fractional Z if present.
        If round_charge=True, applies pointwise np.rint to reduce floating noise.

        If keep_base_species_z=True:
            n_base_new(psi) = sum_i [ Z_i(psi) n_i(psi) ] / Z_base(psi)
            (Z_base is used as a profile, so quasineutrality is preserved pointwise.)

        If keep_base_species_z=False:
            n_new(psi) = sum_i n_i(psi)
            Z_eff(psi) = sum_i [ Z_i(psi) n_i(psi) ] / n_new(psi)
            and base.charge spline is replaced by q_eff(psi)=+Z_eff(psi)*e.

        If keep_base_species_mass=False:
            m_eff(psi) = sum_i [ m_i n_i(psi) ] / n_new(psi)
            but Species.mass is scalar in this API (get_mass has no psi argument), so we
            store a density-weighted scalar effective mass, and return m_eff(psi) for inspection.

        Notes
        -----
        - No explicit gradient updates are needed: get_norm_dens_gradient is computed from self.dens
          via _norm_gradient(self.dens, psi).
        - Removal updates species_data only (species_names is derived from keys).
        """
        sp = self.species_data

        if base_species not in sp:
            raise ValueError(f"Unrecognised base_species {base_species}")
        if base_species == "electron":
            raise ValueError(
                "Refusing to use 'electron' as base_species for ion merging"
            )

        merge_set = sorted(set(list(merge_species) + [base_species]))
        missing = [n for n in merge_set if n not in sp]
        if missing:
            raise ValueError(f"Unrecognised merge_species: {missing}")

        # psi grid
        if psi is None:
            psi = np.linspace(0.0, 1.0, npsi)
        else:
            psi = np.asarray(psi, dtype=float)

        if psi.ndim != 1 or psi.size < 2:
            raise ValueError("psi must be a 1D array with at least 2 points")
        if np.any(np.diff(psi) <= 0):
            raise ValueError("psi grid must be strictly increasing")

        psi_q = psi * units.dimensionless

        # Collect n_i(psi), Z_i(psi), and scalar masses
        dens_arr = []
        Z_arr = []
        m_list = []

        for name in merge_set:
            s = sp[name]
            dens_arr.append(s.get_dens(psi_q).to("meter**-3").m)
            Z_arr.append(self.Z_profile(s, round_charge, psi_q))
            m_list.append(float(getattr(s.mass, "m", s.mass)))  # scalar mass

        dens_arr = np.stack(dens_arr, axis=0)  # (ns, npsi)
        Z_arr = np.stack(Z_arr, axis=0)  # (ns, npsi)
        m_arr = np.asarray(m_list, dtype=float)[:, None]  # (ns, 1)

        base = sp[base_species]

        # --- Merge density & charge ---
        if keep_base_species_z:
            Zb = self.Z_profile(base, round_charge, psi_q)  # profile Z_base(psi)
            if np.any(Zb == 0.0):
                raise ValueError(
                    f"Base species '{base_species}' has Z=0 at some psi points."
                )
            n_new = np.sum(dens_arr * Z_arr, axis=0) / Zb
            Z_eff_profile = Zb.copy()  # unchanged base charge profile
            # charge spline unchanged in this branch
        else:
            n_new = np.sum(dens_arr, axis=0)
            Z_eff_profile = np.sum(dens_arr * Z_arr, axis=0) / np.maximum(n_new, 1e-300)

            # Replace base.charge spline by q_eff(psi)=+Z_eff(psi)*e
            q_eff = Z_eff_profile * units.elementary_charge

            # Use the existing charge spline class if possible, else fall back to UnitSpline
            if getattr(base, "charge", None) is not None:
                base.charge = base.charge.__class__(psi_q, q_eff)
            else:
                base.charge = UnitSpline(psi_q, q_eff)

        # Apply density floor (after computing Z_eff where needed)
        if dens_floor is not None:
            n_new = np.maximum(n_new, float(dens_floor))

        # Write back density spline (same pattern you already use)
        base.dens = base.dens.__class__(psi_q, n_new * units("meter**-3"))

        # --- Mass merge ---
        if keep_base_species_mass:
            m_eff_profile = np.full_like(
                n_new, float(getattr(base.mass, "m", base.mass)), dtype=float
            )
            # base.mass unchanged
        else:
            # m_eff(psi) = sum(m_i n_i)/sum(n_i)
            m_eff_profile = np.sum(m_arr * dens_arr, axis=0) / np.maximum(n_new, 1e-300)
            # Store a scalar effective mass (density-weighted over psi)
            w = np.maximum(n_new, 0.0)
            m_store = float(np.sum(m_eff_profile * w) / np.sum(w))
            base.mass = m_store

        # --- Remove or zero merged-away species ---
        merged_away = [n for n in merge_set if n != base_species]
        if remove_merged:
            for n in merged_away:
                sp.pop(n, None)
        else:
            for n in merged_away:
                s = sp[n]
                s.dens = s.dens.__class__(
                    psi_q, np.zeros_like(psi) * units("meter**-3")
                )

        # --- Optional enforce quasineutrality by adjusting base density on same psi grid ---
        if enforce_qn:
            self.enforce_quasineutrality(
                adjust_species=base_species,
                psi=psi,
                npsi=len(psi),
                floor=dens_floor,
                round_charge=round_charge,
            )

        return psi, {
            "base_species": base_species,
            "merged": merge_set,
            "removed": merged_away if remove_merged else [],
            "zeroed": merged_away if (not remove_merged) else [],
            "keep_base_species_z": keep_base_species_z,
            "keep_base_species_mass": keep_base_species_mass,
            "round_charge": round_charge,
            "Z_eff_profile": Z_eff_profile,
            "m_eff_profile": m_eff_profile,
            "enforce_qn": enforce_qn,
        }

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

        psi_n = np.linspace(0, 1.0, 100) * units.dimensionless

        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))
        else:
            fig = ax[0].figure

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
