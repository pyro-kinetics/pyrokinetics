from __future__ import annotations  # noqa

from typing import TYPE_CHECKING, List, Optional

import numpy as np
from cleverdict import CleverDict

from ..file_utils import ReadableFromFile
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _mass_kg(species) -> float:
    """
    Magnitude of a species' mass in kg.

    ``Species.mass`` is a ``pint`` quantity, but taking its bare ``.m`` would
    trust whatever unit it happens to carry. Convert first so masses collected
    from different species are always on the same scale.
    """
    mass = species.mass
    return float(mass.to("kg").m) if hasattr(mass, "to") else float(mass)


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

    # Species names treated as fast ions by ``exclude_fast``, in addition to any
    # name ending in ``"_fast"``
    _FAST_SPECIES_NAMES = ("alpha", "beam", "fast", "nbi")

    def _included_species(self, exclude_species=None, exclude_fast=False):
        """
        Species remaining once ``exclude_species`` and, optionally, fast species
        are left out.

        Fast species are identified by name only: any name ending in ``"_fast"``,
        or one of ``_FAST_SPECIES_NAMES``. This is deliberately narrower than
        ``BootstrapModel._is_fast`` in ``diagnostics.neoclassical``, which also
        treats any species more than ten times hotter than the electrons as fast.

        Parameters
        ----------
        exclude_species : Iterable[str], optional
            Names of species to leave out
        exclude_fast : bool, default False
            Whether to also leave out species identified as fast by name

        Returns
        -------
        dict
            Names mapped to ``Species`` for the species that remain, in
            ``species_data`` order
        """
        excluded = set() if exclude_species is None else set(exclude_species)

        return {
            name: species
            for name, species in self.species_data.items()
            if name not in excluded
            and not (
                exclude_fast
                and (name.endswith("_fast") or name in self._FAST_SPECIES_NAMES)
            )
        }

    def get_total_pressure(
        self,
        psi_n=None,
        *,
        exclude_species=None,
        exclude_fast: bool = False,
    ):
        r"""
        Total pressure :math:`p = \sum_s n_s T_s`, summed over species.

        Parameters
        ----------
        psi_n : ArrayLike
            Normalised poloidal flux at which to evaluate the pressure
        exclude_species : Iterable[str], optional
            Names of species to leave out of the sum
        exclude_fast : bool, default False
            Whether to also leave out fast species, identified by name. See
            ``_included_species``.

        Returns
        -------
        pint.Quantity
            Total pressure in pascals
        """

        if not hasattr(psi_n, "units"):
            psi_n *= units.dimensionless

        species = self._included_species(exclude_species, exclude_fast)

        return sum(s.get_pressure(psi_n) for s in species.values())

    def get_norm_total_pressure_gradient(
        self,
        psi_n=None,
        *,
        exclude_species=None,
        exclude_fast: bool = False,
    ):
        r"""
        Normalised gradient of the total pressure,
        :math:`-\frac{1}{p} \frac{\partial p}{\partial \rho}`.

        Built as the pressure-weighted sum of each species' normalised pressure
        gradient, so it is consistent with ``LocalSpecies.inverse_lp``.

        Parameters
        ----------
        psi_n : ArrayLike
            Normalised poloidal flux at which to evaluate the gradient
        exclude_species : Iterable[str], optional
            Names of species to leave out of both the pressure and its gradient
        exclude_fast : bool, default False
            Whether to also leave out fast species, identified by name. See
            ``_included_species``.

        Returns
        -------
        pint.Quantity
            Normalised pressure gradient, units [1 / lref_minor_radius]. Zero if the
            total pressure vanishes.
        """
        if not hasattr(psi_n, "units"):
            psi_n *= units.dimensionless

        species = self._included_species(exclude_species, exclude_fast)

        total_pressure = sum(s.get_pressure(psi_n) for s in species.values())

        if np.isclose(total_pressure.magnitude, 0.0):
            return 0.0 / units.lref_minor_radius

        numerator = 0.0 * total_pressure / units.lref_minor_radius

        for s in species.values():
            numerator += s.get_pressure(psi_n) * s.get_norm_pressure_gradient(psi_n)

        return numerator / total_pressure

    def get_total_pressure_prime(
        self,
        psi_n=None,
        *,
        eq=None,
        exclude_species=None,
        exclude_fast: bool = False,
    ):
        r"""
        Derivative of the total pressure with respect to the poloidal flux,
        :math:`\partial p / \partial \psi`, consistent with ``Equilibrium.p_prime``.

        Parameters
        ----------
        psi_n : ArrayLike
            Normalised poloidal flux at which to evaluate the derivative
        eq : Equilibrium, optional
            Equilibrium defining :math:`\psi(\psi_N)`. Defaults to ``self.eq``,
            which ``Pyro`` sets when both a global equilibrium and kinetics are
            loaded.
        exclude_species : Iterable[str], optional
            Names of species to leave out of the total pressure
        exclude_fast : bool, default False
            Whether to also leave out fast species, identified by name. See
            ``_included_species``.

        Returns
        -------
        pint.Quantity
            :math:`\partial p / \partial \psi`, in pascals per unit of the
            equilibrium's :math:`\psi`

        Raises
        ------
        ValueError
            If ``psi_n`` is not given, or no equilibrium is given or attached.

        Notes
        -----
        The derivative is taken with the product rule,
        :math:`\partial p / \partial \psi_N = \sum_s (n_s' T_s + n_s T_s')`, directly
        from each species' own density and temperature splines, so no resampling
        is involved. :math:`\psi` is affine in :math:`\psi_N`,
        :math:`\psi = \psi_{axis} + \psi_N (\psi_{LCFS} - \psi_{axis})`, so the
        conversion to :math:`\psi` is a constant factor.
        """

        if eq is None:
            eq = getattr(self, "eq", None)

        if eq is None:
            raise ValueError(
                "Kinetics.get_total_pressure_prime requires an Equilibrium 'eq' to define dp/dpsi. "
            )

        if psi_n is None:
            raise ValueError(
                "Kinetics.get_total_pressure_prime requires psi_n at which the value should be found."
            )

        if not hasattr(psi_n, "units"):
            psi_n *= units.dimensionless

        species = self._included_species(exclude_species, exclude_fast)

        dp_dpsin = sum(
            (
                s.dens(psi_n, derivative=1) * s.temp(psi_n)
                + s.dens(psi_n) * s.temp(psi_n, derivative=1)
            ).to("pascal")
            for s in species.values()
        )

        dpsi_dpsin = eq.psi_lcfs - eq.psi_axis
        return dp_dpsin / dpsi_dpsin

    @staticmethod
    def _z_profile(species, round_charge, psi_q):
        r"""
        Charge number :math:`Z(\psi_N) = q(\psi_N) / e` of a species.

        Parameters
        ----------
        species : Species
            Species whose charge to evaluate
        round_charge : bool
            Whether to round to the nearest integer, to remove floating point
            noise from charge profiles that should be integer
        psi_q : pint.Quantity
            Normalised poloidal flux, dimensionless

        Returns
        -------
        np.ndarray
            Signed charge number as a plain float array, so electrons give -1
        """
        z = species.get_charge(psi_q).to("elementary_charge").m
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

        Uses Z(psi) := q(psi)/e from Species.get_charge(psi). By default it does NOT
        force rounding, so ψ-dependent / fractional charge is preserved. If
        round_charge=True, applies pointwise integer rounding after converting to e
        (elementary_charge).

        ne(psi) = sum_{ions} Z_i(psi) n_i(psi)
        """
        sp = self.species_data

        if adjust_species not in sp:
            raise ValueError(f"{adjust_species} not found in species_data")
        if adjust_species == "electron":
            raise ValueError("Refusing to use 'electron' as adjust_species")
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
        for name in list(sp.keys()):
            if name in ("electron", adjust_species):
                continue
            s = sp[name]
            Zi = self._z_profile(s, round_charge, psi_q)
            ni = s.get_dens(psi_q).to("meter**-3").m
            charge_sum_other += Zi * ni

        Z_adj = self._z_profile(sp[adjust_species], round_charge, psi_q)
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
        preserve_zeff: bool = False,
    ):
        """
        Global merge analogue of LocalSpecies.merge_species(), implemented for global Species.

        Uses profile charge Z(psi) := q(psi)/e from Species.get_charge(psi).
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

        If preserve_zeff=True:
            Zeff(psi) = sum_ions [ n_i(psi) Z_i(psi)^2 ] / n_e(psi) is evaluated over
            *all* ions before the merge and stored as ``self.zeff_profile``, which
            LocalSpecies.from_kinetics then uses in place of its own calculation.

            This exists because neither Kinetics nor Species carries a Zeff. It is
            derived downstream in LocalSpecies.from_kinetics from whichever species
            survive, so by default a global merge silently resets Zeff towards the
            base species' charge -- merging every ion into deuterium gives Zeff = 1.
            That differs from LocalSpecies.merge_species, which by default leaves its
            cached Zeff untouched. Pass preserve_zeff=True to keep the real plasma's
            electron-ion scattering, matching the local default.

            Note there is no equivalent for the collision frequency: ``nu`` is also
            built in from_kinetics, always self-consistently with the surviving
            species, so a global merge has no stale value to correct.

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

        # Capture Zeff before anything is merged away. Neither Kinetics nor Species
        # carries a Zeff: it is derived in LocalSpecies.from_kinetics from whichever
        # species survive, so a global merge otherwise silently collapses it towards
        # the base species' charge. Summing over every ion, not just merge_set, since
        # species left out of the merge still scatter electrons.
        if preserve_zeff:
            if "electron" not in sp:
                raise ValueError(
                    "preserve_zeff requires an electron species to normalise against"
                )
            ne = sp["electron"].get_dens(psi_q).to("meter**-3").m
            zeff = np.zeros_like(psi, dtype=float)
            for name in list(self.species_names):
                if name == "electron":
                    continue
                s = sp[name]
                zeff += (
                    s.get_dens(psi_q).to("meter**-3").m
                    * self._z_profile(s, round_charge, psi_q) ** 2
                )
            zeff /= np.maximum(ne, 1e-300)
            self.zeff_profile = UnitSpline(psi_q, zeff * units.elementary_charge)

        # Collect n_i(psi), Z_i(psi), and scalar masses
        dens_arr = []
        Z_arr = []
        m_list = []

        for name in merge_set:
            s = sp[name]
            dens_arr.append(s.get_dens(psi_q).to("meter**-3").m)
            Z_arr.append(self._z_profile(s, round_charge, psi_q))
            m_list.append(_mass_kg(s))  # scalar mass, in kg

        dens_arr = np.stack(dens_arr, axis=0)  # (ns, npsi)
        Z_arr = np.stack(Z_arr, axis=0)  # (ns, npsi)
        m_arr = np.asarray(m_list, dtype=float)[:, None]  # (ns, 1)

        base = sp[base_species]

        # --- Merge density & charge ---
        if keep_base_species_z:
            Zb = self._z_profile(base, round_charge, psi_q)  # profile Z_base(psi)
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

        # Write back density spline
        base.dens = base.dens.__class__(psi_q, n_new * units("meter**-3"))

        # --- Mass merge ---
        if keep_base_species_mass:
            m_eff_profile = np.full_like(n_new, _mass_kg(base), dtype=float)
            # base.mass unchanged
        else:
            # m_eff(psi) = sum(m_i n_i)/sum(n_i)
            m_eff_profile = np.sum(m_arr * dens_arr, axis=0) / np.maximum(n_new, 1e-300)
            # Store a scalar effective mass (density-weighted over psi). Must be
            # a pint quantity: every other Species.mass is one, and the plain
            # float this used to assign silently propagated into any expression
            # that combined it with a dimensional quantity.
            w = np.maximum(n_new, 0.0)
            m_store = float(np.sum(m_eff_profile * w) / np.sum(w))
            base.mass = m_store * units.kg

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
