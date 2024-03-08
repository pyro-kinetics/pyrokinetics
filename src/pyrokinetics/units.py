from contextlib import contextmanager
from typing import Optional

import numpy as np
import pint
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline


class PyroNormalisationError(Exception):
    """Exception raised when trying to convert simulation units
    requires physical reference values"""

    def __init__(self, system, units):
        super().__init__()
        self.system = system if isinstance(system, str) else system._system.name
        self.units = units

    def __str__(self):
        return (
            f"Cannot convert '{self.units}' to '{self.system}' normalisation. "
            f"Possibly '{self.system}' is missing physical reference values. "
            "You may need to load a kinetics or equilibrium file"
        )


class Normalisation:
    """
    Base class for SimulationNormalisation and ConventionNormalisation.

    Places no constraints on subclasses. Allows us to detect Pyrokinetics normalisation
    objects so that PyroQuantity can convert to normalisations as well as perform
    standard unit conversions.
    """

    pass


class PyroQuantity(pint.Quantity):
    def _replace_nan(self, value, system: Optional[str]):
        """Check bad conversions: if reference value not available,
        ``value`` will be ``NaN``"""
        if not np.isnan(value).any():
            return value
        # Special case zero, because that's always fine (except for
        # offset units, but we don't use those)
        if self == 0.0:
            return 0.0 * value.units
        raise PyroNormalisationError(system, self.units)

    def to_base_units(self, system: Optional[str] = None):
        with self._REGISTRY.as_system(system):
            value = super().to_base_units()
            return self._replace_nan(value, system)

    def _convert_simulation_units(self, norm):
        """Replace simulation units by their corresponding physical unit"""
        units = dict()
        for unit, power in self._units.items():
            if (new_unit := f"{unit}_{norm.name}") in self._REGISTRY:
                unit = new_unit
            units[unit] = power
        return self._REGISTRY.Quantity(self._magnitude, pint.util.UnitsContainer(units))

    @staticmethod
    def _is_base_unit(unit):
        """If ``unit`` is a reference unit, return the type of base unit, else return None"""
        base_units = [
            "beta_ref",
            "bref",
            "lref",
            "mref",
            "nref",
            "qref",
            "tref",
            "vref",
            "rhoref",
        ]
        for base in base_units:
            if unit.startswith(base):
                return base
        return None

    def _convert_base_units(self, norm):
        """Replace base units with those for other normalisation"""
        units = dict()
        for unit, power in self._units.items():
            if new_unit := self._is_base_unit(unit):
                unit = str(getattr(norm, new_unit))
            units[unit] = power
        return pint.util.UnitsContainer(units)

    def to(self, other=None, *contexts, **ctx_kwargs):
        """Return Quantity rescaled to other units or normalisation

        Raises `PyroNormalisationError` if value is NaN, as this
        indicates required physical reference values are missing
        """

        if isinstance(other, Normalisation):
            with self._REGISTRY.context(other.context, *contexts, **ctx_kwargs):
                as_physical = self._convert_simulation_units(other)
                value = as_physical.to(self._convert_base_units(other))
                return self._replace_nan(value, other)

        return super().to(other, *contexts, **ctx_kwargs)


class PyroUnitRegistry(pint.UnitRegistry):
    """Specialisation of `pint.UnitRegistry` that expands
    some methods to be aware of pyrokinetics normalisation objects.
    """

    _quantity_class = PyroQuantity

    def __init__(self):
        super().__init__(force_ndarray=True)

        self._on_redefinition = "ignore"

        self.define("elementary_charge = 1.602176634e−19 coulomb")
        self.define("qref = elementary_charge")

        # IMAS normalises to the actual deuterium mass, so lets add that
        # as a constant
        self.define("deuterium_mass = 3.3435837724e-27 kg")

        # We can immediately define reference masses in physical units.
        # WARNING: This might need refactoring to use a [mref] dimension
        # if we start having other possible reference masses
        self.define("mref_deuterium = deuterium_mass")
        self.define("mref_electron = electron_mass")

        # For each normalisation unit, we create a unique dimension for
        # that unit and convention
        self.define("bref_B0 = [bref]")
        self.define("lref_minor_radius = [lref]")
        self.define("nref_electron = [nref]")
        self.define("tref_electron = [tref]")
        self.define("vref_nrl = [vref] = ([tref] / [mref])**(0.5)")
        self.define(
            "rhoref_pyro = [rhoref] = ([tref] / [mref])**(0.5) * [mref] / [bref_B0])"
        )
        self.define("beta_ref_ee_B0 = [beta_ref]")

        # vrefs are related by constant, so we can always define this one
        self.define("vref_most_probable = (2**0.5) * vref_nrl")
        self.define("rhoref_gs2 = (2**0.5) * rhoref_pyro")

        # Now we define the "other" normalisation units that require more
        # information, such as bunit_over_B0 or the aspect_ratio
        self.define("bref_Bunit = NaN bref_B0")
        self.define("lref_major_radius = NaN lref_minor_radius")
        self.define("nref_deuterium = NaN nref_electron")
        self.define("tref_deuterium = NaN tref_electron")
        self.define("rhoref_unit = NaN rhoref_pyro")

        # Too many combinations of beta units, this almost certainly won't
        # scale, so just do the only one we know is used for now
        self.define("beta_ref_ee_Bunit = NaN beta_ref_ee_B0")

    def _after_init(self):
        super()._after_init()
        # Enable the Boltzmann context by default so we can always convert
        # eV to Kelvin
        self.enable_contexts("boltzmann")

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

    def _try_transform(self, src_value, src_unit, src_dim, dst_dim):
        path = pint.util.find_shortest_path(self._active_ctx.graph, src_dim, dst_dim)
        if not path:
            return None

        src = self.Quantity(src_value, src_unit)
        for a, b in zip(path[:-1], path[1:]):
            src = self._active_ctx.transform(a, b, self, src)

        return src._magnitude, src._units

    def _convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        In addition to what is done by the PlainRegistry,
        converts between units with different dimensions by following
        transformation rules defined in the context.

        Parameters
        ----------
        value :
            value
        src : UnitsContainer
            source units.
        dst : UnitsContainer
            destination units.
        inplace :
             (Default value = False)

        Returns
        -------
        callable
            converted value
        """

        if not self._active_ctx:
            return super()._convert(value, src, dst, inplace)

        src_dim = self._get_dimensionality(src)
        dst_dim = self._get_dimensionality(dst)

        # Try converting the quantity with units as given
        if converted := self._try_transform(value, src, src_dim, dst_dim):
            value, src = converted
            return super()._convert(value, src, dst, inplace)

        # That wasn't possible, so now we break up the units and see
        # if we can convert them individually.

        # These are the new units resulting from any transformations
        new_units = src

        for unit, power in src.items():
            # Here, we're assuming that the transformation is based on [dim]**1,
            # while the unit in our quantity might be e.g. its inverse
            unit_uc = pint.util.UnitsContainer({unit: 1})
            unit_dim = self._get_dimensionality(unit_uc)

            # Now we try to convert between this unit and one of the
            # destination units
            for dst_part, dst_power in dst.items():
                dst_part_uc = pint.util.UnitsContainer({dst_part: 1})
                dst_part_dim = self._get_dimensionality(dst_part_uc)
                # If we're dealing with an inverse unit, we need to
                # invert the value to get the transformation right.
                # This is a bit hacky. Assuming we don't have any
                # non-multiplicative units, we should always be able
                # to convert zero though
                force_int = False
                try:
                    value_power = value**power
                except ValueError:
                    value_power = float(value) ** power
                    force_int = True
                except ZeroDivisionError:
                    value_power = value

                if converted := self._try_transform(
                    value_power, unit_uc, unit_dim, dst_part_dim
                ):
                    value, new_unit = converted
                    # Undo any inversions
                    try:
                        value = value ** (1.0 / dst_power)
                    except ZeroDivisionError:
                        value = value
                    if force_int:
                        value = int(value)
                    # It worked, so we can replace the original unit
                    # with the transformed one
                    new_units = (
                        new_units
                        / pint.util.UnitsContainer({unit: power})
                        * (new_unit**dst_power)
                    )

        return super()._convert(value, new_units, dst, inplace)


class UnitSpline:
    """
    Unit-aware wrapper classes for 1D splines.

    Parameters
    ----------
    x: Arraylike
        x-coordinates to pass to SciPy splines, with units.
    y: ArrayLike
        y-coordinates to pass to SciPy splines, with units.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike):
        from xarray import DataArray

        if isinstance(x, DataArray):
            x = x.data
        if isinstance(y, DataArray):
            y = y.data
        self._x_units = x.units
        self._y_units = y.units

        x_mag = x.magnitude
        y_mag = y.magnitude
        # Assume x is monotonically increasing/decreasing
        if x_mag[1] > x_mag[0]:
            self._spline = InterpolatedUnivariateSpline(x_mag, y_mag)
        else:
            self._spline = InterpolatedUnivariateSpline(x_mag[::-1], y_mag[::-1])

    def __call__(self, x: ArrayLike, derivative: int = 0) -> np.ndarray:
        u = self._y_units / self._x_units**derivative
        return self._spline(x.magnitude, nu=derivative) * u


class UnitSpline2D(RectBivariateSpline):
    """
    Unit-aware wrapper classes for 2D splines.

    Parameters
    ----------
    x: pint.Quantity
        x-coordinates to pass to SciPy splines, with units.
    y: pint.Quantity
        y-coordinates to pass to SciPy splines, with units.
    z: pint.Quantity
        z-coordinates to pass to SciPy splines, with units.
    """

    def __init__(self, x, y, z):
        self._x_units = x.units
        self._y_units = y.units
        self._z_units = z.units
        self._spline = RectBivariateSpline(x.magnitude, y.magnitude, z.magnitude)

    def __call__(self, x, y, dx=0, dy=0):
        u = self._z_units / (self._x_units**dx * self._y_units**dy)
        return self._spline(x.magnitude, y.magnitude, dx=dx, dy=dy, grid=False) * u


ureg = PyroUnitRegistry()
"""Default unit registry"""
