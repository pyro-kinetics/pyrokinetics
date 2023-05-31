import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from ..units import ureg as units

# Define basic units, COCOS 11
eq_units = {
    "len": units.meter,
    "psi": units.weber,
    "F": units.meter * units.tesla,
    "p": units.pascal,
    "q": units.dimensionless,
    "B": units.tesla,
    "I": units.ampere,
}

# Add derivatives
eq_units["F_prime"] = eq_units["F"] / eq_units["psi"]
eq_units["FF_prime"] = eq_units["F"] ** 2 / eq_units["psi"]
eq_units["p_prime"] = eq_units["p"] / eq_units["psi"]
eq_units["q_prime"] = eq_units["q"] / eq_units["psi"]
eq_units["len_prime"] = eq_units["len"] / eq_units["psi"]


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
        self._x_units = x.units
        self._y_units = y.units
        self._spline = InterpolatedUnivariateSpline(x.magnitude, y.magnitude)

    def __call__(self, x: ArrayLike, derivative: int = 0) -> np.ndarray:
        u = self._y_units / self._x_units ** derivative
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
        u = self._z_units / (self._x_units ** dx * self._y_units ** dy)
        return self._spline(x.magnitude, y.magnitude, dx=dx, dy=dy, grid=False) * u
