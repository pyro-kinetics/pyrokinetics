from ..units import ureg as units
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

# Define basic units
eq_units = {
    "len": units.meter,
    "psi": units.weber / units.rad,
    "f": units.meter * units.tesla,
    "p": units.pascal,
    "q": units.dimensionless,
    "b": units.tesla,
}

# Add derivatives
eq_units["f_prime"] = eq_units["f"] / eq_units["psi"]
eq_units["ff_prime"] = eq_units["f"] ** 2 / eq_units["psi"]
eq_units["p_prime"] = eq_units["p"] / eq_units["psi"]
eq_units["q_prime"] = eq_units["q"] / eq_units["psi"]
eq_units["len_prime"] = eq_units["len"] / eq_units["psi"]


class UnitSpline(InterpolatedUnivariateSpline):
    """
    Unit-aware wrapper classes for 1D splines.

    WARNING: Do not use functions like 'derviative()' or 'partial_derivative()', as
    these create a new scipy spline, and unit info will be lost.

    Parameters
    ----------
    x: pint.Quantity
        x-coordinates to pass to SciPy splines, with units.
    y: pint.Quantity
        y-coordinates to pass to SciPy splines, with units.
    args*
        Positional arguments to pass to SciPy splines.
    kwargs**
        Keyword arugments to pass to SciPy splines.
    """

    def __init__(self, x, y, *args, **kwargs):
        self.x_units = x.units
        self.y_units = y.units
        super().__init__(x.magnitude, y.magnitude, *args, **kwargs)

    def __call__(self, x, nu=0, **kwargs):
        u = self.y_units
        if nu:
            u /= self.x_units**nu
        return super().__call__(x.magnitude, nu=nu, **kwargs) * u


class UnitSpline2D(RectBivariateSpline):
    """
    Unit-aware wrapper classes for 2D splines.

    WARNING: Do not use functions like 'derviative()' or 'partial_derivative()', as
    these create a new scipy spline, and unit info will be lost.

    Parameters
    ----------
    x: pint.Quantity
        x-coordinates to pass to SciPy splines, with units.
    y: pint.Quantity
        y-coordinates to pass to SciPy splines, with units.
    z: pint.Quantity
        z-coordinates to pass to SciPy splines, with units.
    args*
        Positional arguments to pass to SciPy splines.
    kwargs**
        Keyword arugments to pass to SciPy splines.
    """

    def __init__(self, x, y, z, *args, **kwargs):
        self.x_units = x.units
        self.y_units = y.units
        self.z_units = z.units
        super().__init__(x.magnitude, y.magnitude, z.magnitude, *args, **kwargs)

    def __call__(self, x, y, dx=0, dy=0, **kwargs):
        u = self.z_units
        if dx:
            u /= self.x_units**dx
        if dy:
            u /= self.y_units**dy
        return super().__call__(x.magnitude, y.magnitude, dx=dx, dy=dy, **kwargs) * u
