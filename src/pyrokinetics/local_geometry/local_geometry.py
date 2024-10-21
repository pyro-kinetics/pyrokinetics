from __future__ import annotations

"""Defines the base ``LocalGeometry`` class.

This class describes a closed flux surface in the poloidal plane. The base
class defines an arbitrary curve using plain arrays, while subclasses instead
parameterise the curve in some way, such as the Miller geometry or by Fourier
methods.
"""
from collections import namedtuple
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Dict, NamedTuple, Optional, Tuple, TypeVar
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import least_squares
from typing_extensions import Self

from ..constants import pi
from ..decorators import not_implemented
from ..equilibrium import Equilibrium
from ..factory import Factory
from ..typing import ArrayLike
from ..units import Array, Float, ureg

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from ..normalisation import SimulationNormalisation as Normalisation


class Derivatives(NamedTuple):
    dRdtheta: Array
    dRdr: Array
    dZdtheta: Array
    dZdr: Array


def shape_params(fit: Iterable[str]):
    """Decorator for ``ShapeParams``, defined on :class:`LocalGeometry` subclasses.

    ``ShapeParams`` should inherit ``NamedTuple``, and have fields matching each of the
    shaping parameters for that type of local geometry.

    Decorating this class adds functions for decomposing a collection of shaping
    parameters into those determined by fitting and those that are known in advance.

    Parameters
    ----------
    fit
        The fields that are determined by least squares fitting.
    """

    def decorator(cls):
        # Add internal named tuple for fitting parameters
        cls.FitParams = namedtuple("FitParams", fit)

        # Add additional named tuple for those that aren't included in fitting
        fixed = [x for x in cls._fields if x not in fit]
        cls.FixedParams = namedtuple("FixedParams", fixed)

        # Add utility type that can be used to re-assemble a ShapeParams after
        # it has been decomposed into fixed and fitting parameters
        cls.PackingInfo = namedtuple("PackingInfo", ["shapes", "splits", "units"])

        # Add method to extact fitting parameters into a flattened 1D array.
        # This is needed to pass info into SciPy fitting routines.
        def unpack(self) -> Tuple[NDArray[np.float64], cls.PackingInfo]:
            params = [getattr(self, x) for x in self.FitParams._fields]
            shapes = [np.shape(x) for x in params]
            splits = np.cumsum([np.size(x) for x in params[:-1]])
            units = [ureg.Quantity(x).units for x in params]
            params_array = np.hstack([ureg.Quantity(x).magnitude for x in params])
            return params_array, self.PackingInfo(shapes, splits, units)

        cls.unpack = unpack

        # Add method to rebuild self after fitting
        def repack(self, params, packing_info) -> cls:
            vals = np.hsplit(params, packing_info.splits)
            shapes = packing_info.shapes
            units = packing_info.units
            fits = [np.reshape(x, s) * u for x, s, u in zip(vals, shapes, units)]
            fits = self.FitParams(*fits)
            return self.__class__(
                **{k: getattr(fits, k) for k in self.FitParams._fields},
                **{k: getattr(self, k) for k in self.FixedParams._fields},
            )

        cls.repack = repack

        return cls

    return decorator


class LocalGeometry:

    psi_n: Float
    r"""Normalised :math:`\Psi`"""

    rho: Float
    r"""Normalised minor radius.

    :math:`r/a`, where :math:`a` is the minor radius of the last closed flux
    surface.
    """

    a_minor: Float
    """Minor radius of the last closed flux surface."""

    Rmaj: Float
    r"""Normalised major radius.

    :math:`R/a`, where :math:`a` is the minor radius of the last closed flux
    surface.
    """

    Z0: Float
    r"""Normalised vertical position of midpoint.

    :math:`Z_{mid}/a`, where :math:`a` is the minor radius of the last closed
    flux surface.
    """

    Fpsi: Float
    """Toroidal field function"""

    FF_prime: Float
    r"""Toroidal field function multiplies by its derivative w.r.t :math:`\psi`"""

    B0: Float
    r"""Toroidal field at major radius.

    :math:`F_\psi/R`
    """

    bunit_over_b0: Float
    r"""Ratio of GACODE normalising field to :math:`B_0`.

    :math:`B_{unit}=q/r \partial \psi/\partial r`
    """

    dpsidr: Float
    r""":math:`\partial \psi / \partial r`"""

    q: Float
    """Safety factor"""

    shat: Float
    r"""Magnetic shear :math:`r/q \partial q/ \partial r`"""

    beta_prime: Float
    r""":math:`\beta = 2 \mu_0 \partial p \partial \rho 1/B_0^2`"""

    R: Array
    """Fitted ``R`` data"""

    Z: Array
    """Fitted ``Z`` data"""

    b_poloidal: Array
    """Fitted ``B_poloidal`` data"""

    theta: Array
    """Fitted theta data"""

    dRdtheta: Array
    r"""Derivative of fitted :math:`R` w.r.t :math:`\theta`"""

    dRdr: Array
    r"""Derivative of fitted :math:`R` w.r.t :math:`r`"""

    dZdtheta: Array
    r"""Derivative of fitted :math:`Z` w.r.t :math:`\theta`"""

    dZdr: Array
    r"""Derivative of fitted :math:`Z` w.r.t :math:`r`"""

    @shape_params(fit=[])
    class ShapeParams(NamedTuple):
        """The parameters used to describe a curve. Implemented by subclasses."""

        pass

    _ShapeParams = TypeVar("_ShapeParams")

    DEFAULT_INPUTS: ClassVar[Dict[str, float]] = {
        "psi_n": 0.5,
        "rho": 0.5,
        "Rmaj": 3.0,
        "Z0": 0.0,
        "a_minor": 1.0,
        "Fpsi": 0.0,
        "FF_prime": 0.0,
        "B0": 0.0,
        "q": 2.0,
        "shat": 1.0,
        "beta_prime": 0.0,
        "dpsidr": 1.0,
        "bt_ccw": -1,
        "ip_ccw": -1,
    }

    def __init__(
        self,
        psi_n: Float = DEFAULT_INPUTS["psi_n"],
        rho: Float = DEFAULT_INPUTS["rho"],
        Rmaj: Float = DEFAULT_INPUTS["Rmaj"],
        Z0: Float = DEFAULT_INPUTS["Z0"],
        a_minor: Float = DEFAULT_INPUTS["a_minor"],
        Fpsi: Float = DEFAULT_INPUTS["Fpsi"],
        FF_prime: Float = DEFAULT_INPUTS["FF_prime"],
        B0: Float = DEFAULT_INPUTS["B0"],
        q: Float = DEFAULT_INPUTS["q"],
        shat: Float = DEFAULT_INPUTS["shat"],
        beta_prime: Float = DEFAULT_INPUTS["beta_prime"],
        dpsidr: Float = DEFAULT_INPUTS["dpsidr"],
        bt_ccw: float = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: float = DEFAULT_INPUTS["ip_ccw"],
    ):
        """General geometry object representing local fit parameters."""
        self.psi_n = psi_n
        self.rho = rho
        self.Rmaj = Rmaj
        self.Z0 = Z0
        self.a_minor = a_minor
        self.Fpsi = Fpsi
        self.FF_prime = FF_prime
        self.B0 = B0
        self.q = q
        self.shat = shat
        self.beta_prime = beta_prime
        self.dpsidr = dpsidr
        self.bt_ccw = bt_ccw
        self.ip_ccw = ip_ccw

        self._already_warned = False

    def default(self):
        """Default parameters for geometry.

        Applies to all subclasses, as each define their own ``__init__``
        function and ``DEFAULT_INPUTS`` class variable.

        The default parameters are the same as the GA-STD case
        """
        self.__init__(**self.DEFAULT_INPUTS)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __setattr__(self, key, value):
        if value is None:
            super().__setattr__(key, value)

        if hasattr(self, key):
            attr = getattr(self, key)
            if hasattr(attr, "units") and not hasattr(value, "units"):
                value *= attr.units
                if not self._already_warned and str(attr.units) != "dimensionless":
                    warn(
                        f"missing unit from {key}, adding {attr.units}. "
                        "To suppress this warning, specify units. "
                        f"Will maintain units if not specified from now on"
                    )
                    self._already_warned = True
        super().__setattr__(key, value)

    @property
    def _shape_params(self) -> ShapeParams:
        data = {field: getattr(self, field) for field in self.ShapeParams._fields}
        return self.ShapeParams(**data)

    @_shape_params.setter
    def _shape_params(self, params: ShapeParams) -> None:
        for field in self.ShapeParams._fields:
            setattr(self, field, getattr(params, field))

    def keys(self):
        return self.__dict__.keys()

    @classmethod
    def from_global_eq(
        cls,
        eq: Equilibrium,
        psi_n: float,
        norms: Normalisation,
        show_fit: bool = False,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
        **kwargs,
    ) -> Self:
        """Creates a :class:`LocalGeometry` from an :class:`Equilibrium`

        Parameters
        ----------
        eq
            The global equilibrium from which a flux surface should be
            extracted.
        psi_n
            The magnetic flux funciton :math:`psi` at which to extract a flux
            surface. Normalised to take the value of 0 on the magnetic axis and
            1 on the last closed flux surface.
        norms
            The system of normalised units to use.
        show_fit
            If ``True``, plots the resulting fit using Matplotlib.
        axes
            Axes on which to plot if ``show_fit`` is ``True``. If supplied, the
            plot will not be shown, and it is up to the user to call
            ``plt.show()``, ``plt.savefig()`` or similar.  If ``axes`` is
            ``None``, a new set of axes are created and the plot is shown to
            the caller.
        **kwargs
            Additional arguments that are passed to the fitting routine.
        """

        # TODO FluxSurface is COCOS 11, this uses something else. Here we switch from
        # a clockwise theta grid to a counter-clockwise one, and divide any psi
        # quantities by 2 pi
        fs = eq.flux_surface(psi_n=psi_n)
        # Convert to counter-clockwise, discard repeated endpoint
        R = fs["R"].data[:0:-1]
        Z = fs["Z"].data[:0:-1]
        b_poloidal = fs["B_poloidal"].data[:0:-1]

        R_major = fs.R_major
        rho = fs.r_minor
        Zmid = fs.Z_mid

        theta = np.arctan2(Z - Zmid, R - R_major) % (2 * np.pi)

        Fpsi = fs.F
        B0 = Fpsi / R_major
        FF_prime = fs.FF_prime * (2 * np.pi)

        dpsidr = fs.psi_gradient / (2 * np.pi)
        q = fs.q
        shat = fs.magnetic_shear
        dpressure_drho = fs.pressure_gradient * fs.a_minor

        # beta_prime needs special treatment...
        beta_prime = (2 * ureg.mu0 * dpressure_drho / B0**2).to_base_units().m

        # Store Equilibrium values
        local_geometry = cls(
            psi_n=psi_n,
            rho=rho,
            Rmaj=R_major,
            Z0=Zmid,
            a_minor=fs.a_minor,
            Fpsi=Fpsi,
            FF_prime=FF_prime,
            B0=B0,
            q=q,
            shat=shat,
            beta_prime=beta_prime,
            dpsidr=dpsidr,
            ip_ccw=np.sign(q / B0),
            bt_ccw=np.sign(B0),
        )

        # Calculate shaping coefficients
        local_geometry._shape_params = cls._fit_shape_params(
            R, Z, b_poloidal, R_major, Zmid, rho, dpsidr, **kwargs
        )

        local_geometry.R, local_geometry.Z = local_geometry.get_flux_surface(theta)
        local_geometry.b_poloidal = local_geometry.get_b_poloidal(theta)
        local_geometry.theta = theta
        dRdtheta, dRdr, dZdtheta, dZdr = local_geometry.get_RZ_derivatives(
            local_geometry.theta
        )
        local_geometry.dRdtheta = dRdtheta
        local_geometry.dRdr = dRdr
        local_geometry.dZdtheta = dZdtheta
        local_geometry.dZdr = dZdr
        local_geometry.jacob = local_geometry.R * (dRdr * dZdtheta - dZdr * dRdtheta)

        # Bunit for GACODE codes
        local_geometry.bunit_over_b0 = local_geometry.get_bunit_over_b0()

        if show_fit or axes is not None:
            local_geometry.plot_equilibrium_to_local_geometry_fit(
                axes=axes, show_fit=show_fit
            )

        # Set references and normalise
        norms.set_bref(local_geometry)
        norms.set_lref(local_geometry)
        local_geometry.normalise(norms)
        return local_geometry

    @classmethod
    def from_local_geometry(
        cls,
        other: Self,
        verbose: bool = False,
        show_fit: bool = False,
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
        **kwargs,
    ) -> Self:
        r"""Create a new ``LocalGeometry`` from another or a subclass.

        Gradients in shaping parameters are fitted from the poloidal field.

        Parameters
        ----------
        local_geometry
            ``LocalGeometry`` or subclass to fit to.
        verbose
            Print more data to terminal when performing a fit.
        show_fit
            If ``True``, plots the resulting fit using Matplotlib.
        axes
            Axes on which to plot if ``show_fit`` is ``True``. If supplied, the
            plot will not be shown, and it is up to the user to call
            ``plt.show()``, ``plt.savefig()`` or similar.  If ``axes`` is
            ``None``, a new set of axes are created and the plot is shown to
            the caller.
        """

        # Load in parameters that
        result = cls(
            psi_n=other.psi_n,
            rho=other.rho,
            Rmaj=other.Rmaj,
            Z0=other.Z0,
            a_minor=other.a_minor,
            Fpsi=other.Fpsi,
            FF_prime=other.FF_prime,
            B0=other.B0,
            q=other.q,
            shat=other.shat,
            beta_prime=other.beta_prime,
            dpsidr=other.dpsidr,
            ip_ccw=other.ip_ccw,
            bt_ccw=other.bt_ccw,
        )

        result._shape_params = cls._fit_shape_params(
            other.R,
            other.Z,
            other.b_poloidal,
            other.Rmaj,
            other.Z0,
            other.rho,
            other.dpsidr,
            verbose=verbose,
            **kwargs,
        )
        result.R, result.Z = result.get_flux_surface(other.theta)
        result.b_poloidal = result.get_b_poloidal(other.theta)
        result.theta = other.theta
        dRdtheta, dRdr, dZdtheta, dZdr = result.get_RZ_derivatives(result.theta)
        result.dRdtheta = dRdtheta
        result.dRdr = dRdr
        result.dZdtheta = dZdtheta
        result.dZdr = dZdr

        # Bunit for GACODE codes
        result.bunit_over_b0 = result.get_bunit_over_b0()

        if show_fit or axes is not None:
            result.plot_equilibrium_to_local_geometry_fit(axes=axes, show_fit=show_fit)

        return result

    @classmethod
    def from_gk_data(cls, **kwargs):
        """
        Initialise from data gathered from GKCode object, and additionally set
        bunit_over_b0
        """
        local_geometry = cls(**kwargs)

        # Values are not yet normalised
        local_geometry.bunit_over_b0 = local_geometry.get_bunit_over_b0()

        # Get dpsidr from Bunit/B0
        local_geometry.dpsidr = (
            local_geometry.bunit_over_b0 / local_geometry.q * local_geometry.rho
        )

        # This is arbitrary, maybe should be a user input
        theta = np.linspace(0, 2 * pi, 256)

        local_geometry.R, local_geometry.Z = local_geometry.get_flux_surface(theta)
        local_geometry.b_poloidal = local_geometry.get_b_poloidal(theta)
        local_geometry.theta = theta

        (
            local_geometry.dRdtheta,
            local_geometry.dRdr,
            local_geometry.dZdtheta,
            local_geometry.dZdr,
        ) = local_geometry.get_RZ_derivatives(local_geometry.theta)

        return local_geometry

    def normalise(self, norms):
        """
        Convert LocalGeometry Parameters to current NormalisationConvention
        Note this creates the attribute unit_mapping which is used to apply
        units to the LocalGeometry object
        Parameters
        ----------
        norms : SimulationNormalisation
            Normalisation convention to convert to

        """
        self._generate_local_geometry_units(norms)

        for key, val in self.unit_mapping.items():
            if val is None:
                continue

            if not hasattr(self, key):
                continue

            attribute = getattr(self, key)

            if hasattr(attribute, "units"):
                new_attr = attribute.to(val, norms.context)
            elif attribute is not None:
                new_attr = attribute * val

            setattr(self, key, new_attr)

    def _generate_local_geometry_units(self, norms):
        """
        Generate dictionary for the different units of each attribute

        Parameters
        ----------
        norms

        Returns
        -------

        """
        general_units = {
            "psi_n": ureg.dimensionless,
            "rho": norms.lref,
            "Rmaj": norms.lref,
            "a_minor": norms.lref,
            "Z0": norms.lref,
            "B0": norms.bref,
            "q": ureg.dimensionless,
            "shat": ureg.dimensionless,
            "Fpsi": norms.bref * norms.lref,
            "FF_prime": norms.bref,
            "dRdtheta": norms.lref,
            "dZdtheta": norms.lref,
            "dRdr": ureg.dimensionless,
            "dZdr": ureg.dimensionless,
            "dpsidr": norms.lref * norms.bref,
            "jacob": norms.lref**2,
            "R": norms.lref,
            "Z": norms.lref,
            "b_poloidal": norms.bref,
            "beta_prime": norms.bref**2 / norms.lref,
            "bunit_over_b0": ureg.dimensionless,
            "bt_ccw": ureg.dimensionless,
            "ip_ccw": ureg.dimensionless,
        }

        # Make shape specific units
        shape_specific_units = self._generate_shape_coefficients_units(norms)

        self.unit_mapping = {**general_units, **shape_specific_units}

    @classmethod
    def _fit_shape_params(
        cls,
        R: Array,
        Z: Array,
        b_poloidal: Array,
        Rmaj: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        verbose: bool = False,
        **kwargs,
    ) -> ShapeParams:
        r"""
        Calculates shaping coefficients from geometric parameters and :math:`B_\theta`.

        Should be overridden in subclasses.

        Parameters
        ----------
        R
            R for the given flux surface
        Z
            Z for the given flux surface
        b_poloidal
            :math:`B_\theta` for the given flux surface
        Rmaj
            Major radius of the centre of the flux surface
        Z0
            Vertical height of the centre of the flux surface
        rho
            Normalised minor radius of the flux surface
        dpsidr
            :math:`\partial \psi / \partial r`
        verbose
            Controls verbosity
        """
        del R, Z, b_poloidal, Rmaj, Z0, rho, dpsidr, verbose  # inputs unused
        return cls.ShapeParams()

    @not_implemented
    def _generate_shape_coefficients_units(self, norms):
        """
        Converts shaping coefficients to current normalisation
        Parameters
        ----------
        norms

        Returns
        -------

        """
        pass

    @not_implemented
    def get_RZ_derivatives(self, theta: Array, params=None):
        pass

    def get_grad_r(
        self,
        theta: ArrayLike,
        params=None,
    ) -> np.ndarray:
        """
        MXH definition of grad r from
        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Also see eqn 39 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009


        Parameters
        ----------
        kappa: Scalar
            elongation
        shift: Scalar
            Shafranov shift
        theta: ArrayLike
            Array of theta points to evaluate grad_r on

        Returns
        -------
        grad_r : Array
            grad_r(theta)
        """

        dRdtheta, dRdr, dZdtheta, dZdr = self.get_RZ_derivatives(theta, params)

        g_tt = dRdtheta**2 + dZdtheta**2

        grad_r = np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

        return grad_r

    @classmethod
    def _fit_params(
        cls,
        theta: Array,
        b_pol: Array,
        params: _ShapeParams,
        R0: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        verbose: bool = False,
        max_cost: float = 1.0,
    ) -> _ShapeParams:
        """Generate a new set of fitting parameters.

        Uses least squares minimisation of the poloidal field.

        Parameters
        ----------
        theta
            Angular displacement along the curve.
        b_pol
            Poloidal component of the magnetic field.
        params
            Starting guesses for the fitted parameters.
        """
        # Unpack params into a 1D array, keeping track of how to rebuild afterwards
        fit_params, packing_info = params.unpack()

        # Fitting function to be passed to scipy.least_squares
        def residuals(fit_params, shape_params, packing, R0, Z0, rho, dpsidr):
            params = shape_params.repack(fit_params, packing)
            b_pol_new = cls._b_poloidal(theta, R0, Z0, rho, dpsidr, params)
            return ureg.Quantity(b_pol - b_pol_new).magnitude

        args = (params, packing_info, R0, Z0, rho, dpsidr)
        result = least_squares(residuals, fit_params, args=args)
        if not result.success:
            msg = f"Least squares fitting in {cls.__name__} failed: {result.message}"
            raise RuntimeError(msg)
        if (cost := result.cost) > max_cost:
            msg = f"Poor least squares fitting in {cls.__name__}, residual: {cost}"
            warn(msg)
        # TODO verbose
        del verbose

        # Pack fits back into named tuple
        return params.repack(result.x, packing_info)

    @classmethod
    def _b_poloidal(
        cls,
        theta: Array,
        R0: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        params: NamedTuple,
    ) -> Array:
        """Calculate :math:`b_{pol}` for a given set of shaping parameters"""
        R, _ = cls._flux_surface(theta, R0, Z0, rho, params)
        return cls._grad_r(theta, rho, params) * np.abs(dpsidr) / R

    @classmethod
    def _flux_surface(
        cls, theta: Array, R0: Float, Z0: Float, rho: Float, params: NamedTuple
    ) -> Tuple[Array, Array]:
        """Get flux surface curve for a given set of shaping parameters.

        Must be overridden by subclasses.
        """
        del theta, R0, Z0, rho, params
        return (np.zeros(0), np.zeros(0))

    @classmethod
    def _grad_r(cls, theta: Array, rho: Float, params: NamedTuple) -> Array:
        """MXH definition of grad r.

        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Also see eqn 39 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009
        """
        dRdtheta, dRdr, dZdtheta, dZdr = cls._RZ_derivatives(theta, rho, params)
        g_tt = dRdtheta**2 + dZdtheta**2
        return np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: NamedTuple
    ) -> Derivatives:
        r"""Partial Derivatives of :math:`R(r, \theta)` and :math:`Z(r, \theta)`

        Must be overridden by subclasses, and return:

        - :math:`\frac{\partial R}{\partial \theta}`
        - :math:`\frac{\partial R}{\partial r}`
        - :math:`\frac{\partial Z}{\partial \theta}`
        - :math:`\frac{\partial Z}{\partial r}`
        """
        del theta, rho, params
        return Derivatives(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

    def get_b_poloidal(self, theta: ArrayLike, params=None) -> np.ndarray:
        r"""
        Returns Miller prediction for get_b_poloidal given flux surface parameters

        Parameters
        ----------
        params : List
            List with LocalGeometry type specific values

        Returns
        -------
        local_geometry_b_poloidal : Array
            Array of get_b_poloidal from Miller fit
        """

        R, Z = self.get_flux_surface(theta)

        return np.abs(self.dpsidr) / R * self.get_grad_r(theta, params)

    def get_dLdtheta(self, theta):
        """
        Returns dLdtheta used in loop integrals

        See eqn 93 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009

        Parameters
        ----------
        theta : ArrayLike
            Poloidal angle to evaluate at

        Returns
        -------
        dLdtheta : Poloidal derivative of Arclength
        """

        dRdtheta, dRdr, dZdtheta, dZdr = self.get_RZ_derivatives(theta)

        return np.sqrt(dRdtheta**2 + dZdtheta**2)

    def get_bunit_over_b0(self):
        r"""
        Get Bunit/B0 using q and loop integral of Bp

        :math:`\frac{B_{unit}}{B_0} = \frac{R_0}{2\pi r_{minor}} \oint \frac{a}{R} \frac{dl_N}{\nabla r}`

        where :math:`dl_N = \frac{dl}{a_{minor}}` coming from the normalising a_minor

        See eqn 97 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009

        Returns
        -------
        bunit_over_b0 : Float
             :math:`\frac{B_{unit}}{B_0}`

        """

        def bunit_integrand(theta):
            R, _ = self.get_flux_surface(theta)
            R_grad_r = R * self.get_grad_r(theta)
            dLdtheta = self.get_dLdtheta(theta)
            # Expect dimensionless quantity
            result = ureg.Quantity(dLdtheta / R_grad_r).magnitude
            # Avoid SciPy warning when returning array with a single element
            if np.ndim(result) == 1 and np.size(result) == 1:
                result = result[0]
            return result

        integral = quad(bunit_integrand, 0.0, 2 * np.pi)[0]

        return integral * self.Rmaj / (2 * pi * self.rho)

    def get_f_psi(self):
        r"""
        Calculate safety factor from b poloidal field, R, Z and q
        :math:`f = \frac{2\pi q}{\oint \frac{dl}{R^2 B_{\theta}}}`

        See eqn 97 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009

        Returns
        -------
        f : Float
            Prediction for :math:`f_\psi` from B_poloidal
        """

        def f_psi_integrand(theta):
            R, _ = self.get_flux_surface(theta)
            b_poloidal = self.get_b_poloidal(theta)
            dLdtheta = self.get_dLdtheta(theta)
            result = ureg.Quantity(dLdtheta / (R**2 * b_poloidal)).magnitude
            # Avoid SciPy warning when returning array with a single element
            if np.ndim(result) == 1 and np.size(result) == 1:
                result = result[0]
            return result

        bref = self.b_poloidal.units
        lref = self.R.units

        @ureg.wraps(bref**-1 * lref**-1, (), False)
        def get_integral():
            return quad(f_psi_integrand, 0.0, 2 * np.pi)[0]

        integral = get_integral()
        q = self.q

        return 2 * pi * q / integral

    def test_safety_factor(self):
        r"""
        Calculate safety factor from LocalGeometry object b poloidal field
        :math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`

        Returns
        -------
        q : Float
            Prediction for :math:`q` from fourier B_poloidal
        """

        def q_integrand(theta):
            R, _ = self.get_flux_surface(theta)
            b_poloidal = self.get_b_poloidal(theta)
            dLdtheta = self.get_dLdtheta(theta)
            result = ureg.Quantity(dLdtheta / (R**2 * b_poloidal)).magnitude
            # Avoid SciPy warning when returning array with a single element
            if np.ndim(result) == 1 and np.size(result) == 1:
                result = result[0]
            return result

        f_psi = self.Fpsi
        bref = self.b_poloidal.units
        lref = self.R.units

        @ureg.wraps(bref**-1 * lref**-1, (), False)
        def get_integral():
            return quad(q_integrand, 0.0, 2 * np.pi)[0]

        integral = get_integral()

        return integral * f_psi / (2 * pi)

    def plot_equilibrium_to_local_geometry_fit(
        self, axes: Optional[Tuple[plt.Axes, plt.Axes]] = None, show_fit=False
    ):
        import matplotlib.pyplot as plt

        # Get flux surface and b_poloidal
        R_fit, Z_fit = self.get_flux_surface(theta=self.theta)

        bpol_fit = self.get_b_poloidal(
            theta=self.theta,
        )

        # Set up plot if one doesn't exist already
        if axes is None:
            fig, axes = plt.subplots(1, 2)
        else:
            fig = axes[0].get_figure()

        # Plot R, Z
        axes[0].plot(self.R.m, self.Z.m, label="Data")
        axes[0].plot(R_fit.m, Z_fit.m, "--", label="Fit")
        axes[0].set_xlabel("R")
        axes[0].set_ylabel("Z")
        axes[0].set_aspect("equal")
        axes[0].set_title(f"Fit to flux surface for {self.local_geometry}")
        axes[0].legend()
        axes[0].grid()

        # Plot Bpoloidal
        axes[1].plot(self.theta.m, self.b_poloidal.m, label="Data")
        axes[1].plot(self.theta.m, bpol_fit.m, "--", label="Fit")
        axes[1].legend()
        axes[1].set_xlabel("theta")
        axes[1].set_title(f"Fit to poloidal field for {self.local_geometry}")
        axes[1].set_ylabel("Bpol")
        axes[1].grid()

        if show_fit:
            plt.show()
        else:
            return fig, axes

    def __repr__(self):
        str_list = [f"{type(self)}(\n" f"type  = {self.local_geometry},\n"]
        str_list.extend([f"{k} = {getattr(self, k)}\n" for k in self.DEFAULT_INPUTS])
        str_list.extend(
            [f"{k} = {getattr(self, k)}\n" for k in self._shape_coefficient_names()]
        )
        str_list.extend([f"bunit_over_b0 = {self.bunit_over_b0}"])

        return "".join(str_list)


# Create global factory for LocalGeometry objects
local_geometry_factory = Factory(super_class=LocalGeometry)
