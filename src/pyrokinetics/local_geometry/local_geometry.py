from __future__ import annotations

"""Defines the base ``LocalGeometry`` class.

This class describes a closed flux surface in the poloidal plane. The base
class defines an arbitrary curve using plain arrays, while subclasses instead
parameterise the curve in some way, such as the Miller geometry or by Fourier
methods.
"""
import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)
from warnings import warn

import numpy as np
import pint
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import least_squares
from typing_extensions import Self

from ..equilibrium import Equilibrium
from ..factory import Factory
from ..units import Array, Float, ureg

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from ..normalisation import SimulationNormalisation as Normalisation


class Derivatives(NamedTuple):
    dRdtheta: Array
    dRdr: Array
    dZdtheta: Array
    dZdr: Array


class PackingInfo(NamedTuple):
    shapes: List[Tuple[int, ...]]
    splits: NDArray[np.int64]
    units: List[pint.Quantity]


@dataclasses.dataclass(frozen=True)
class ShapeParams:
    """Base class that manages local geometry shaping coefficients.

    ``ShapeParams`` subclasses should have have fields matching each of the
    shaping parameters for that type of local geometry. They should also
    define a class variable ``FIT_PARAMS`` which lists the names of fields
    that are to be determined by least-squares fitting.
    """

    FIT_PARAMS: ClassVar[List[str]] = []

    @property
    def FIXED_PARAMS(self) -> List[str]:
        return [
            x.name for x in dataclasses.fields(self) if x.name not in self.FIT_PARAMS
        ]

    def unpack(self) -> Tuple[NDArray[np.float64], PackingInfo]:
        """Extact fitting parameters into a flattened 1D array.

        This is needed to pass info into SciPy fitting routines.
        """
        params = [getattr(self, x) for x in self.FIT_PARAMS]
        shapes = [np.shape(x) for x in params]
        splits = np.cumsum([np.size(x) for x in params[:-1]])
        units = [ureg.Quantity(x).units for x in params]
        params_array = np.hstack([ureg.Quantity(x).magnitude for x in params])
        return params_array, PackingInfo(shapes, splits, units)

    def repack(self, params, packing_info) -> Self:
        """Rebuild self after unpacking and fitting"""
        vals = np.hsplit(params, packing_info.splits)
        shapes = packing_info.shapes
        units = packing_info.units
        fits = [np.reshape(x, s) * u for x, s, u in zip(vals, shapes, units)]
        return self.__class__(
            **{k: v for k, v in zip(self.FIT_PARAMS, fits)},
            **{k: getattr(self, k) for k in self.FIXED_PARAMS},
        )

    def __iter__(self):
        """Allows tuple-like unpacking"""
        return iter(dataclasses.astuple(self))

    def __len__(self):
        return len(dataclasses.fields(self))


@dataclasses.dataclass
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
    r"""Toroidal field at major radius, :math:`F_\psi/R`."""

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

    dpsidr: Float
    r"""Derivative of :math:`\psi` w.r.t :math:`r`"""

    R: Array
    """Major radius along the curve of the flux surface."""

    Z: Array
    """Vertical position along the curve of the flux surface."""

    b_poloidal: Array
    r""":math:`B_\theta` along the curve of the flux surface."""

    theta: Array
    r""":math:`\theta` along the curve of the flux surface."""

    dRdtheta: Array
    r"""Derivative of :math:`R` w.r.t :math:`\theta`"""

    dRdr: Array
    r"""Derivative of :math:`R` w.r.t :math:`r`"""

    dZdtheta: Array
    r"""Derivative of :math:`Z` w.r.t :math:`\theta`"""

    dZdr: Array
    r"""Derivative of :math:`Z` w.r.t :math:`r`"""

    bt_ccw: int = -1
    r"""+1 if :math:`B_\theta` is counter-clockwise, -1 otherwise."""

    ip_ccw: int = -1
    r"""+1 if the plasma current is counter-clockwise, -1 otherwise."""

    jacob: Array = dataclasses.field(init=False)
    r"""Jacobian determinant along the flux surface."""

    _already_warned: bool = dataclasses.field(init=False, repr=False)

    local_geometry: ClassVar[str] = "LocalGeometry"

    ShapeParams: ClassVar[Type] = ShapeParams

    DEFAULT_INPUTS: ClassVar[Dict[str, Any]] = {
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
    """The default parameters are the same as the GA-STD case"""

    def __post_init__(self):
        self.jacob = self.R * (self.dRdr * self.dZdtheta - self.dZdr * self.dRdtheta)
        self._already_warned = False

    @property
    def _shape_params(self) -> ShapeParams:
        data = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self.ShapeParams)
        }
        return self.ShapeParams(**data)

    @_shape_params.setter
    def _shape_params(self, params: ShapeParams) -> None:
        for field in dataclasses.fields(self.ShapeParams):
            setattr(self, field.name, getattr(params, field.name))

    def _init_with_shape_params(
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
        bt_ccw: int = DEFAULT_INPUTS["bt_ccw"],
        ip_ccw: int = DEFAULT_INPUTS["ip_ccw"],
        theta: Optional[Array] = None,
        overwrite_dpsidr: bool = True,
        **shape_params,
    ) -> None:
        """Initialise a new instance using a given set of shaping parameters.

        Used in the ``__init__`` functions of subclasses and when building from
        a global equilibrium or another local geometry.

        When building from GK input data, should always overwrite ``dpsidr``.
        When building from equilibrium data or another local geometry, should not
        overwrite ``dpsidr``.
        """
        if theta is None:
            theta = np.linspace(0, 2 * np.pi, 256)
        params = self.ShapeParams(**shape_params)

        # Get flux surface curve, B_poloidal, and derivatives
        R, Z = self._flux_surface(theta, Rmaj, Z0, rho, params)
        derivatives = self._RZ_derivatives(theta, rho, params)
        bunit_over_b0 = self._bunit_over_b0(Rmaj, Z0, rho, params)
        if overwrite_dpsidr:
            dpsidr = rho * bunit_over_b0 / q
        b_poloidal = self._b_poloidal(theta, Rmaj, Z0, rho, dpsidr, params)

        LocalGeometry.__init__(
            self,
            psi_n=psi_n,
            rho=rho,
            Rmaj=Rmaj,
            Z0=Z0,
            a_minor=a_minor,
            Fpsi=Fpsi,
            FF_prime=FF_prime,
            B0=B0,
            bunit_over_b0=bunit_over_b0,
            q=q,
            shat=shat,
            beta_prime=beta_prime,
            dpsidr=dpsidr,
            R=R,
            Z=Z,
            b_poloidal=b_poloidal,
            theta=theta,
            dRdtheta=derivatives.dRdtheta,
            dRdr=derivatives.dRdr,
            dZdtheta=derivatives.dZdtheta,
            dZdr=derivatives.dZdr,
            ip_ccw=ip_ccw,
            bt_ccw=bt_ccw,
        )
        self._shape_params = params

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

        # Calculate shaping coefficients
        params = cls._fit_shape_params(
            R, Z, b_poloidal, R_major, Zmid, rho, dpsidr, **kwargs
        )

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
            theta=theta,
            overwrite_dpsidr=False,
            **dataclasses.asdict(params),
        )

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
        # Calculate shaping coefficients
        params = cls._fit_shape_params(
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
            theta=other.theta,
            overwrite_dpsidr=False,
            **dataclasses.asdict(params),
        )

        if show_fit or axes is not None:
            result.plot_equilibrium_to_local_geometry_fit(axes=axes, show_fit=show_fit)

        return result

    def normalise(self, norms) -> None:
        """Convert to current NormalisationConvention.

        Parameters
        ----------
        norms: SimulationNormalisation
            Normalisation convention to convert to
        """

        for key, val in self._unit_mapping(norms).items():
            if val is None:
                continue

            if not hasattr(self, key):
                continue

            attribute = getattr(self, key)

            if hasattr(attribute, "units"):
                new_attr = attribute.to(val, norms.context)
            elif attribute is not None:
                new_attr = attribute * val
            else:
                new_attr = attribute

            setattr(self, key, new_attr)

    @classmethod
    def _unit_mapping(cls, norms) -> Dict[str, pint.Quantity]:
        """Return dict of units for each attribute"""

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
        shape_specific_units = cls._generate_shape_coefficients_units(norms)

        return {**general_units, **shape_specific_units}

    @classmethod
    def _generate_shape_coefficients_units(cls, norms) -> Dict[str, pint.Quantity]:
        """Return dict of units for shape parameters.

        Should be overridden by subclasses.
        """
        del norms  # unused in base class
        return {}

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
        r"""Get shaping coefficients from geometric parameters and :math:`B_\theta`.

        Should be overridden by subclasses.

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
        del R, Z, b_poloidal, Rmaj, Z0, rho, dpsidr, verbose, kwargs  # unused
        return cls.ShapeParams()

    @classmethod
    def _fit_params(
        cls,
        theta: Array,
        b_pol: Array,
        params: ShapeParams,
        R0: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        verbose: bool = False,
        max_cost: float = 1.0,
    ) -> ShapeParams:
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
        Rmaj: Float,
        Z0: Float,
        rho: Float,
        dpsidr: Float,
        params: ShapeParams,
    ) -> Array:
        r"""Calculate :math:`B_\theta` for a given set of shaping parameters"""
        R, _ = cls._flux_surface(theta, Rmaj, Z0, rho, params)
        return cls._grad_r(theta, rho, params) * np.abs(dpsidr) / R

    def get_b_poloidal(self, theta: Array) -> Array:
        r""":math:`B_\theta` at a new :math:`\theta` grid."""
        params = self._shape_params
        if len(params) == 0:
            return np.interp(theta, self.theta, self.b_poloidal)
        else:
            return self._b_poloidal(
                theta, self.Rmaj, self.Z0, self.rho, self.dpsidr, params
            )

    @classmethod
    def _flux_surface(
        cls, theta: Array, Rmaj: Float, Z0: Float, rho: Float, params: ShapeParams
    ) -> Tuple[Array, Array]:
        """Get flux surface curve for a given set of shaping parameters.

        Must be overridden by subclasses.
        """
        del theta, Rmaj, Z0, rho, params
        return (np.zeros(0), np.zeros(0))

    def get_flux_surface(self, theta: Array) -> Tuple[Array, Array]:
        r"""Get flux surface curve on a new :math:`\theta` grid."""
        params = self._shape_params
        if len(params) == 0:
            R = np.interp(theta, self.theta, self.R)
            Z = np.interp(theta, self.theta, self.R)
            return R, Z
        else:
            return self._flux_surface(theta, self.Rmaj, self.Z0, self.rho, params)

    @classmethod
    def _RZ_derivatives(
        cls, theta: Array, rho: Float, params: ShapeParams
    ) -> Derivatives:
        r"""Partial Derivatives of :math:`R(r, \theta)` and :math:`Z(r, \theta)`

        Must be overridden by subclasses.
        """
        del theta, rho, params
        return Derivatives(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

    def get_RZ_derivatives(self, theta: Array) -> Derivatives:
        r"""Partial Derivatives of :math:`R(r, \theta)` and :math:`Z(r, \theta)`"""
        params = self._shape_params
        if len(params) == 0:
            dRdtheta = np.interp(theta, self.theta, self.dRdtheta)
            dRdr = np.interp(theta, self.theta, self.dRdr)
            dZdtheta = np.interp(theta, self.theta, self.dZdtheta)
            dZdr = np.interp(theta, self.theta, self.dZdr)
            return Derivatives(
                dRdtheta=dRdtheta, dRdr=dRdr, dZdtheta=dZdtheta, dZdr=dZdr
            )
        else:
            return self._RZ_derivatives(theta, self.rho, params)

    @classmethod
    def _dLdtheta(cls, theta: Array, rho: Float, params: ShapeParams) -> Array:
        """Poloidal derivative of arc length. Used in loop integrals.

        See eqn 93 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009
        """
        derivatives = cls._RZ_derivatives(theta, rho, params)
        return np.sqrt(derivatives.dRdtheta**2 + derivatives.dZdtheta**2)

    @classmethod
    def _grad_r(cls, theta: Array, rho: Float, params: ShapeParams) -> Array:
        """MXH definition of grad r.

        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Also see eqn 39 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009
        """
        dRdtheta, dRdr, dZdtheta, dZdr = cls._RZ_derivatives(theta, rho, params)
        g_tt = dRdtheta**2 + dZdtheta**2
        return np.sqrt(g_tt) / (dRdr * dZdtheta - dRdtheta * dZdr)

    def get_grad_r(self) -> Array:
        """MXH definition of grad r.

        MXH, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model."
        Physics of Plasmas 5.4 (1998): 973-978.

        Also see eqn 39 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009
        """
        return self._grad_r(self.theta, self.rho, self._shape_params)

    @classmethod
    def _bunit_over_b0(cls, Rmaj: Float, Z0: Float, rho: Float, params: ShapeParams):
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
            R, _ = cls._flux_surface(theta, Rmaj, Z0, rho, params)
            R_grad_r = R * cls._grad_r(theta, rho, params)
            dLdtheta = cls._dLdtheta(theta, rho, params)
            # Expect dimensionless quantity
            result = ureg.Quantity(dLdtheta / R_grad_r).magnitude
            # Avoid SciPy warning when returning array with a single element
            if np.ndim(result) == 1 and np.size(result) == 1:
                result = result[0]
            return result

        integral = quad(bunit_integrand, 0.0, 2 * np.pi)[0]

        return integral * Rmaj / (2 * np.pi * rho)

    def get_f_psi(self) -> Float:
        r""":math:`f = \frac{2\pi q}{\oint \frac{dl}{R^2 B_{\theta}}}`

        See eqn 97 in Candy Plasma Phys. Control. Fusion 51 (2009) 105009
        """

        def f_psi_integrand(theta):
            R, _ = self._flux_surface(
                theta, self.Rmaj, self.Z0, self.rho, self._shape_params
            )
            b_poloidal = self._b_poloidal(
                theta, self.Rmaj, self.Z0, self.rho, self.dpsidr, self._shape_params
            )
            dLdtheta = self._dLdtheta(theta, self.rho, self._shape_params)
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

        return 2 * np.pi * q / integral

    def test_safety_factor(self):
        r""":math:`q = \frac{1}{2\pi} \oint \frac{f dl}{R^2 B_{\theta}}`"""

        def q_integrand(theta):
            R, _ = self._flux_surface(
                theta, self.Rmaj, self.Z0, self.rho, self._shape_params
            )
            b_poloidal = self._b_poloidal(
                theta, self.Rmaj, self.Z0, self.rho, self.dpsidr, self._shape_params
            )
            dLdtheta = self._dLdtheta(theta, self.rho, self._shape_params)
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

        return integral * f_psi / (2 * np.pi)

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
            [
                f"{k.name} = {getattr(self, k.name)}\n"
                for k in dataclasses.fields(self.ShapeParams)
            ]
        )
        str_list.extend([f"bunit_over_b0 = {self.bunit_over_b0}"])

        return "".join(str_list)

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

    def keys(self):
        return self.__dict__.keys()


# Create global factory for LocalGeometry objects
local_geometry_factory = Factory(super_class=LocalGeometry)
