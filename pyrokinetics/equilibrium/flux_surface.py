from __future__ import annotations  # noqa
import re
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from ..dataset_wrapper import DatasetWrapper
from ..units import ureg as units
from .utils import eq_units


@units.wraps(units.meter, [units.m, units.m, units.weber / units.rad] * 2, strict=False)
def _flux_surface_contour(
    R: ArrayLike,
    Z: ArrayLike,
    psi_RZ: ArrayLike,
    R_axis: float,
    Z_axis: float,
    psi: float,
) -> np.ndarray:
    r"""
    Given linearly-spaced RZ coordinates and psi at these positions, returns the R and Z
    coordinates of a contour at given psi. Describes the path of a single magnetic flux
    surface within a tokamak. Aims to return the closest closed contour to the position
    ``(R_axis, Z_axis)``.

    Parameters
    ----------
    R: ArrayLike
        Linearly spaced and monotonically increasing 1D grid of major radius
        coordinates, i.e. the radial distance from the central column of a tokamak.
    Z: ArrayLike
        Linearly spaced and monotonically increasing 1D grid of z-coordinates describing
        the distance from the midplane of a tokamak.
    psi_RZ: ArrayLike
        2D grid of :math:`\psi`, the poloidal magnetic flux function, over the range
        (R,Z).
    R_axis: float
        R position of the magnetic axis.
    Z_axis: float
        Z position of the magnetic axis.
    psi: float
        The choice of :math:`\psi` on which to fit a contour.

    Returns
    -------
    np.ndarray
        2D array containing R and Z coordinates of the flux surface contour. Indexing
        with [0,:] gives a 1D array of R coordinates, while [1,:] gives a 1D array of
        Z coordinates. The endpoints are repeated, so [:,0] == [:,-1].

    Raises
    ------
    ValueError
        If the shapes of ``R``, ``Z``, and ``psi_RZ`` don't match.
    RuntimeError
        If no flux surface contours could be found.

    Warnings
    --------
    For performance reasons, this function does not check that ``R`` or ``Z`` are
    linearly spaced or monotonically increasing. If this condition is not upheld, the
    results are undefined.
    """

    # Check some basic conditions on R, Z, psi_RZ
    R = np.asfarray(R)
    Z = np.asfarray(Z)
    psi_RZ = np.asfarray(psi_RZ)
    if len(R.shape) != 1:
        raise ValueError("The grid R should be 1D.")
    if len(Z.shape) != 1:
        raise ValueError("The grid Z should be 1D.")
    if not np.array_equal(psi_RZ.shape, (len(R), len(Z))):
        raise ValueError(
            f"The grid psi_RZ has shape {psi_RZ.shape}. "
            f"It should have shape {(len(R), len(Z))}."
        )

    # Get contours, raising error if none are found
    contours_raw = find_contours(psi_RZ, psi)
    if not contours_raw:
        raise RuntimeError(f"Could not find flux surface contours for psi={psi}")

    # Normalise to RZ grid
    # Normalisation assumes R and Z are linspace grids with a positive spacing.
    # The raw contours have a range of 0 to len(x)-1, where x is r or z.
    scaling = np.array([(x[-1] - x[0]) / (len(x) - 1) for x in (R, Z)])
    RZ_min = np.array([R[0], Z[0]])
    contours = [contour * scaling + RZ_min for contour in contours_raw]

    # Find the contour that is, on average, closest to the magnetic axis, as this
    # procedure may find additional open contours outside the last closed flux surface.
    if len(contours) > 1:
        RZ_axis = np.array([[R_axis, Z_axis]])
        mean_dist = [np.mean(np.linalg.norm(c - RZ_axis, axis=1)) for c in contours]
        contour = contours[np.argmin(mean_dist)]
    else:
        contour = contours[0]

    # Adjust the contour arrays so that we begin at the OMP (outside midplane)
    omp_idx = np.argmax(contour[0, :])
    contour = np.roll(contour, -omp_idx, axis=0)

    # Return transpose so we have array of [[Rs...],[Zs...]]
    return contour.T


class FluxSurface(DatasetWrapper):
    r"""
    Information about a single flux surface of a tokamak plasma equilibrium. Users are
    not expected to initialise ``FluxSurface`` objects directly, but instead should
    generate them from ``Equilibrium`` objects. ``FluxSurface`` is used as an
    intermediate object when generating ``LocalGeometry`` objects from global plasma
    equilibria. For more information, see the 'Notes' section for ``Equilibrium``.

    Parameters
    ----------

    R: ArrayLike, units [meter]
        1D grid of major radius coordinates describing the path of the flux surface.
        The endpoints should be repead.
    Z: ArrayLike, units [meter]
        1D grid of tokamak Z-coordinates describing the path of the flux surface.
        This is usually the height above the plasma midplane, but Z=0 may be set at any
        reference point. Should have same length as ``R``, and the endpoints should be
        repeated.
    b_radial: ArrayLike, units [tesla]
        1D grid of the radial magnetic field following the path described by R and Z.
        Should have the same length as ``R``.
    b_vertical: ArrayLike, units [tesla]
        1D grid of the vertical magnetic field following the path described by R and Z.
        Should have the same length as ``R``.
    b_toroidal: ArrayLike, units [tesla]
        1D grid of the toroidal magnetic field following the path described by R and Z.
        Should have the same length as ``R``.
    psi: float, units [weber / radian]
        The poloidal magnetic flux function :math:`\psi`.
    f: float, units [meter * tesla]
        The poloidal current function.
    f_prime: float, units [meter * tesla * radian / weber]
        The derivative of the poloidal current function :math:`f`  with respect to
        :math:`\psi`.
    p: float, units [pascal]
        Plasma pressure.
    p_prime: float, units [pascal * radian / weber]
        The derivative of the plasma pressure with respect to :math:`\psi`.
    q: float, units [dimensionless]
        The safety factor.
    q_prime: float, units [radian / weber]
        The derivative of the safety factor with respect to :math:`\psi``.
    R_major: float, units [meter]
        The major radius position of the center of each flux surface. This should be
        given by the mean of the maximum and minimum major radii of the flux surface.
    R_major_prime: float, units [meter * radian / weber]
        The derivative of the major radius position of the center of each flux surface
        with respect to :math:`\psi`.
    r_minor: float, units [meter]
        The minor radius of the flux surface. This should be half of the difference
        between the maximum and minimum major radii of the flux surface.
    r_minor_prime: float, units [meter * radian / weber]
        The derivative of the minor radius of the flux surface with respect to
        :math:`\psi`.
    Z_mid: float, units [meter]
        The z-midpoint of the flux surface. This should be the mean of the maximum and
        minimum z-positions of the flux surface.
    Z_mid_prime: float, units [meter * radian / weber]
        The derivative of the z-midpoint of the flux surface with respect to
        :math:`\psi`.
    psi_axis: float, units [weber / radian]
        The value of the poloidal magnetic flux function :math:`\psi` on the magnetic
        axis.
    psi_lcfs: float, units [weber / radian]
        The value of the poloidal magnetic flux function :math:`\psi` on the last closed
        flux surface.
    a_minor: float, units [meter]
        The minor radius of the last closed flux surface (LCFS). The minor radius of a
        flux surface is half of the difference between its maximum and minimum major
        radii. Though not necessarily applicable to this flux surface, a_minor is often
        used as a reference length.

    Attributes
    ----------

    data: xarray.Dataset
        The internal representation of the ``FluxSurface`` object. The function
        ``__getitem__`` redirects indexing lookups here, but the Dataset itself may be
        accessed directly by the user if they wish to perform more complex actions.
    psi: float, units [weber / radian]
    psi_n: float, units [dimensionless]
    rho: float, units [dimensionless]
    f: float, units [meter * tesla]
    f_prime: float, units [meter * tesla * radian / weber]
    p: float, units [pascal]
    p_prime: float, units [pascal * radian / weber]
    q: float, units [dimensionless]
    q_prime: float, units [dimensionless * radian / weber]
    R_major: float, units [meter]
    R_major_prime: float, units [meter * radian / weber]
    r_minor: float, units [meter]
    r_minor_prime: float, units [meter * radian / weber]
    Z_mid: float, units [meter]
    Z_mid_prime: float, units [meter * radian / weber]
    psi_axis: float, units [weber / radian]
    psi_lcfs: float, units [weber / radian]
    a_minor: float, units [meter]
    magnetic_shear: float, units [dimensionless]

    See Also
    --------

    Equilibrium: Object representing a global equilibrium.

    Notes
    -----

    The user can get derivatives using attribute look-up notation for names of the
    form ``d{name_1}_d{name_2}``. Derivatives with respect to :math:`\psi`, can also
    be obtained with the notation ``{name}_prime``. For example:

    ::

        # derivative of f with respect to psi
        flux_surface.df_dpsi
        flux_surface.f_prime
        # derivative of q with respect to r_minor
        flux_surface.dq_dr_minor
    """

    _init_units = {
        "self": None,
        "R": eq_units["len"],
        "Z": eq_units["len"],
        "b_radial": eq_units["b"],
        "b_vertical": eq_units["b"],
        "b_toroidal": eq_units["b"],
        "psi": eq_units["psi"],
        "f": eq_units["f"],
        "f_prime": eq_units["f_prime"],
        "p": eq_units["p"],
        "p_prime": eq_units["p_prime"],
        "q": eq_units["q"],
        "q_prime": eq_units["q_prime"],
        "R_major": eq_units["len"],
        "R_major_prime": eq_units["len_prime"],
        "r_minor": eq_units["len"],
        "r_minor_prime": eq_units["len_prime"],
        "Z_mid": eq_units["len"],
        "Z_mid_prime": eq_units["len_prime"],
        "psi_axis": eq_units["psi"],
        "psi_lcfs": eq_units["psi"],
        "a_minor": eq_units["len"],
    }

    @units.wraps(None, [*_init_units.values()], strict=False)
    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        b_radial: np.ndarray,
        b_vertical: np.ndarray,
        b_toroidal: np.ndarray,
        psi: float,
        f: float,
        f_prime: float,
        p: float,
        p_prime: float,
        q: float,
        q_prime: float,
        R_major: float,
        R_major_prime: float,
        r_minor: float,
        r_minor_prime: float,
        Z_mid: float,
        Z_mid_prime: float,
        psi_axis: float,
        psi_lcfs: float,
        a_minor: float,
    ):
        # Check floats
        psi = float(psi) * eq_units["psi"]
        f = float(f) * eq_units["f"]
        f_prime = float(f_prime) * eq_units["f_prime"]
        p = float(p) * eq_units["p"]
        p_prime = float(p_prime) * eq_units["p_prime"]
        q = float(q) * eq_units["q"]
        q_prime = float(q_prime) * eq_units["q_prime"]
        R_major = float(R_major) * eq_units["len"]
        R_major_prime = float(R_major_prime) * eq_units["len_prime"]
        r_minor = float(r_minor) * eq_units["len"]
        r_minor_prime = float(r_minor_prime) * eq_units["len_prime"]
        Z_mid = float(Z_mid) * eq_units["len"]
        Z_mid_prime = float(Z_mid_prime) * eq_units["len_prime"]
        psi_axis = float(psi_axis) * eq_units["psi"]
        psi_lcfs = float(psi_lcfs) * eq_units["psi"]
        a_minor = float(a_minor) * eq_units["len"]

        # Check the grids R, Z, b_radial, b_vertical, and b_toroidal
        R = np.asarray(R, dtype=float) * eq_units["len"]
        Z = np.asarray(Z, dtype=float) * eq_units["len"]
        b_radial = np.asarray(b_radial, dtype=float) * eq_units["b"]
        b_vertical = np.asarray(b_vertical, dtype=float) * eq_units["b"]
        b_toroidal = np.asarray(b_toroidal, dtype=float) * eq_units["b"]
        # Check that all grids have the same shape and have matching endpoints
        RZ_grids = {
            "R": R,
            "Z": Z,
            "b_radial": b_radial,
            "b_vertical": b_vertical,
            "b_toroidal": b_toroidal,
        }
        for name, grid in RZ_grids.items():
            if len(grid.shape) != 1:
                raise ValueError(f"The grid {name} must be 1D.")
            if not np.array_equal(grid.shape, (len(R),)):
                raise ValueError(f"The grid {name} should have length {len(R)}.")
            if not np.isclose(grid[0], grid[-1]):
                raise ValueError(f"The grid {name} must have matching endpoints.")

        # Determine theta grid, poloidal magnetic flux grid
        theta = np.arctan2(Z - Z_mid, R - R_major)
        b_poloidal = np.hypot(b_radial, b_vertical)

        # Assemble grids into xarray Dataset
        def make_var(val, desc):
            return ("theta_dim", val, {"units": str(val.units), "long_name": desc})

        coords = {
            "theta": make_var(theta, "Poloidal Angle"),
        }

        data_vars = {
            "R": make_var(R, "Radial Position"),
            "Z": make_var(Z, "Vertical Position"),
            "b_radial": make_var(b_radial, "Radial Magnetic Flux Density"),
            "b_vertical": make_var(b_vertical, "Vertical Magnetic Flux Density"),
            "b_toroidal": make_var(b_toroidal, "Toroidal Magnetic Flux Density"),
            "b_poloidal": make_var(b_poloidal, "Poloidal Magnetic Flux Density"),
        }

        attrs = {
            "psi": psi,
            "psi_n": (psi - psi_axis) / (psi_lcfs - psi_axis),
            "f": f,
            "f_prime": f_prime,
            "p": p,
            "p_prime": p_prime,
            "q": q,
            "q_prime": q_prime,
            "R_major": R_major,
            "R_major_prime": R_major_prime,
            "r_minor": r_minor,
            "r_minor_prime": r_minor_prime,
            "Z_mid": Z_mid,
            "Z_mid_prime": Z_mid_prime,
            "rho": r_minor / a_minor,
            "psi_axis": psi_axis,
            "psi_lcfs": psi_lcfs,
            "a_minor": a_minor,
            "magnetic_shear": (r_minor * q_prime) / (q * r_minor_prime),
        }

        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

        # Store primed attrs to aid in dynamic derivative calculation
        self._primes = {
            "psi": 1.0 * units.dimensionless,
            "psi_n": 1.0 / (psi_lcfs - psi_axis),
            "f": f_prime,
            "p": p_prime,
            "q": q_prime,
            "R_major": R_major_prime,
            "r_minor": r_minor_prime,
            "Z_mid": Z_mid_prime,
            "rho": r_minor_prime / a_minor,
        }

    def __getattr__(self, name: str) -> Any:
        """
        Return derivatives. The user can lookup attributes of the form ``{name}_prime``,
        or ``d{name_1}_d{name_2}``. On failure, falls back on ``__getattr__`` from the
        super class.
        """
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            try:
                prime = re.match(r"^([A-z_]+)_prime", name)
                if prime is not None:
                    return self._primes[prime.group(1)]
                dydx = re.match(r"^d([A-z_]+)_d([A-z_]+)$", name)
                if dydx is not None:
                    return self._primes[dydx.group(1)] / self._primes[dydx.group(2)]
            except KeyError:
                pass
            raise exc

    def plot(
        self,
        quantity: str,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        r"""
        Plot a quantity defined on the :math:`\theta` grid. These include ``R``,
        ``Z``, ``b_radial``, ``b_vertical``, ``b_poloidal`` and ``b_toroidal``.

        Parameters
        ----------
        quantity: str
            Name of the quantity to plot. Must be defined over the grid ``theta``.
        ax: Optional[plt.Axes]
            Axes object on which to plot. If not provided, a new figure is created.
        show: bool, default False
            Immediately show Figure after creation.
        x_label: Optional[str], default None
            Overwrite the default x label. Set to an empty string ``""`` to disable.
        y_label: Optional[str], default None
            Overwrite the default y label. Set to an empty string ``""`` to disable.
        **kwargs
            Additional arguments to pass to Matplotlib's ``plot`` call.

        Returns
        -------
        plt.Axes
            The Axes object created after plotting.

        Raises
        ------
        ValueError
            If ``quantity`` is not a quantity defined over the :math:`\theta` grid,
            or is not the name of a FluxSurface quantity.
        """
        if quantity not in self.data_vars:
            raise ValueError(
                f"Must be provided with a quantity defined on the theta grid."
                f"The quantity '{quantity}' is not recognised."
            )

        quantity_dims = self[quantity].dims
        if "theta_dim" not in quantity_dims or len(quantity_dims) != 1:
            raise ValueError(
                f"Must be provided with a quantity defined on the theta grid."
                f"The quantity '{quantity}' has coordinates {quantity_dims}."
            )

        if ax is None:
            _, ax = plt.subplots(1, 1)

        x_data = self["theta"]
        if x_label is None:
            x_label = f"{x_data.long_name} / ${x_data.data.units:L~}$"

        y_data = self[quantity]
        if y_label is None:
            y_label = f"{y_data.long_name} / ${y_data.data.units:L~}$"

        ax.plot(x_data.data.magnitude, y_data.data.magnitude, **kwargs)
        if x_label != "":
            ax.set_xlabel(x_label)
        if y_label != "":
            ax.set_ylabel(y_label)

        if show:
            plt.show()

        return ax

    def plot_path(
        self,
        ax: Optional[plt.Axes] = None,
        aspect: bool = False,
        show: bool = False,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        r"""
        Plot the path of the flux surface in :math:`(R, Z)` coordinates.

        Parameters
        ----------
        ax: Optional[plt.Axes]
            Axes object on which to plot. If not provided, a new figure is created.
        aspect: bool, default False
            If True, ensures the axes have the correct aspect ratio. If the user
            supplies their own ``ax``, has no effect.
        show: bool, default False
            Immediately show Figure after creation.
        x_label: Optional[str], default None
            Overwrite the default x label. Set to an empty string ``""`` to disable.
        y_label: Optional[str], default None
            Overwrite the default y label. Set to an empty string ``""`` to disable.
        **kwargs
            Additional arguments to pass to Matplotlib's ``plot`` call.

        Returns
        -------
        plt.Axes
            The Axes object created after plotting.
        """
        x_data = self["R"]
        if x_label is None:
            x_label = f"{x_data.long_name} / ${x_data.data.units:L~}$"

        y_data = self["Z"]
        if y_label is None:
            y_label = f"{y_data.long_name} / ${y_data.data.units:L~}$"

        if ax is None:
            _, ax = plt.subplots(1, 1)
            if aspect:
                ax.set_aspect("equal")

        ax.plot(x_data.data.magnitude, y_data.data.magnitude, **kwargs)
        if x_label != "":
            ax.set_xlabel(x_label)
        if y_label != "":
            ax.set_ylabel(y_label)

        if show:
            plt.show()

        return ax
