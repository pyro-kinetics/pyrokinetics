from __future__ import annotations  # noqa
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from contourpy import contour_generator
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
    # TODO contour_generator has an experimental threaded running mode, though you
    # need to divide the domain into chunks and stitch it back together in the end.
    cont_gen = contour_generator(x=Z, y=R, z=psi_RZ)
    contours = cont_gen.lines(psi)
    if not contours:
        raise RuntimeError(f"Could not find flux surface contours for psi={psi}")

    # Find the contour that is, on average, closest to the magnetic axis, as this
    # procedure may find additional open contours outside the last closed flux surface.
    if len(contours) > 1:
        RZ_axis = np.array([Z_axis, R_axis])
        mean_dist = [np.mean(np.linalg.norm(c - RZ_axis, axis=1)) for c in contours]
        contour = contours[np.argmin(mean_dist)]
    else:
        contour = contours[0]

    # Adjust the contour arrays so that we begin at the OMP (outside midplane)
    omp_idx = np.argmax(contour[:, 1])
    contour = np.roll(contour, -omp_idx, axis=0)

    # Return transpose so we have array of [[Zs...],[Rs...]], then swap to
    # [[Rs...,Zs...]]. Finally, ensure the endpoints match.
    endpoints = [[contour[0, 1]], [contour[0, 0]]]
    return np.concatenate((contour.T[::-1], endpoints), axis=1)


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
    b_poloidal: ArrayLike, units [tesla]
        1D grid of the magnitude of the poloidal magnetic field following the path
        described by R and Z. Should have the same length as ``R``.
    R_major: float, units [meter]
        The major radius position of the center of each flux surface. This should be
        given by the mean of the maximum and minimum major radii of the flux surface.
    r_minor: float, units [meter]
        The minor radius of the flux surface. This should be half of the difference
        between the maximum and minimum major radii of the flux surface.
    Z_mid: float, units [meter]
        The z-midpoint of the flux surface. This should be the mean of the maximum and
        minimum z-positions of the flux surface.
    f: float, units [meter * tesla]
        The poloidal current function.
    p: float, units [pascal]
        Plasma pressure.
    q: float, units [dimensionless]
        The safety factor.
    magnetic_shear: float, units [dimensionless]
        Defined as :math:`\frac{r}{q}\frac{dq}{dr}`, where :math:`r` is the minor radius
        and :math:`q` is the safety factor.
    shafranov_shift: float, units [dimensionless]
        The derivative of `R_major` with respect to `r_minor`
    midplane_shift: float, units [dimensionless]
        The derivative of `Z_mid` with respect to `r_minor`
    pressure_gradient: float, units [pascal / meter]
        The derivative of pressure with respect to `r_minor`.
    psi_gradient: float, units [weber * radian**-1 * meter**-1]
        The derivative of the poloidal magnetic flux function :math:`\psi` with respect
        to `r_minor`.
    a_minor: float, units [meter]
        The minor radius of the last closed flux surface (LCFS). Though not necessarily
        applicable to this flux surface, a_minor is often used as a reference length in
        gyrokinetic simulations.

    Attributes
    ----------

    data: xarray.Dataset
        The internal representation of the ``FluxSurface`` object. The function
        ``__getitem__`` redirects indexing lookups here, but the Dataset itself may be
        accessed directly by the user if they wish to perform more complex actions.
    rho: float, units [dimensionless]
    R_major: float, units [meter]
    r_minor: float, units [meter]
    Z_mid: float, units [meter]
    f: float, units [meter * tesla]
    p: float, units [pascal]
    q: float, units [dimensionless]
    magnetic_shear: float, units [dimensionless]
    shafranov_shift: float, units [dimensionless]
    midplane_shift: float, units [dimensionless]
    pressure_gradient: float, units [pascal / meter]
    psi_gradient: float, units [weber * radian**-1 * meter**-1]
    a_minor: float, units [meter]

    See Also
    --------

    Equilibrium: Object representing a global equilibrium.
    """

    # This dict defines the units for each argument to __init__.
    # The values are passed to the units.wraps decorator.
    _init_units = {
        "self": None,
        "R": eq_units["len"],
        "Z": eq_units["len"],
        "b_poloidal": eq_units["b"],
        "R_major": eq_units["len"],
        "r_minor": eq_units["len"],
        "Z_mid": eq_units["len"],
        "f": eq_units["f"],
        "p": eq_units["p"],
        "q": eq_units["q"],
        "magnetic_shear": units.dimensionless,
        "shafranov_shift": units.dimensionless,
        "midplane_shift": units.dimensionless,
        "pressure_gradient": eq_units["p"] / eq_units["len"],
        "psi_gradient": eq_units["psi"] / eq_units["len"],
        "a_minor": eq_units["len"],
    }

    @units.wraps(None, [*_init_units.values()], strict=False)
    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        b_poloidal: np.ndarray,
        R_major: float,
        r_minor: float,
        Z_mid: float,
        f: float,
        p: float,
        q: float,
        magnetic_shear: float,
        shafranov_shift: float,
        midplane_shift: float,
        pressure_gradient: float,
        psi_gradient: float,
        a_minor: float,
    ):
        # Check floats
        R_major = float(R_major) * eq_units["len"]
        r_minor = float(r_minor) * eq_units["len"]
        Z_mid = float(Z_mid) * eq_units["len"]
        f = float(f) * eq_units["f"]
        p = float(p) * eq_units["p"]
        q = float(q) * eq_units["q"]
        magnetic_shear = float(magnetic_shear) * units.dimensionless
        shafranov_shift = float(shafranov_shift) * units.dimensionless
        midplane_shift = float(shafranov_shift) * units.dimensionless
        pressure_gradient = float(pressure_gradient) * eq_units["p"] / eq_units["len"]
        psi_gradient = float(psi_gradient) * eq_units["psi"] / eq_units["len"]
        a_minor = float(a_minor) * eq_units["len"]

        # Check the grids R, Z, b_radial, b_vertical, and b_toroidal
        R = np.asarray(R, dtype=float) * eq_units["len"]
        Z = np.asarray(Z, dtype=float) * eq_units["len"]
        b_poloidal = np.asarray(b_poloidal, dtype=float) * eq_units["b"]
        # Check that all grids have the same shape and have matching endpoints
        RZ_grids = {
            "R": R,
            "Z": Z,
            "b_poloidal": b_poloidal,
        }
        for name, grid in RZ_grids.items():
            if len(grid.shape) != 1:
                raise ValueError(f"The grid {name} must be 1D.")
            if not np.array_equal(grid.shape, (len(R),)):
                raise ValueError(f"The grid {name} should have length {len(R)}.")
            if not np.isclose(grid[0], grid[-1]):
                raise ValueError(f"The grid {name} must have matching endpoints.")

        # Determine theta grid from R and Z
        theta = np.arctan2(Z - Z_mid, R - R_major)

        # Assemble grids into xarray Dataset
        def make_var(val, desc):
            return ("theta_dim", val, {"units": str(val.units), "long_name": desc})

        coords = {
            "theta": make_var(theta, "Poloidal Angle"),
        }

        data_vars = {
            "R": make_var(R, "Radial Position"),
            "Z": make_var(Z, "Vertical Position"),
            "b_poloidal": make_var(b_poloidal, "Poloidal Magnetic Flux Density"),
        }

        attrs = {
            "R_major": R_major,
            "r_minor": r_minor,
            "Z_mid": Z_mid,
            "f": f,
            "p": p,
            "q": q,
            "magnetic_shear": magnetic_shear,
            "shafranov_shift": shafranov_shift,
            "midplane_shift": midplane_shift,
            "pressure_gradient": pressure_gradient,
            "psi_gradient": psi_gradient,
            "a_minor": a_minor,
            "rho": r_minor / a_minor,
        }

        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

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
