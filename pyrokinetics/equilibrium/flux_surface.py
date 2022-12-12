from __future__ import annotations  # noqa

import numpy as np
from numpy.typing import ArrayLike
from skimage.measure import find_contours

from ..dataset_wrapper import DatasetWrapper
from ..units import ureg as units
from .equilibrium_units import eq_units


@units.wraps(units.meter, [units.m, units.m, units.weber / units.rad] * 2, strict=False)
def _flux_surface_contour(
    R: ArrayLike,
    Z: ArrayLike,
    psi_RZ: ArrayLike,
    R_axis: float,
    Z_axis: float,
    psi: float,
):
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
    R = np.asanyarray(R, dtype=float)
    Z = np.asanyarray(Z, dtype=float)
    psi_RZ = np.asanyarray(psi_RZ, dtype=float)
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
    f: float, units [meter * tesla]
    f_prime: float, units [meter * tesla * radian / weber]
    p: float, units [pascal]
    p_prime: float, units [pascal * radian / weber]
    q: float, units [dimensionless]
    q_prime: float, units [radian / weber]
    R_major: float, units [meter]
    R_major_prime: float, units [meter * radian / weber]
    r_minor: float, units [meter]
    r_minor_prime: float, units [meter * radian / weber]
    Z_mid: float, units [meter]
    Z_mid_prime: float, units [meter * radian / weber]
    psi_axis: float, units [weber / radian]
    psi_lcfs: float, units [weber / radian]
    a_minor: float, units [meter]

    See Also
    --------

    Equilibrium: Object representing a global equilibrium.
    """

    _init_units = {
        "self": None,
        "R": eq_units["len"],
        "Z": eq_units["len"],
        "b_radial": eq_units["b"],
        "b_vertical": eq_units["b"],
        "b_toroidal": eq_units["b"],
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
            "theta": make_var(theta, "Angular Position"),
        }

        data_vars = {
            "R": make_var(R, "R Major Position"),
            "Z": make_var(Z, "Vertical Position"),
            "b_radial": make_var(
                b_radial, "Magnetic Flux Density, Major-Radius Direction"
            ),
            "b_vertical": make_var(
                b_vertical, "Magnetic Flux Density, Vertical Direction"
            ),
            "b_toroidal": make_var(
                b_toroidal, "Magnetic Flux Density, Toroidal Direction"
            ),
            "b_poloidal": make_var(
                b_poloidal, "Magnitude of Poloidal Magnetic Flux Density"
            ),
        }

        attrs = {
            "f": f.magnitude[()],
            "f_prime": f_prime.magnitude[()],
            "p": p.magnitude[()],
            "p_prime": p_prime.magnitude[()],
            "q": q.magnitude[()],
            "q_prime": q_prime.magnitude[()],
            "R_major": R_major.magnitude[()],
            "R_major_prime": R_major_prime.magnitude[()],
            "r_minor": r_minor.magnitude[()],
            "r_minor_prime": r_minor_prime.magnitude[()],
            "Z_mid": Z_mid.magnitude[()],
            "Z_mid_prime": Z_mid_prime.magnitude[()],
            "psi_axis": psi_axis.magnitude[()],
            "psi_lcfs": psi_lcfs.magnitude[()],
            "a_minor": a_minor.magnitude[()],
        }

        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

    f = property(lambda self: self.data.f * eq_units["f"])
    f_prime = property(lambda self: self.data.f_prime * eq_units["f_prime"])
    p = property(lambda self: self.data.p * eq_units["p"])
    p_prime = property(lambda self: self.data.p_prime * eq_units["p_prime"])
    q = property(lambda self: self.data.q * eq_units["q"])
    q_prime = property(lambda self: self.data.q_prime * eq_units["q_prime"])
    R_major = property(lambda self: self.data.R_major * eq_units["len"])
    R_major_prime = property(
        lambda self: self.data.R_major_prime * eq_units["len_prime"]
    )
    r_minor = property(lambda self: self.data.r_minor * eq_units["len"])
    r_minor_prime = property(
        lambda self: self.data.r_minor_prime * eq_units["len_prime"]
    )
    Z_mid = property(lambda self: self.data.Z_mid * eq_units["len"])
    Z_mid_prime = property(lambda self: self.data.Z_mid_prime * eq_units["len_prime"])
    psi_axis = property(lambda self: self.data.psi_axis * eq_units["psi"])
    psi_lcfs = property(lambda self: self.data.psi_lcfs * eq_units["psi"])
    a_minor = property(lambda self: self.data.a_minor * eq_units["len"])

    def psi_derivative(self, key: str) -> float:
        r"""
        Returns the derivative of the attribute referrenced by 'key' with respect to
        the poloidal flux function, :math:`\psi`.
        """
        if key not in self.data.attrs:
            raise KeyError(f"'{key}' not found.")
        return getattr(self, f"{key}_prime")

    def psin_derivative(self, key: str) -> float:
        r"""
        Returns the derivative of the attribute referrenced by 'key' with respect to
        the normalised poloidal flux function, :math:`\psi_n`.
        """
        return self.psi_derivative(key) * (self.psi_lcfs - self.psi_axis)

    def r_minor_derivative(self, key: str) -> float:
        r"""
        Returns the derivative of the attribute referrenced by 'key' with respect to
        the minor radius of the flux surface, :math:`r`.
        """
        return self.psi_derivative(key) / self.r_minor_prime

    def rho_derivative(self, key: str) -> float:
        r"""
        Returns the derivative of the attribute referrenced by 'key' with respect to
        the normalised minor radius of the flux surface, :math:`\rho=r/a_\text{minor}`.
        """
        return self.r_minor_derivative(key) * self.a_minor
