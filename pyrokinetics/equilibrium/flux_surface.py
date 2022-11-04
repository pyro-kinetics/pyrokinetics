from __future__ import annotations  # noqa

import numpy as np
from numpy.typing import ArrayLike
from skimage.measure import find_contours

from ..dataset_wrapper import DatasetWrapper
from ..normalisation import ureg as units


@units.wraps(units.meter, [units.m, units.m, units.weber / units.rad] * 2, strict=False)
def _flux_surface_contour(
    r: ArrayLike,
    z: ArrayLike,
    psi_rz: ArrayLike,
    r_axis: float,
    z_axis: float,
    psi: float,
):
    r"""
    Given linearly-spaced RZ coordinates and psi at these positions, returns the R and Z
    coordinates of a contour at given psi. Describes the path of a single magnetic flux
    surface within a tokamak. Aims to return the closest closed contour to the position
    ``(r_axis, z_axis)``.

    Parameters
    ----------
    r: ArrayLike
        Linearly spaced and monotonically increasing 1D grid of major radius
        coordinates, i.e. the radial distance from the central column of a tokamak.
    z: ArrayLike
        Linearly spaced and monotonically increasing 1D grid of z-coordinates describing
        the distance from the midplane of a tokamak.
    psi_rz: ArrayLike
        2D grid of :math:`\psi`, the poloidal magnetic flux function, over the range
        (r,z).
    r_axis: float
        r position of the magnetic axis.
    z_axis: float
        z position of the magnetic axis.
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
        If the shapes of ``r``, ``z``, and ``psi_rz`` don't match.
    RuntimeError
        If no flux surface contours could be found.

    Warnings
    --------
    For performance reasons, the grids ``r`` and ``z``, this function does not check
    that ``r`` or ``z`` are linearly spaced or monotonically increasing. If this
    condition is not upheld, the results are undefined.
    """

    # Check some basic conditions on r, z, psi_rz
    r = np.asanyarray(r, dtype=float)
    z = np.asanyarray(z, dtype=float)
    psi_rz = np.asanyarray(psi_rz, dtype=float)
    if len(r.shape) != 1:
        raise ValueError("The grid r should be 1D.")
    if len(z.shape) != 1:
        raise ValueError("The grid z should be 1D.")
    if not np.array_equal(psi_rz.shape, (len(r), len(z))):
        raise ValueError(
            f"The grid psi_rz has shape {psi_rz.shape}. "
            f"It should have shape {(len(r), len(z))}."
        )

    # Get contours, raising error if none are found
    contours_raw = find_contours(psi_rz, psi)
    if not contours_raw:
        raise RuntimeError(f"Could not find flux surface contours for psi={psi}")

    # Normalise to RZ grid
    # Normalisation assumes R and Z are linspace grids with a positive spacing.
    # The raw contours have a range of 0 to len(x)-1, where x is r or z.
    scaling = np.array([(x[-1] - x[0]) / (len(x) - 1) for x in (r, z)])
    rz_min = np.array([r[0], z[0]])
    contours = [contour * scaling + rz_min for contour in contours_raw]

    # Find the contour that is, on average, closest to the magnetic axis, as this
    # procedure may find additional open contours outside the last closed flux surface.
    if len(contours) > 1:
        rz_axis = np.array([[r_axis, z_axis]])
        mean_dist = [np.mean(np.linalg.norm(c - rz_axis, axis=1)) for c in contours]
        contour = contours[np.argmin(mean_dist)]
    else:
        contour = contours[0]

    # Adjust the contour arrays so that we begin at the OMP (outside midplane)
    omp_idx = np.argmax(contour[0, :])
    contour = np.roll(contour, -omp_idx, axis=0)

    # Return transpose so we have array of [[rs...],[zs...]]
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

    r: numpy.ndarray
        1D grid of major radius coordinates describing the path of the flux surface.
        This is the radius from the central column of a tokamak, and not the radial
        distance from the magnetic axis. Units are in meters. The endpoints should be
        repeated.
    z: numpy.ndarray
        1D grid of tokamak z-coordinates describing the path of the flux surface.
        This is usually the height above the plasma midplane, but z=0 may be set at any
        reference point. Should have same length as ``r``, and units are in meters.
        The endpoints should be repeated.
    br: numpy.ndarray
        1D grid of the radial magnetic field following the path described by r and z.
        Should have the same length as ``r``, and units are in Teslas.
    bz: numpy.ndarray
        1D grid of the vertical magnetic field following the path described by r and z.
        Should have the same length as ``r``, and units are in Teslas.
    bt: numpy.ndarray
        1D grid of the toroidal magnetic field following the path described by r and z.
        Should have the same length as ``r``, and units are in Teslas.
    f: float
        The poloidal current function. Units are meter-Teslas (mT).
    f_prime: float
        The derivative of the poloidal current function ``f`` with respect to ``psi``.
        Units are Rad m T/Wb, where 'Rad' is radians, 'm' is meters, 'T' is Teslas, and
        'Wb' is Webers.
    p: float
        Plasma pressure, in units of pascals.
    p_prime: float
        The derivative of the plasma pressure with respect to ``psi``. Units are
        pascal-radians per Weber.
    q: float
        The safety factor. Has no units.
    q_prime: float
        The derivative of the safety factor with respect to ``psi``. Has units of
        radians per Weber.
    r_major: float
        The major radius position of the center of each flux surface. Units are in
        meters. This should be given by the mean of the maximum and minimum major radii
        of the flux surface.
    r_major_prime: float
        The derivative of the major radius position of the center of each flux surface
        with respect to :math:`\psi`. Units are in meters-radians per Weber.
    r_minor: float
        The minor radius of the flux surface, in meters. This should be half of the
        difference between the maximum and minimum major radii of the flux surface.
    r_minor_prime: float
        The derivative of the minor radius of the flux surface with respect to
        :math:`\psi`, in meter-radians per Weber.
    z_mid: float
        The z-midpoint of the flux surface, in meters. This should be the mean of the
        maximum and minimum z-positions of the flux surface.
    z_mid_prime: float
        The derivative of the z-midpoint of the flux surface with respect to
        :math:`\psi`, in meter-radians per Weber.
    psi_axis: float
        The value of the poloidal magnetic flux function :math:`\psi` on the magnetic
        axis. Units of Webers per radian.
    psi_lcfs: float
        The value of the poloidal magnetic flux function :math:`\psi` on the last closed
        flux surface. Units of Webers per radian.
    a_minor: float
        The minor radius of the last closed flux surface (LCFS), in units of meters. The
        minor radius of a flux surface is half of the difference between its maximum and
        minimum major radii. Though not necessarily applicable to this flux surface,
        a_minor is often used as a reference length.

    Attributes
    ----------

    data: xarray.Dataset
        The internal representation of the ``FluxSurface`` object. The functions
        ``__getattr__`` and ``__getitem__`` redirect most attribute/indexing lookups
        here, but the Dataset itself may be accessed directly by the user if they wish
        to perform more complex manipulations.
    """

    def __init__(
        self,
        r: np.ndarray,
        z: np.ndarray,
        br: np.ndarray,
        bz: np.ndarray,
        bt: np.ndarray,
        f: float,
        f_prime: float,
        p: float,
        p_prime: float,
        q: float,
        q_prime: float,
        r_major: float,
        r_major_prime: float,
        r_minor: float,
        r_minor_prime: float,
        z_mid: float,
        z_mid_prime: float,
        psi_axis: float,
        psi_lcfs: float,
        a_minor: float,
    ):
        # Check floats
        f = float(f)
        f_prime = float(f_prime)
        p = float(p)
        p_prime = float(p_prime)
        q = float(q)
        q_prime = float(q_prime)
        r_major = float(r_major)
        r_major_prime = float(r_major_prime)
        r_minor = float(r_minor)
        r_minor_prime = float(r_minor_prime)
        z_mid = float(z_mid)
        z_mid_prime = float(z_mid_prime)
        psi_axis = float(psi_axis)
        psi_lcfs = float(psi_lcfs)
        a_minor = float(a_minor)

        # Check the grids r, z, br, bz, and bt
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        br = np.asarray(br, dtype=float)
        bz = np.asarray(bz, dtype=float)
        bt = np.asarray(bt, dtype=float)
        # Check that all grids have the same shape and have matching endpoints
        for name, grid in {"r": r, "z": z, "br": br, "bz": bz, "bt": bt}.items():
            if len(grid.shape) != 1:
                raise ValueError(f"The grid {name} must be 1D.")
            if not np.array_equal(grid.shape, (len(r),)):
                raise ValueError(f"The grid {name} should have length {len(r)}.")
            if not np.isclose(grid[0], grid[-1]):
                raise ValueError(f"The grid {name} must have matching endpoints.")

        # Collect plain variables into a dict
        variables = {
            "f": f,
            "p": p,
            "q": q,
            "r_major": r_major,
            "r_minor": r_minor,
            "z_mid": z_mid,
        }

        # Collect psi derivatives into a dict
        psi_derivatives = {
            "df_dpsi": f_prime,
            "dp_dpsi": p_prime,
            "dq_dpsi": q_prime,
            "drmajor_dpsi": r_major_prime,
            "drminor_dpsi": r_minor_prime,
            "dzmid_dpsi": z_mid_prime,
        }

        # Generate psi_n derivatives
        psi_n_derivatives = {
            k.replace("dpsi", "dpsin"): v * (psi_lcfs - psi_axis)
            for k, v in psi_derivatives.items()
        }

        # Generate r_minor derivatives
        r_minor_derivatives = {
            k.replace("dpsi", "drminor"): v / r_minor_prime
            for k, v in psi_derivatives.items()
        }

        # Generate r_minor derivatives
        rho_derivatives = {
            k.replace("drminor", "drho"): v * a_minor
            for k, v in r_minor_derivatives.items()
        }

        # Remove the pointless derivatives
        r_minor_derivatives.pop("drminor_drminor")
        rho_derivatives.pop("drminor_drho")

        # Assemble grids into xarray Dataset
        # TODO units
        super().__init__(
            coords={
                "path_idx": np.arange(len(r)),
            },
            data_vars={
                "r": ("path_idx", r),
                "z": ("path_idx", z),
                "br": ("path_idx", br),
                "bz": ("path_idx", bz),
                "bp": ("path_idx", np.hypot(br, bz)),
                "bt": ("path_idx", bt),
            },
            attrs={
                **variables,
                **psi_derivatives,
                **psi_n_derivatives,
                **r_minor_derivatives,
                **rho_derivatives,
                "psi_axis": psi_axis,
                "psi_lcfs": psi_lcfs,
                "a_minor": a_minor,
            },
        )
