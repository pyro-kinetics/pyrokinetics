from typing import Callable
import numpy as np
from skimage.measure import find_contours


def get_flux_surface(
    R: np.ndarray,
    Z: np.ndarray,
    psi_RZ: Callable[[np.ndarray, np.ndarray], np.ndarray],
    psi_axis: float,
    psi_bdry: float,
    R_axis: float,
    Z_axis: float,
    psi_n: float,
):
    """
    Given linearly-spaced RZ coordinates and a function psi(R,Z), returns the
    R and Z coordinates of a contour at normalised psi_n. Describes a single magnetic
    flux surface within a tokamak.

    Parameters
    ----------
    R : np.ndarray
        Linearly spaced and monotonically increasing 1D grid of major radius
        coordinates, i.e. the radial distance from the central column of a tokamak.
    Z : np.ndarray
        Linearly spaced and monotonically increasing 1D grid of z-coordinates describing
        the distance from the midplane of a tokamak.
    psi_RZ: Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function describing (non-normalised) psi, the poloidal magnetic flux function,
        in the range (R,Z).
    psi_axis: float
        The value of psi along the magnetic axis.
    psi_bdry: float
        The value of psi at the Last Closed Flux Surface (LCFS)
    R_axis: float
        R position of the magnetic axis.
    Z_axis: float
        Z position of the magnetic axis.
    psi_n: float
        Normalised psi coordinate on which to fit a contour. A value close to 0.0 will
        be near the magnetic axis, while a value of 1.0 will be on the last closed flux
        surface.

    Returns
    -------
    np.ndarray
        2D array containing R and Z coordinates of the flux surface contour. Indexing
        with [0,:] gives a 1D array of R coordinates, while [1,:] gives a 1D array of
        Z coordinates. The endpoints are repeated, so [:,0] == [:,-1].

    Raises
    ------
    ValueError
        If psi_n is outside the range 0.0 <= psi_n <= 1.0.
    RuntimeError
        If no flux surface contours could be found.
    """

    if psi_n > 1.0 or psi_n < 0.0:
        raise ValueError(f"psi_n={psi_n}, but we require 0.0 <= psi_n <= 1.0")

    # Generate 2D mesh of normalised psi
    # TODO psi_RZ should be a grid, could avoid expensive spline interpolation
    # TODO Pass in psi_n grid instead of psi? Could avoid two args
    psi_2d = psi_RZ(R, Z)
    psin_2d = (psi_2d - psi_axis) / (psi_bdry - psi_axis)

    # Get contours, raising error if none are found
    contours_raw = find_contours(psin_2d, psi_n)
    if not contours_raw:
        raise RuntimeError(f"Could not find flux surface contours for psi_n={psi_n}")

    # Normalise to RZ grid
    # Normalisation assumes R and Z are linspace grids with a positive spacing.
    # The raw contours have a range of 0 to len(x)-1, for x in [R,Z].
    scaling = np.array([(x[-1] - x[0]) / (len(x) - 1) for x in (R, Z)])
    RZ_min = np.array([R[0], Z[0]])
    contours = [contour * scaling + RZ_min for contour in contours_raw]

    # Find the contour that is, on average, closest to the magnetic axis, as this
    # procedure may find additional open contours outside the last closed flux surface.
    if len(contours) > 1:
        # TODO Determine R_axis and Z_axis here? Could avoid two args
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
