"""Reflection parity of complex field profiles from individual runs and scans."""

from numbers import Real

import numpy as np
import xarray as xr

from ..gk_code.gk_output import GKOutput


class Parity:
    r"""Calculate reflection parity and store it in an existing wrapped output.

    Calling ``Parity(output, ...)`` immediately adds the result to ``output``.
    Retrieve it through ``output["parity"]``. The original wrapper,
    metadata, normalization state, coordinates, and source variables are kept.

    .. math::

        f_\pm(\theta) = \frac{f(\theta) \pm f(2c-\theta)}{2},
        \qquad E_\pm = \int |f_\pm|^2 w\,d\theta,
        \qquad P = \frac{E_+ - E_-}{E_+ + E_-}.

    Parameters
    ----------
    output : GKOutput
        Gyrokinetic output to update in place.
    field : {"phi", "apar", "bpar"}, default "phi"
        Field whose parity is calculated. The theta dimension is reduced; all
        other dimensions and coordinates are retained.
    dl_dtheta : xarray.DataArray, optional
        Positive integration weights, normally ``sqrt(g_theta_theta)`` for
        integration along a field line. Must already be evaluated on the field
        grid. Dimensions must be a subset of the field dimensions, including
        ``theta``, and shared coordinates must match exactly.
        If omitted, integrate directly over the profile coordinate.
    center : float, default 0.0
        Reflection centre ``c``, in the same units as the profile coordinate.
        Must lie strictly inside the coordinate range. A single centre applies
        to every profile. Call Parity again with a new value and
        a new value to update the stored result.

    Notes
    -----
    A result of +1 denotes an even field, -1 an odd field, and zero equal
    squared norms of the even and odd components. Reflection acts on the full
    complex field, without conjugation. The result is invariant under a
    nonzero constant amplitude or phase factor. These squared norms are not
    physical mode energies, and parity alone does not identify an instability.

    Only the largest interval symmetric about ``center`` contained in the
    supplied coordinate range is used. Unpaired tails are excluded. Linear
    interpolation pairs samples at equal distances from the centre, including
    on nonuniform grids; no periodic wrapping or extrapolation is performed.
    The two sides' weights are averaged, equivalent to integrating the even
    squared amplitudes on both sides. Trapezoidal integration keeps the result
    in [-1, 1] up to roundoff. Zero profiles and profiles containing NaN or
    infinity return NaN. A resolved profile is needed for accurate interpolation.

    Examples
    --------
    >>> Parity(pyro.gk_output, field="apar", center=0.5)  # doctest: +SKIP
    >>> result = pyro.gk_output["parity"]  # doctest: +SKIP
    """

    def __init__(
        self,
        output: GKOutput,
        field: str = "phi",
        dl_dtheta: xr.DataArray | None = None,
        center: float = 0.0,
    ) -> None:
        if not isinstance(output, GKOutput):
            raise TypeError("output must be a GKOutput")
        if field not in ("phi", "apar", "bpar"):
            raise ValueError("field must be one of 'phi', 'apar', or 'bpar'")
        if field not in output:
            raise ValueError(f"GKOutput does not contain the field {field!r}")
        field_data = output[field]
        result = self._compute(field_data, dl_dtheta=dl_dtheta, center=center)
        result.attrs["source_variable"] = field
        output.data = output.data.assign({"parity": result})

    @staticmethod
    def _compute(
        field: xr.DataArray,
        dl_dtheta: xr.DataArray | None = None,
        center: float = 0.0,
    ) -> xr.DataArray:
        if not isinstance(field, xr.DataArray):
            raise TypeError("field must be an xarray.DataArray")
        if "theta" not in field.dims or "theta" not in field.coords:
            raise ValueError("field must have a theta coordinate")
        coordinate = field["theta"]
        values = getattr(coordinate.data, "magnitude", coordinate.data)
        if (
            coordinate.dims != ("theta",)
            or coordinate.size < 2
            or not np.all(np.isfinite(values))
            or not np.all(np.diff(values) > 0)
        ):
            raise ValueError(
                "theta must be a finite, strictly increasing 1D coordinate "
                "with at least two points"
            )

        if dl_dtheta is not None:
            if not isinstance(dl_dtheta, xr.DataArray):
                raise TypeError("dl_dtheta must be an xarray.DataArray")
            if "theta" not in dl_dtheta.dims or not set(dl_dtheta.dims) <= set(
                field.dims
            ):
                raise ValueError(
                    "dl_dtheta dimensions must be a subset of field dimensions "
                    "and include 'theta'"
                )
            if "theta" not in dl_dtheta.coords:
                raise ValueError("dl_dtheta must have a theta coordinate")
            xr.align(field, dl_dtheta, join="exact", copy=False)
            weights = getattr(dl_dtheta.data, "magnitude", dl_dtheta.data)
            if (
                np.iscomplexobj(weights)
                or not np.all(np.isfinite(weights))
                or not np.all(weights > 0)
            ):
                raise ValueError("dl_dtheta must contain finite, positive real weights")

        # TODO: Implement a robust method for automatically finding the mode
        # centre, including displaced, multi-peaked, and mixed-parity profiles.
        coordinate = field["theta"].values
        if not isinstance(center, Real) or not np.isfinite(center):
            raise ValueError("center must be a finite real scalar")
        if not coordinate[0] < center < coordinate[-1]:
            raise ValueError(
                "center must lie strictly inside the profile coordinate range"
            )
        theta_reflected = 2 * center - field.theta

        field_inverse = (
            field.interp(theta=theta_reflected)
            .assign_coords(theta=field.theta)
        )

        field_inverse = field.sel(theta = 2*center - field.theta)
        field_plus = (field + field_inverse)/2
        field_minus = (field - field_inverse)/2
        E_plus = abs(field_plus)*2.integration(dim="theta)
        E_minus = abs(field_minus)*2.integration(dim="theta)
        result = (E_Plus - E_minus) / (E_Plus + E_Minus)

        

        field = field.copy(data=getattr(field.data, "magnitude", field.data))
        scale = np.abs(field).max(dim="theta", skipna=False)
        field = field / scale.where(np.isfinite(scale) & (scale > 0))
        positive, negative = sample_sides(field)
        even_squared = np.abs((positive + negative) / 2) ** 2
        odd_squared = np.abs((positive - negative) / 2) ** 2
        if dl_dtheta is not None:
            weights = dl_dtheta.copy(
                data=getattr(dl_dtheta.data, "magnitude", dl_dtheta.data)
            )
            positive_weight, negative_weight = sample_sides(weights)
            paired_weight = (positive_weight + negative_weight) / 2
            even_squared = even_squared * paired_weight
            odd_squared = odd_squared * paired_weight

        # Both full-domain norms have a factor of two, which cancels.
        even_norm = even_squared.integrate(coord="theta")
        odd_norm = odd_squared.integrate(coord="theta")
        total_norm = even_norm + odd_norm
        result = (even_norm - odd_norm) / total_norm.where(total_norm > 0)
        result.name = "parity"
        return result
