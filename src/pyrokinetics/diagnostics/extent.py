"""Calculates the extent of a field using the 95% rule"""

from numbers import Real

import numpy as np
import xarray as xr

from ..gk_code.gk_output import GKOutput


class Extent:
    def __init__(
        self,
        output: GKOutput,
        field: str = "phi",
        dl_dtheta: xr.DataArray | None = None,
        fraction: float = 0.0,
    ) -> None:
        if not isinstance(output, GKOutput):
            raise TypeError("output must be a GKOutput")
        if field not in ("phi", "apar", "bpar"):
            raise ValueError("field must be one of 'phi', 'apar', or 'bpar'")
        if field not in output:
            raise ValueError(f"GKOutput does not contain the field {field!r}")
        field_data = output[field]
        result = self._compute(field_data, dl_dtheta=dl_dtheta, fraction=fraction)
        result.attrs["source_variable"] = field
        output.data = output.data.assign({"extent": result})

    @staticmethod
    def _compute(
        field: xr.DataArray,
        dl_dtheta: xr.DataArray | None = None,
        fraction: float = 0.0,
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
            coordinate = field["theta"].values

        if not isinstance(fraction, Real) or not np.isfinite(fraction):
            raise ValueError("fraction must be a finite real scalar")
        if not coordinate[0] < fraction < coordinate[-1]:
            raise ValueError(
                "fraction must lie strictly inside the profile coordinate range"
            )

        field_amplitude = np.abs(field)
        power = (field_amplitude / field_amplitude.max()) ** 2
        total_power = power.integrate("theta")
        cdf = power.cumulative_integrate("theta") / total_power
        tail = (1 - fraction) / 2
        lo, hi = np.interp([tail, 1 - tail], cdf.values, coordinate)
        extent_value = float(hi - lo)
        result = xr.DataArray(extent_value, attrs={"name": "extent"})
        return result
