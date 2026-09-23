"""Calculates the extent of a field using the 95% rule"""

from numbers import Real

import numpy as np
import xarray as xr

from ..gk_code.gk_output import GKOutput
from ..dataset_wrapper import DatasetWrapper


class Extent:
    def __init__(
        self,
        output: GKOutput,
        dl_dtheta: xr.DataArray | None = None,
        fraction: float = 0.95,
    ) -> None:
        if not isinstance(output, DatasetWrapper):
            raise TypeError("output must be a DatasetWrapper")

        fields = [
            name for name in ("phi", "apar", "bpar")
            if name in output
        ]
        if not fields:
            raise ValueError("GKOutput contains none of 'phi', 'apar', or 'bpar'")

        extents = []
        field_bounds = []

        for name in fields:
            extent, bounds = self._compute(
                output[name],
                dl_dtheta=dl_dtheta,
                fraction=fraction,
            )
            extents.append(extent)
            field_bounds.append(bounds)

        field_coordinate = xr.IndexVariable("field", fields)

        output.data = output.data.assign(
            extent=xr.concat(extents, dim=field_coordinate),
            bounds=xr.concat(field_bounds, dim=field_coordinate).assign_coords(
                bound=["lo", "hi"],
            ),
        )

    @staticmethod
    def _compute(
        field: xr.DataArray,
        dl_dtheta: xr.DataArray | None = None,
        fraction: float = 0.05,
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
        if not 0 < fraction < 1:
            raise ValueError("fraction must lie strictly between 0 and 1")

        field_amplitude = np.abs(np.real(field))
        power = (field_amplitude / field_amplitude.max()) ** 2
        total_power = power.integrate("theta")
        cdf = power.cumulative_integrate("theta") / total_power
        tail = (1 - fraction) / 2
        bounds = xr.apply_ufunc(
            np.interp,
            xr.DataArray([tail, 1 - tail], dims="bound"),
            cdf,
            cdf.theta,
            input_core_dims=[["bound"], ["theta"], ["theta"]],
            output_core_dims=[["bound"]],
            vectorize=True,
        )
        lo = bounds.isel(bound=0, drop=True)
        hi = bounds.isel(bound=1, drop=True)
        width = hi - lo
        result = xr.DataArray(width, attrs={"name": "extent"})
        bounds = xr.DataArray(bounds, attrs={"name": "extent"})
        return result,bounds
