"""Calculates the extent of eigenfunctions using the 95% rule."""

from numbers import Real

import numpy as np
import xarray as xr

from ..dataset_wrapper import DatasetWrapper
from ..gk_code.gk_output import GKOutput


class Extent:
    """
    Extent in ballooning angle ``theta`` of each eigenfunction.

    The extent is the width in ``theta`` holding ``fraction`` of the integral
    of ``|eigenfunction|**2``, leaving ``(1 - fraction) / 2`` in each tail.
    ``|eigenfunction|**2`` is used as it does not depend on the eigenfunction's
    complex phase. If the output holds ``eigenfunctions_squared`` (as a
    ``PyroScan`` loaded with ``time_mode="average"`` does), that is used.

    ``extent`` and ``bounds`` (the ``theta`` of each tail, along ``bound``) are
    added to ``output``.
    """

    def __init__(self, output: GKOutput, fraction: float = 0.95) -> None:
        if not isinstance(output, DatasetWrapper):
            raise TypeError("output must be a DatasetWrapper")

        if "eigenfunctions_squared" in output:
            power = output["eigenfunctions_squared"]
        elif "eigenfunctions" in output:
            power = np.abs(output["eigenfunctions"]) ** 2
        else:
            raise ValueError("output contains no eigenfunctions")

        extent, bounds = self._compute(power, fraction=fraction)

        output.data = output.data.assign(
            extent=extent,
            bounds=bounds.assign_coords(bound=["lo", "hi"]),
        )

    @staticmethod
    def _compute(power: xr.DataArray, fraction: float) -> xr.DataArray:
        if not isinstance(power, xr.DataArray):
            raise TypeError("power must be an xarray.DataArray")
        if "theta" not in power.dims or "theta" not in power.coords:
            raise ValueError("power must have a theta coordinate")
        coordinate = power["theta"]
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

        if not isinstance(fraction, Real) or not np.isfinite(fraction):
            raise ValueError("fraction must be a finite real scalar")
        if not 0 < fraction < 1:
            raise ValueError("fraction must lie strictly between 0 and 1")

        total_power = power.integrate("theta")
        cdf = power.cumulative_integrate("theta") / total_power
        cdf = cdf.copy(data=getattr(cdf.data, "magnitude", cdf.data))
        theta_units = getattr(cdf.theta.data, "units", None)
        theta = xr.DataArray(
            getattr(cdf.theta.data, "magnitude", cdf.theta.data),
            dims=cdf.theta.dims,
            coords=cdf.theta.coords,
        )
        tail = (1 - fraction) / 2
        bounds = xr.apply_ufunc(
            np.interp,
            xr.DataArray([tail, 1 - tail], dims="bound"),
            cdf,
            theta,
            input_core_dims=[["bound"], ["theta"], ["theta"]],
            output_core_dims=[["bound"]],
            vectorize=True,
        )
        if theta_units is not None:
            bounds = bounds.copy(data=bounds.data * theta_units)
        lo = bounds.isel(bound=0, drop=True)
        hi = bounds.isel(bound=1, drop=True)
        width = hi - lo
        result = xr.DataArray(width, attrs={"name": "extent"})
        bounds = xr.DataArray(bounds, attrs={"name": "extent"})
        return result, bounds
