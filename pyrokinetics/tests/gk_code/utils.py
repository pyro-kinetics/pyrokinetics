import xarray as xr
import numpy as np
from pathlib import Path


def get_golden_answer_data(filename: Path) -> xr.Dataset:
    """
    Read stored golden answer data, converting to complex where needed.
    As netcdf4 does not support complex types, an extra dimension 'ReIm' was added to
    the datasets, and the real and imaginary parts were split. Here, we must recombine
    them.
    """
    ds = xr.open_dataset(filename)
    ds = ds.isel(ReIm=0) + 1j * ds.isel(ReIm=1)
    return ds


def array_similar(x, y, nan_to_zero: bool = False) -> bool:
    """
    Ensure arrays are similar, after squeezing dimensions of len 1 and (potentially)
    replacing nans with zeros. Transposes both to same coords.
    """
    # Deal with changed nans
    if nan_to_zero:
        x, y = np.nan_to_num(x), np.nan_to_num(y)
    # Squeeze out any dims of size 1
    x, y = x.squeeze(drop=True), y.squeeze(drop=True)
    # transpose both to the same shape
    # only transpose the coords that exist in both
    coords = x.coords
    x, y = x.transpose(*coords), y.transpose(*coords)
    return np.allclose(x, y)
