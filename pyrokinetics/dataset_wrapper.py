import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ._version import __version__
from .typing import PathLike

import xarray as xr
import netCDF4 as nc


class DatasetWrapper:
    """
    Base class for classes that store their data in an Xarray Dataset. Defines a number
    of useful functions, such as ``__getattr__`` and ``__getitem__`` that redirect to
    the underlying Dataset, and methods to read or write to disk. Ensures that the
    underlying Dataset contains metadata about the current session. The user may
    access the underlying Dataset via ``self.data``.

    Parameters
    ----------
    data_vars: Optional[Dict[str, Any]]
        Variables to be passed to the underlying Dataset.
    coords: Optional[Dict[str, Any]]
        Coordinates to be passed to the underlying Dataset.
    attrs: Optional[Dict[str,Any]]
        Attributes to be passed to the underlying Dataset.
    title: Optional[str]
        Sets the 'title' attribute in the underlying Dataset. Uses the derived class
        name by default.

    Attributes
    ----------
    data: xarray.Dataset
        The underlying Dataset.
    """

    # Define UUID as a class-level variable so that it's fixed during each session.
    __uuid = uuid.uuid4()

    def __init__(
        self,
        data_vars: Optional[Dict[str, Any]] = None,
        coords: Optional[Dict[str, Any]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
    ) -> None:

        # Set default title if the user hasn't provided one
        if title is None:
            title = self.__class__.__name__

        # Initialise attrs to an empty dict if the user hasn't provided one
        if attrs is None:
            attrs = {}

        # Set metadata
        for key, val in self._metadata(title).items():
            attrs[key] = val

        # Set underlying dataset
        self.data = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def _metadata(cls, title: str) -> Dict[str, str]:
        return {
            "title": str(title),
            "software_name": "Pyrokinetics",
            "software_version": __version__,
            "date_created": str(datetime.now()),
            "uuid": str(cls.__uuid),
            "netcdf4_version": nc.__version__,
        }

    def __getattr__(self, attr: str) -> Any:
        """Redirect attribute lookup to self.data"""
        try:
            return getattr(self.data, attr)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )

    def __getitem__(self, key: str) -> Any:
        """Redirect indexing to self.data"""
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(
                f"'{self.__class__.__name__}' object does not contain '{key}'"
            )

    def __str__(self) -> str:
        """Returns stringified xarray Dataset from self.data"""
        return str(self.data)

    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        self.data.to_netcdf(*args, **kwargs)

    @classmethod
    def from_netcdf(
        cls,
        path: PathLike,
        *args,
        overwrite_metadata: bool = False,
        overwrite_title: Optional[str] = None,
        load: bool = False,
        **kwargs,
    ):
        """
        Initialise self.data from a netCDF file.

        Parameters
        ----------

        path: PathLike
            Path to the netCDF file on disk.
        *args:
            Positional arguments forwarded to xarray.open_dataset.
        overwrite_metadata: bool, default False
            Take ownership of the netCDF data, overwriting attributes such as 'title',
            'software_name', 'date_created', etc.
        overwrite_title: Optional[str]
            If ``overwrite_metadata`` is ``True``, this is used to set the ``title``
            attribute in ``self.data``. If unset, the derived class name is used.
        load: bool, default False
            When ``False``, a file handle is kept open and xarray will read lazily
            from disk. When ``True``, all data is loaded into memory.
        **kwargs:
            Keyword arguments forwarded to xarray.open_dataset.

        Returns
        -------
        Derived
            Instance of a derived class with self.data initialised. Derived classes
            which need to do more than this should override this method with their
            own implementation.
        """
        instance = cls.__new__(cls)
        with xr.open_dataset(Path(path), *args, **kwargs) as dataset:
            if overwrite_metadata:
                if overwrite_title is None:
                    title = cls.__name__
                else:
                    title = str(overwrite_title)
                for key, val in cls._metadata(title).items():
                    dataset.attrs[key] = val
            instance.data = dataset.load() if load else dataset
        return instance
