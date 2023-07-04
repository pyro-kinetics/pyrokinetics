from ast import literal_eval
from pathlib import Path
from textwrap import dedent
from typing import Dict, Any, Optional, Mapping

from .metadata import metadata
from .typing import PathLike
from .units import ureg

import netCDF4 as nc
import pint  # noqa
import pint_xarray  # noqa
import xarray as xr


class DatasetWrapper:
    """
    Base class for classes that store their data in an Xarray Dataset. Defines a number
    of useful functions, such as ``__getitem__`` that redirects to the underlying
    Dataset, and methods to read or write to disk. Ensures that the underlying Dataset
    contains metadata about the current session. The user may access the underlying
    Dataset via ``self.data``.

    Parameters
    ----------
    data_vars: Optional[Dict[str, Any]]
        Variables to be passed to the underlying Dataset.
    coords: Optional[Dict[str, Any]]
        Coordinates to be passed to the underlying Dataset.
    attrs: Optional[Dict[str,Any]]
        Attributes to be passed to the underlying Dataset. An associated read-only
        property is created for each attr.
    title: Optional[str]
        Sets the 'title' attribute in the underlying Dataset. Uses the derived class
        name by default.

    Attributes
    ----------
    data: xarray.Dataset
        The underlying Dataset.
    coords: Mapping[str, xarray.DataArray]
        Redirects to the dataset coordinates
    dims: Mapping[str, int]
        Redirects to the dataset dims
    data_vars: Mapping[str, xarray.DataArray]
        Redirects to the dataset data_vars
    attrs: Dict[str, Any]
        Redirects to the dataset attrs
    """

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

        # Save attribute units and strip them from the dict
        # Write attrs to a new dict to avoid modifying the original
        self._attr_units = {}
        new_attrs = {}
        for key, value in attrs.items():
            if hasattr(value, "units") and hasattr(value, "magnitude"):
                self._attr_units[key] = value.units
                new_attrs[key] = value.magnitude
            else:
                new_attrs[key] = value

        # Save _attr_units in the dataset
        new_attrs["attribute_units"] = repr(
            {k: str(v) for k, v in self._attr_units.items()}
        )

        # Add metadata to attrs dict
        obj_name = self.__class__.__name__  # name of derived class, not DatasetWrapper
        meta_dict = metadata(title, obj_name, netcdf4_version=nc.__version__)
        for key, val in meta_dict.items():
            new_attrs[key] = val

        # Set underlying dataset
        self.data = xr.Dataset(data_vars=data_vars, coords=coords, attrs=new_attrs)

    @property
    def data(self) -> xr.Dataset:
        """
        Property for managing the underlying Xarray Dataset. The 'getter' returns
        the Dataset without changes, while the 'setter' uses the pint-array 'quantify'
        function to ensure units attributes are integrated properly.
        """
        return self._data

    @data.setter
    def data(self, ds: xr.Dataset) -> None:
        self._data = ds.pint.quantify(unit_registry=ureg)

    @property
    def coords(self) -> Mapping[str, xr.DataArray]:
        """Redirects to underlying Xarray Dataset coords."""
        return self.data.coords

    @property
    def data_vars(self) -> Mapping[str, xr.DataArray]:
        """Redirects to underlying Xarray Dataset data_vars."""
        return self.data.data_vars

    @property
    def attrs(self) -> Dict[str, Any]:
        """Redirects to underlying Xarray Dataset attrs."""
        return self.data.attrs

    @property
    def dims(self) -> Mapping[str, int]:
        """Redirects to underlying Xarray Dataset dims."""
        return self.data.dims

    def __getitem__(self, key: str) -> Any:
        """Redirect indexing to self.data"""
        try:
            return self.data[key]
        except KeyError:
            raise KeyError(
                f"'{self.__class__.__name__}' object does not contain '{key}'"
            )

    def __getattr__(self, name: str) -> Any:
        """
        Redirect attribute lookup to self.data.attrs.
        Re-assigns units if they were stripped on initialisation.
        """
        try:
            value = self.data.attrs[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        if name in self._attr_units:
            return value * self._attr_units[name]
        else:
            return value

    def __repr__(self) -> str:
        """Returns stringified xarray Dataset from self.data"""
        dataset_repr = repr(self.data)
        my_repr = dataset_repr.replace(
            "<xarray.Dataset>",
            f"<pyrokinetics.{self.__class__.__name__}>\n(Wraps <xarray.Dataset>)",
        )
        return my_repr

    # class: dataset_wrapper
    def __contains__(self, name: str) -> bool:
        return name in self.data


    def to_netcdf(self, *args, **kwargs) -> None:
        """Writes self.data to disk. Forwards all args to xarray.Dataset.to_netcdf."""
        data = self.data.expand_dims("ReIm", axis=-1)  # Add ReIm axis at the end
        data = xr.concat([data.real, data.imag], dim="ReIm")

        data.pint.dequantify().to_netcdf(*args, **kwargs)

    @classmethod
    def from_netcdf(
        cls,
        path: PathLike,
        *args,
        overwrite_metadata: bool = False,
        overwrite_title: Optional[str] = None,
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
        **kwargs:
            Keyword arguments forwarded to xarray.open_dataset.

        Returns
        -------
        Derived
            Instance of a derived class with self.data initialised. Derived classes
            which need to do more than this should override this method with their
            own implementation.

        Raises
        ------
        RuntimeError
            If the netcdf is for the wrong type of object.
        """
        instance = cls.__new__(cls)
        with xr.open_dataset(Path(path), *args, **kwargs) as dataset:
            if dataset.attrs.get("object_type", cls.__name__) != cls.__name__:
                raise RuntimeError(
                    dedent(
                        f"""\
                        netcdf of type {dataset.attrs["object_type"]} cannot be used
                        to create objects of type {cls.__name__}.
                        """
                    )
                )
            if overwrite_metadata:
                if overwrite_title is None:
                    title = cls.__name__
                else:
                    title = str(overwrite_title)
                new_metadata = metadata(
                    title, cls.__name__, netcdf4_version=nc.__version__
                )
                for key, val in new_metadata.items():
                    dataset.attrs[key] = val
            instance.data = dataset
        # Set up attr_units
        attr_units_as_str = literal_eval(dataset.attribute_units)
        instance._attr_units = {k: ureg(v).units for k, v in attr_units_as_str.items()}
        attrs = instance.attrs


        # isel drops attrs so need to add back in
        instance.data = instance.data.isel(ReIm=0) + 1j * instance.data.isel(ReIm=1)
        instance.data.attrs = attrs

        return instance
