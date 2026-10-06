from __future__ import annotations

import copy
import json
import os
import pathlib
import warnings
from contextlib import contextmanager
from functools import partial, reduce
from itertools import product
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np
import pint
import pint_xarray  # noqa: F401 (registers the .pint accessor)
import xarray as xr
from pint import Quantity

from .dataset_wrapper import DatasetWrapper
from .gk_code import GKInput
from .normalisation import ConventionNormalisation
from .pyro import Pyro
from .units import ureg


def _serialize_path(path: pathlib.Path, base: pathlib.Path) -> str:
    """Convert absolute path to relative-for-JSON."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _resolve_path(path: str | pathlib.Path, base: pathlib.Path) -> pathlib.Path:
    """Resolve possibly-relative path against a base directory."""
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


# ---- time handling ----
VALID_TIME_MODES = ("last", "average", "trace")


def reduce_time(
    da,
    *,
    time_mode,
    tolerance_time_range=None,
):
    """
    Reduce a DataArray over time according to the chosen policy.

    time_mode:
        "last"      → take final time
        "average"   → average over tolerance_time_range
        "trace"     → preserve the full time series unchanged
    """
    if "time" not in da.dims:
        return da

    if time_mode == "last":
        return da.isel(time=-1).drop_vars("time", errors="ignore")

    if time_mode == "average":
        if tolerance_time_range is None:
            raise ValueError("tolerance_time_range required for time averaging")

        t = da["time"]
        t_max = float(t.max())
        t_min = t_max * tolerance_time_range

        return (
            da.sel(time=slice(t_min, t_max))
            .mean(dim="time")
            .drop_vars("time", errors="ignore")
        )

    if time_mode == "trace":
        return da

    raise ValueError(
        f"Unknown time_mode={time_mode!r}; expected one of {VALID_TIME_MODES}"
    )


# ---- xarray selection ----
def select_kx_ky_time(
    da,
    *,
    kx_min=None,
    sum_ky=False,
    sum_kx=False,
    time_mode,
    tolerance_time_range=None,
):
    if "kx" in da.dims:
        if sum_kx:
            da = da.sum(dim="kx")
        elif kx_min is not None:
            da = da.sel(kx=kx_min)

    da = reduce_time(
        da,
        time_mode=time_mode,
        tolerance_time_range=tolerance_time_range,
    )
    if sum_ky and "ky" in da.dims:
        da = da.sum(dim="ky")

    return da


# ---- error handling ----
def handle_failed_run(buffers, templates, gk_file, error=None):
    import warnings

    warnings.warn(
        f"Failed to load GK output for {gk_file}: {type(error).__name__}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )

    for name, buf in buffers.items():
        buf.append(templates[name])


def normalize_failed_runs(buffers: dict[str, list]) -> None:
    """
    Replace None placeholders (from early failures) with NaN-filled DataArrays
    once a reference shape is available.
    """
    for name, values in buffers.items():
        ref = next((v for v in values if v is not None), None)
        if ref is None:
            continue

        buffers[name] = [xr.full_like(ref, np.nan) if v is None else v for v in values]


# ---- dataset assembly ----
def _is_numeric(values):
    return np.issubdtype(np.asarray(values).dtype, np.number)


def _same_axis(a, b, tol):
    if _is_numeric(a) and _is_numeric(b):
        return np.shape(a) == np.shape(b) and np.all(np.abs(a - b) <= tol)
    return np.array_equal(a, b)


def _union_axis(axes, tol):
    """
    The union of several runs' values of one coordinate.

    Numeric values are sorted and values within ``tol`` of each other are
    merged, so grids of different resolution interleave in order rather than
    one being appended to the other. Other values keep first-seen order.
    """
    if all(_is_numeric(a) for a in axes):
        union = []
        for v in np.sort(np.concatenate([np.ravel(a) for a in axes])):
            if not union or v - union[-1] > tol:
                union.append(v)
        return np.asarray(union)

    return np.asarray(list(dict.fromkeys(v for a in axes for v in np.ravel(a))))


def _positions(values, axis, tol):
    """Index of each of ``values`` on ``axis``."""
    if _is_numeric(axis):
        idx = np.clip(np.searchsorted(axis, values - tol), 0, len(axis) - 1)
        if np.any(np.abs(axis[idx] - values) > tol):
            raise ValueError("Coordinate value not found on the merged axis")
        return idx
    lookup = {v: i for i, v in enumerate(axis)}
    return np.asarray([lookup[v] for v in np.ravel(values)])


def stack_runs(arrays, last):
    """
    Put every run's array for one quantity onto a single set of non-scan axes.

    Runs of a scan need not share a grid: TGLF runs at ``NMODES=2`` and ``4``
    give ``mode`` axes of different length, runs at different resolution give
    different ``theta`` grids, and runs can share a shape but not its values
    (e.g. ``kx`` in a ``theta0`` scan). Where the runs agree they are stacked
    as they are. Otherwise each dimension with a coordinate becomes the sorted
    union of the runs' values (matched to a relative tolerance), and a run is
    NaN wherever it has no value; a dimension without a coordinate is padded
    to the longest run.

    Returns ``(magnitudes, units, coords, shape)``: one array per run on the
    common axes, their pint units (``None`` if unitless), and the coordinates
    and shape of those axes, in the order of ``last.dims``.
    """
    dims = last.dims
    for a in arrays:
        if set(a.dims) != set(dims):
            raise ValueError(
                f"Runs have different dimensions: {a.dims} and {dims}; "
                "they cannot be stacked."
            )
    arrays = [a.transpose(*dims) for a in arrays]

    raw, units = [], None
    for a in arrays:
        data = a.data
        if hasattr(data, "magnitude"):
            units = data.units
            data = data.magnitude
        raw.append(np.asarray(data))

    coord_dims = [dim for dim in dims if dim in last.coords]
    run_axes = {dim: [np.asarray(a[dim].values) for a in arrays] for dim in coord_dims}
    # Coordinate values match to a tolerance relative to their largest value
    tols = {}
    for dim, axes in run_axes.items():
        if all(_is_numeric(x) for x in axes):
            scale = max((np.max(np.abs(a)) for a in axes if np.size(a)), default=0.0)
            tols[dim] = 1e-8 * scale
        else:
            tols[dim] = 0.0

    aligned = len({r.shape for r in raw}) == 1 and all(
        _same_axis(x, run_axes[dim][-1], tols[dim])
        for dim in coord_dims
        for x in run_axes[dim]
    )
    if aligned:
        return raw, units, dict(last.coords), last.shape

    axes = {}
    for i, dim in enumerate(dims):
        if dim in coord_dims:
            axes[dim] = _union_axis(run_axes[dim], tols[dim])
        else:
            axes[dim] = np.arange(max(r.shape[i] for r in raw))
    shape = tuple(len(axes[dim]) for dim in dims)

    dtype = complex if any(np.iscomplexobj(r) for r in raw) else float
    padded = []
    for a, r in zip(arrays, raw):
        out = np.full(shape, np.nan, dtype=dtype)
        take = tuple(
            (
                _positions(np.asarray(a[dim].values), axes[dim], tols[dim])
                if dim in coord_dims
                else np.arange(r.shape[i])
            )
            for i, dim in enumerate(dims)
        )
        out[np.ix_(*take)] = r
        padded.append(out)

    return padded, units, {dim: axes[dim] for dim in coord_dims}, shape


def integrate_over_valid_range(da, dim="theta"):
    """
    Trapezoid integral of ``da`` along ``dim``, over each run's own valid range.

    Runs stacked on a merged axis are NaN off their own points: in gaps inside
    their range (a coarser grid) and beyond its ends (a shorter range). Gaps are
    filled by linear interpolation, for which the trapezoid rule is exact, so
    they do not change the integral. Only segments with both ends valid are
    then summed, so each run is integrated exactly over its own range and never
    extrapolated beyond it. A run with no valid segment gives NaN.

    The pint accessor keeps units that plain ``interpolate_na`` would strip.
    """
    filled = da.pint.interpolate_na(dim, method="linear")
    units = getattr(filled.data, "units", None)
    y = filled.copy(data=getattr(filled.data, "magnitude", filled.data))
    x = y[dim]

    segments = 0.5 * (x.shift({dim: -1}) - x) * (y + y.shift({dim: -1}))
    integral = segments.sum(dim, skipna=True, min_count=1)

    return integral if units is None else integral * units


def add_quantity(
    ds, name, arrays, base_shape, scan_coords, scan_dims=None, squeeze_dims=None
):
    """
    Stack one quantity from every run of a scan into ``ds``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add the quantity to.
    name : str
        Name of the quantity.
    arrays : list
        One entry per run, in run order.
    base_shape : tuple
        Shape of the scan itself, which the run entries are reshaped onto: the
        lengths of the scan dimensions, in ``scan_dims`` order.
    scan_coords : dict
        Coordinates describing the scan points, in any form accepted by xarray.
        In a gridded scan these are the scan dimensions themselves; in a
        non-gridded one (see ``PyroHypercube``) the varied parameters are
        non-dimension coordinates along a single sample dimension.
    scan_dims : tuple, default None
        Names of the scan dimensions. Defaults to the keys of ``scan_coords``,
        i.e. the gridded case.
    squeeze_dims : iterable, default None
        Names of varied parameters that may also appear as a dimension of an
        individual run's output (e.g. ``ky``), and so must be squeezed out of it
        before stacking. Defaults to the keys of ``scan_coords``.
    """
    if not any(isinstance(a, xr.DataArray) for a in arrays):
        return ds

    if scan_dims is None:
        scan_dims = tuple(scan_coords.keys())
    if squeeze_dims is None:
        squeeze_dims = tuple(scan_coords.keys())

    last = arrays[-1]
    for dim in squeeze_dims:
        if dim in last.dims:
            last = last.squeeze(dim, drop=True)
            arrays = [
                a.squeeze(dim, drop=True) if isinstance(a, xr.DataArray) else a
                for a in arrays
            ]

    raw, units, run_coords, run_shape = stack_runs(arrays, last)

    shape = base_shape + run_shape
    stacked = np.stack(raw).reshape(shape)

    if units is not None:
        stacked = stacked * units

    dims = tuple(scan_dims) + last.dims

    arr = xr.DataArray(
        stacked,
        dims=dims,
        coords={
            **scan_coords,
            **run_coords,
        },
    )

    arr = arr.reset_coords(drop=True)
    ds[name] = arr

    return ds


def coord_quantity(coord):
    """
    A coordinate's values as a pint ``Quantity``, or ``None`` if it has no units.

    Units are held in one of two places depending on the coordinate. An xarray
    index cannot hold a pint array, so a dimension coordinate keeps its units in
    ``attrs``; a non-dimension coordinate — how ``PyroHypercube`` carries its
    varied parameters — is quantified in place.
    """
    data = coord.data
    if hasattr(data, "units"):
        return data

    units = coord.attrs.get("units", None)
    if units is None:
        return None
    if isinstance(units, str):
        units = ureg(units).units
    return data * units


def magnitude_and_units(values):
    """
    Split a scan parameter's values into a plain array and a unit.

    Unitless scan parameters are stored as plain arrays/lists, and are given
    dimensionless units.
    """
    if isinstance(values, Quantity):
        return np.asarray(values.magnitude), values.units
    return np.asarray(values), ureg.dimensionless


class ScanLayout(NamedTuple):
    """
    How the runs of a scan map onto the dimensions of its output Dataset.

    A gridded ``PyroScan`` has one dimension per varied parameter; a
    non-gridded ``PyroHypercube`` has a single sample dimension carrying every
    varied parameter as a non-dimension coordinate. Everything else about
    loading a scan's outputs is common to both, so only this mapping is
    overridden.
    """

    #: Coordinates of the output Dataset, in any form accepted by xarray.
    coords: Dict[str, Any]
    #: Units of each coordinate that has them, keyed by coordinate name.
    coord_units: Dict[str, Any]
    #: Lengths of the scan dimensions, in ``dims`` order.
    shape: Tuple[int, ...]
    #: Names of the scan dimensions.
    dims: Tuple[str, ...]
    #: Varied parameters that may clash with a dimension of a single run.
    squeeze_dims: Tuple[str, ...]


class PyroScan:
    """
    Creates a dictionary of pyro objects in pyro_dict

    Need a templates pyro object

    Dict of parameters to scan through
    { param : [values], }
    """

    JSON_ATTRS = [
        "value_fmt",
        "value_separator",
        "parameter_separator",
        "parameter_dict",
        "file_name",
        "base_directory",
        "runfile_dict",
        "p_prime_type",
        "parameter_map",
    ]

    def __init__(
        self,
        pyro=None,
        parameter_dict=None,
        p_prime_type=0,
        value_fmt=".2f",
        value_separator="_",
        parameter_separator="/",
        file_name=None,
        base_directory=".",
        load_default_parameter_keys=True,
        pyroscan_json=None,
        runfile_dict=None,
        load_base_pyro=False,
    ):
        # Mapping from parameter to location in Pyro
        self.parameter_map = {}

        # Need to intialise and pyro_dict pyroscan_json before base_directory
        self.pyro_dict = {}
        self.pyroscan_json = {}
        self.parameter_func = {}

        self.base_directory = base_directory

        # Format values/parameters
        self.value_fmt = value_fmt

        self.value_separator = value_separator

        if parameter_separator in ["/", "\\"]:
            self.parameter_separator = os.path.sep
        else:
            self.parameter_separator = parameter_separator

        self.runfile_dict = runfile_dict or {}

        self.base_directory = pathlib.Path(base_directory).resolve()

        if file_name is not None:
            self.file_name = file_name
        elif pyro is not None:
            self.file_name = GKInput._factory[pyro.gk_code].default_file_name

        if parameter_dict is None:
            self.parameter_dict = {}
        else:
            self.parameter_dict = parameter_dict

        self.p_prime_type = p_prime_type

        # Load in pyroscan json if there
        if pyroscan_json is not None:
            pyroscan_json = pathlib.Path(pyroscan_json).resolve()
            with open(pyroscan_json) as f:
                self.pyroscan_json = json.load(f)

            json_dir = pyroscan_json.parent
            for key, value in self.pyroscan_json.items():
                if key == "parameter_dict":
                    self.parameter_dict = {
                        k: (
                            np.asarray(raw[0]) * ureg(raw[1])
                            if isinstance(raw, list)
                            and len(raw) == 2
                            and isinstance(raw[1], str)
                            else raw
                        )
                        for k, raw in value.items()
                    }
                    self.pyroscan_json["parameter_dict"] = self.parameter_dict
                    continue
                elif key == "parameter_func":
                    self.parameter_func = {k: tuple(v) for k, v in value.items()}
                    continue
                elif key == "base_directory":
                    # Resolve relative path against JSON location
                    resolved = _resolve_path(value, json_dir)

                    # User override still wins
                    if base_directory != ".":
                        resolved = pathlib.Path(base_directory).resolve()

                    setattr(self, key, resolved)
                    continue

                setattr(self, key, value)
        else:
            self.pyroscan_json = {attr: getattr(self, attr) for attr in self.JSON_ATTRS}

        if pyro is not None:
            if not isinstance(pyro, Pyro):
                raise TypeError("pyro must be a Pyro instance")
            self.base_pyro = pyro
        elif self.file_name is None:
            raise ValueError(
                "file_name must be specified or in json if pyro is not given"
            )
        elif load_base_pyro:
            pyro_base = pathlib.Path(pyroscan_json).resolve().parent
            in_loc = pyro_base / "pyroscan_base.input"
            self.base_pyro = Pyro(gk_file=in_loc)

            # Restore normalisation reference values if they were saved.
            # This is needed when the base input file type (e.g. TGLF)
            # cannot store normalisations that the original pyro had
            # (e.g. from a GENE run).
            norms_file = pyro_base / "pyroscan_norms.json"
            if norms_file.exists():
                self.base_pyro.read_reference_values(norms_file)
        else:
            raise ValueError("Either provide a pyro object or enable load_base_pyro")

        if (
            load_default_parameter_keys and pyroscan_json is None
        ):  # if parameter keys are loaded from json there is no need to set defaults
            self.load_default_parameter_keys()

        # Canonicalise freshly-supplied parameter_dict values into pyrokinetics
        # simulation units so run-directory names are consistent regardless of
        # gk_code convention. On reload the JSON is authoritative — its values
        # already reflect whatever unit was serialised, so we skip conversion
        # there (new-format JSONs are already in pyro sim units, and old-format
        # fixtures pair their magnitudes with their on-disk directory names).
        if pyroscan_json is None:
            # Pass the pyrokinetics convention explicitly: base_pyro.norms.default_convention
            # is set to whichever gk_code was read (gs2/gene/cgyro/...), which would
            # otherwise make run-directory names code-dependent.
            pyro_convention = self.base_pyro.norms.pyrokinetics
            for name, values in list(self.parameter_dict.items()):
                if hasattr(values, "convert_physical_units"):
                    self.parameter_dict[name] = values.convert_physical_units(
                        pyro_convention
                    )

        # Get len of values for each parameter
        self.value_size = [len(value) for value in self.parameter_dict.values()]

        self.build_pyro_dict()

    def build_pyro_dict(self):
        """
        Create one Pyro per run of the scan, and record their run directories.
        """
        if not self.runfile_dict:
            self.ensure_unique_run_names()

        self.pyro_dict = dict(
            self.create_single_run(run) for run in self.outer_product()
        )
        self.run_directories = [pyro.run_directory for pyro in self.pyro_dict.values()]

    def _format_single_run_name_with_fmt(self, parameters, fmt):
        return self.parameter_separator.join(
            f"{param}{self.value_separator}{getattr(value, 'magnitude', value):{fmt}}"
            for param, value in parameters.items()
        )

    def _hashable_value(self, value):
        """
        Convert scan parameter values into hashable objects for duplicate detection.
        Handles pint Quantities, numpy scalars, numpy arrays, and plain Python values.
        """
        if isinstance(value, Quantity):
            value = value.magnitude

        value = np.asarray(value)

        if value.shape == ():
            return value.item()

        return tuple(value.ravel().tolist())

    def _scan_point_key(self, parameters):
        """
        Canonical key for detecting genuinely duplicated scan points.
        """
        return tuple(
            (param, self._hashable_value(value)) for param, value in parameters.items()
        )

    def ensure_unique_run_names(self):
        """
        Ensure automatically generated run directory names are unique across
        the whole scan. If close values collide due to rounding, increase
        value_fmt precision. If scan points are genuinely duplicated, raise.
        """
        runs = list(self.outer_product())

        # First detect genuinely duplicated scan points.
        scan_keys = [self._scan_point_key(run) for run in runs]

        if len(scan_keys) != len(set(scan_keys)):
            raise ValueError(
                "duplicate scan points detected; cannot generate unique directories"
            )

        fmt = self.value_fmt

        def run_names(fmt):
            return [self._format_single_run_name_with_fmt(run, fmt) for run in runs]

        names = run_names(fmt)

        while len(names) != len(set(names)):
            warnings.warn(
                f"Rounding collision detected in generated run directories; "
                f"increasing precision to ensure unique directory names: {fmt}",
                UserWarning,
            )

            if "." in fmt:
                p = int(fmt.split(".")[1][:-1]) + 1
                fmt = f".{p}f"
            else:
                fmt = ".3f"

            names = run_names(fmt)

        self.value_fmt = fmt
        self.pyroscan_json["value_fmt"] = fmt

    def format_single_run_name(self, parameters):
        """
        Concatenate parameter names/values with separator.
        Handles both tuple-style and string-style runfile_dict keys for backward compatibility.
        """
        if self.runfile_dict:
            # Generate the string form of the key
            key_str = "_".join(
                f"{k}_{v.magnitude if isinstance(v, Quantity) else v}"
                for k, v in parameters.items()
            )
            # Since when you load a file parameters are given units you need to remove units before formatting into a string
            # --- Backward compatibility layer ---
            # Check if the runfile_dict still uses tuple keys
            if key_str not in self.runfile_dict:
                # Try matching the tuple version if it exists
                tuple_key = tuple(
                    f"{k}_{v.magnitude if isinstance(v, Quantity) else v}"
                    for k, v in parameters.items()
                )
                if tuple_key in self.runfile_dict:
                    # Convert the entire dict to string keys for future use
                    self.runfile_dict = {
                        "_".join(k): v if isinstance(k, tuple) else v
                        for k, v in self.runfile_dict.items()
                    }
                else:
                    raise KeyError(
                        f"Runfile key not found for parameters: {parameters}. "
                        f"Tried both '{key_str}' and {tuple_key}."
                        f"This comes from the runfile_dict {self.runfile_dict}."
                    )

            # Ensure we always save the runfile_dict into the JSON
            self.pyroscan_json["runfile_dict"] = self.runfile_dict

            # Return the value (now guaranteed to exist)
            return self.runfile_dict[key_str]

        else:

            return self.parameter_separator.join(
                f"{param}{self.value_separator}{getattr(value, 'magnitude', value):{self.value_fmt}}"
                for param, value in parameters.items()
            )

    def create_single_run(self, parameters: dict, name: str = None):
        """
        Create a new Pyro instance from the PyroScan base with new run parameters

        Parameters
        ----------
        parameters: dict
            Parameter values for this run.
        name: str, default None
            Name of the run directory. If unset, it is generated from the
            parameter values by ``format_single_run_name``.
        """
        if name is None:
            name = self.format_single_run_name(parameters)
        new_run = copy.deepcopy(self.base_pyro)

        new_run.gk_file = self.base_directory / name / self.file_name
        new_run.run_parameters = copy.deepcopy(parameters)
        return name, new_run

    def write(
        self,
        file_name=None,
        base_directory=None,
        template_file=None,
        relative_path=True,
    ):
        """
        Creates and writes GK input files for parameters in scan
        """

        if file_name is not None:
            self.file_name = file_name

        if base_directory is not None:
            self.base_directory = pathlib.Path(base_directory)

            # Set run directories
            self.run_directories = [
                self.base_directory / run_dir for run_dir in self.pyro_dict.keys()
            ]

        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Dump json file with pyroscan data
        json_file = self.base_directory / "pyroscan.json"

        json_data = dict(self.pyroscan_json)

        unsaved = [
            k for k, (f, _) in self.parameter_func.items() if not isinstance(f, str)
        ]
        if unsaved:
            warnings.warn(
                f"parameter_func for {unsaved} are functions and are not saved to "
                "pyroscan.json; pass the name of a Pyro method to make them reloadable.",
                stacklevel=2,
            )

        if relative_path:
            json_data["base_directory"] = "."
        else:
            json_data["base_directory"] = str(self.base_directory)

        # Convert parameter_dict values to generic simulation units so the
        # JSON carries run-independent unit names; reloading pairs them back
        # with physical values via pyroscan_norms.json.
        if "parameter_dict" in json_data:
            norms = self.base_pyro.norms.pyrokinetics
            json_data["parameter_dict"] = {
                name: (
                    values.convert_physical_units(norms)
                    if hasattr(values, "convert_physical_units")
                    else values
                )
                for name, values in json_data["parameter_dict"].items()
            }

        with open(json_file, "w+") as f:
            json.dump(json_data, f, cls=NumpyEncoder)

        self.update_self_parameters()

        # Iterate through all runs and write output
        for parameter, run_dir, pyro in zip(
            self.outer_product(), self.run_directories, self.pyro_dict.values()
        ):
            # Write input file
            pyro.write_gk_file(
                file_name=run_dir / self.file_name, template_file=template_file
            )

        self.base_pyro.write_gk_file(
            file_name=self.base_directory / "pyroscan_base.input"
        )

        # Save normalisation reference values so they can be restored
        # when reloading from a base input file that cannot store them
        # (e.g. TGLF inputs generated from a GENE run)

        try:
            self.base_pyro.write_reference_values(
                self.base_directory / "pyroscan_norms.json"
            )
        except Exception:
            warnings.warn(
                "Could not save normalisation reference values to "
                "pyroscan_norms.json. Normalisation-dependent units "
                "may not be available when reloading this PyroScan.",
                stacklevel=2,
            )

    def _apply_parameters(self, parameter: dict, pyro: Pyro) -> None:
        """Apply one run's scanned ``parameter`` values (and any ``parameter_func``) to ``pyro``."""
        for param, value in parameter.items():
            # Get attribute name and keys where param is stored in Pyro
            attr_name, keys_to_param = self.parameter_map[param]

            # Scan values are stored in generic simulation units; a run
            # with physical reference values needs them in its own units
            if isinstance(value, Quantity):
                value = value.to(pyro.norms.pyrokinetics)

            # Get attribute in Pyro storing the parameter
            pyro_attr = getattr(pyro, attr_name)

            # Set the value given the Pyro attribute and location of parameter
            set_in_dict(pyro_attr, keys_to_param, value)

            if param in self.parameter_func.keys():
                func, kwargs = self.parameter_func[param]
                # A string names a Pyro method, so it can be saved to pyroscan.json
                (getattr(pyro, func) if isinstance(func, str) else partial(func, pyro))(
                    **(kwargs or {})
                )

    def update_self_parameters(
        self,
    ):
        """
        Updates all pyro object parameters based on pyro_dict values
        """
        for parameter, pyro in zip(self.outer_product(), self.pyro_dict.values()):
            self._apply_parameters(parameter, pyro)

    def sample_pyro(self, sample, gk_output=None) -> Pyro:
        """
        The Pyro of one run of this scan, with that run's parameters applied
        (``parameter_map`` and ``parameter_func``) and, if there is output, its
        ``gk_output`` slice attached, so diagnostics can be run on it unchanged.

        Works on a scan loaded from ``pyroscan.json``: nothing is read from the
        run directories.

        Parameters
        ----------
        sample: int or str
            Position of the run in ``pyro_dict``, or its name.
        gk_output: xarray.Dataset, default None
            Scan output to slice. Defaults to ``self.gk_output.data`` if loaded,
            and no output is attached if neither exists. A hypercube is sliced
            along ``sample``; a gridded scan along each scanned parameter.

        Returns
        -------
        Pyro
            A copy, so the scan's own ``pyro_dict`` is left untouched.
        """
        names = list(self.pyro_dict)
        i = names.index(sample) if isinstance(sample, str) else int(sample)
        pyro = copy.deepcopy(self.pyro_dict[names[i]])
        parameter = next(p for j, p in enumerate(self.outer_product()) if j == i)
        self._apply_parameters(parameter, pyro)

        if gk_output is None and getattr(self, "gk_output", None) is not None:
            gk_output = self.gk_output.data
        if gk_output is not None:
            if "sample" in gk_output.dims:
                gk_output = gk_output.isel(sample=i)
            else:
                gk_output = gk_output.sel(
                    {
                        k: getattr(v, "m", v)
                        for k, v in parameter.items()
                        if k in gk_output.dims
                    },
                    method="nearest",
                )
            pyro.gk_output = gk_output
        return pyro

    def add_parameter_key(
        self, parameter_key=None, parameter_attr=None, parameter_location=None
    ):
        """
        parameter_key: string to access variable
        parameter_attr: string of attribute storing value in pyro
        parameter_location: list of strings showing path to value in pyro
        """

        if parameter_key is None:
            raise ValueError("Need to specify parameter key")

        if parameter_attr is None:
            raise ValueError("Need to specify parameter attr")

        if parameter_location is None:
            raise ValueError("Need to specify parameter location")

        dict_item = {parameter_key: [parameter_attr, parameter_location]}

        self.parameter_map.update(dict_item)

        # Get attribute name and keys where param is stored in Pyro

        pyro_attr = getattr(self.base_pyro, parameter_attr)
        if parameter_key in self.parameter_dict:
            value = self.parameter_dict[parameter_key]

            if not hasattr(value, "units"):
                units = getattr(
                    get_from_dict(pyro_attr, parameter_location[:-1])[
                        parameter_location[-1]
                    ],
                    "units",
                    1,
                )
                if units != 1:
                    warnings.warn(
                        f"Adding units [{units}] to {parameter_key} as it has not been "
                        "specified. To suppress this warning please add units"
                    )

                    self.parameter_dict[parameter_key] = value * units

        self.pyroscan_json["parameter_map"] = self.parameter_map

    def add_parameter_func(
        self, parameter_key=None, parameter_func=None, parameter_kwargs=None
    ):
        """
        Applies function `parameter_func(pyro, **kwargs)` on pyro object each time after
        parameter_key is set in a scan

        parameter_key: string to access variable
        parameter_func: function that take in a pyro object applies modification, or
            the name of a Pyro method (e.g. ``"enforce_consistent_beta_prime"``).
            Only the name form is saved to ``pyroscan.json``, so a scan reloaded
            from it applies the same derived settings; a function is not saved.
        parameter_kwargs: Dictionary of kwargs to apply to function
        """

        self.parameter_func[parameter_key] = (parameter_func, parameter_kwargs)
        named = {
            k: [f, kw or {}]
            for k, (f, kw) in self.parameter_func.items()
            if isinstance(f, str)
        }
        if named:
            self.pyroscan_json["parameter_func"] = named
        else:
            self.pyroscan_json.pop("parameter_func", None)

    def load_default_parameter_keys(self):
        """
        Loads default parameters name into parameter_map

        {param : ["attribute", ["key_to_location_1", "key_to_location_2" ]] }

        for example

        {'electron_temp_gradient': [
            "local_species", ['electron','inverse_lt']] }
        """

        self.parameter_map = {}

        # ky
        parameter_key = "ky"
        parameter_attr = "numerics"
        parameter_location = ["ky"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Electron temperature gradient
        parameter_key = "electron_temp_gradient"
        parameter_attr = "local_species"
        parameter_location = ["electron", "inverse_lt"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Electron density gradient
        parameter_key = "electron_dens_gradient"
        parameter_attr = "local_species"
        parameter_location = ["electron", "inverse_ln"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Deuterium temperature gradient
        parameter_key = "deuterium_temp_gradient"
        parameter_attr = "local_species"
        parameter_location = ["deuterium", "inverse_lt"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Deuterium density gradient
        parameter_key = "deuterium_dens_gradient"
        parameter_attr = "local_species"
        parameter_location = ["deuterium", "inverse_ln"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # ExB shear
        parameter_key = "gamma_exb"
        parameter_attr = "numerics"
        parameter_location = ["gamma_exb"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

        # Elongation
        parameter_key = "kappa"
        parameter_attr = "local_geometry"
        parameter_location = ["kappa"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

    def load_gk_output(
        self,
        output_convention="pyrokinetics",
        tolerance_time_range=0.8,
        time_mode="average",
        netcdf_file=None,
        load_fields=True,
        eigenvalues_from_fields=False,
        load_fluxes=True,
        load_moments=False,
        sum_ky=False,
        sum_kx=False,
        drop_nan=False,
        load_tearing_parameter=False,
        **kwargs,
    ):
        """
        Loads PyroScanGKOutput into self.gk_output

        Parameters
        ----------
        output_convention: str default 'pyrokinetics'
            ConventionNormalisation to convert output to
        tolerance_time_range: float default 0.8
            Start fraction of the time axis used for time averages and the
            growth rate tolerance (i.e. ``tolerance_time_range=0.8`` averages
            over the last 20% of the simulation).
        time_mode: str default 'average'
            How to reduce the outputs along the time axis, whether the runs are
            linear or nonlinear. One of:

            * ``"average"`` – mean over ``[tolerance_time_range * t_max, t_max]``.
              Fields and eigenfunctions are complex, and their time mean is
              meaningless (linear fields are normalised to their final
              amplitude; the phase of a nonlinear Fourier coefficient keeps
              moving), so for them ``|field|**2`` is averaged instead, taken
              before any kx/ky sum. It is stored as ``phi_squared``,
              ``apar_squared``, ``bpar_squared`` and ``eigenfunctions_squared``
              (with squared units), not under the field names.
            * ``"last"``    – take the final time point.
            * ``"trace"``   – preserve the full time trace, no reduction.
        netcdf_file: PathLike default None
            If supplied then load PyroScanGKOutput from existing netCDF
        load_fields (bool, default True) – Flag to load fields or not
        eigenvalues_from_fields (bool, default False) – With ``load_fields=False``,
            still read each run's fields to derive growth_rate, mode_frequency and
            growth_rate_tolerance from them (the smooth path ``load_fields=True``
            takes; otherwise a code such as GS2 reports its own, noisier
            eigenvalue output), then discard the fields run by run so only the
            scalars are stored. Has no effect when ``load_fields=True``.
        load_fluxes (bool, default True) – Flag to load fluxes or not
        load_moments (bool, default False) – Flag to load moments or not
        sum_ky (bool, default False) – If True, sum fluxes, fields and eigenfunctions
            over ky. If False, preserve the ky dimension.
        sum_kx (bool, default False) – Applies to fields and eigenfunctions (fluxes
            are already kx-integrated). If True, sum over kx; if False, preserve the
            kx dimension.
        drop_nan (bool, default False) – If NaNs are found in the output then that data is dropped. Off by default
        load_tearing_parameter (bool, default False) – Linear runs only. Store
            ``FieldLine.compute_linear_tearing_parameter`` of every run, computed
            with that run's own geometry before any time reduction, as
            ``tearing_parameter`` (one value per mode for TGLF/GFTM), and
            ``FieldLine.compute_linear_parity`` as ``apar_even_fraction``
            (> 0.5 is tearing parity), and the same for phi as ``phi_even_fraction``
            (> 0.5 is ballooning-like phi parity).
        **kwargs – Arguments to pass to the GKOutputReader.
        Returns
        -------
        None
        """
        if time_mode not in VALID_TIME_MODES:
            raise ValueError(
                f"time_mode={time_mode!r} is not valid; "
                f"expected one of {VALID_TIME_MODES}"
            )

        # Load from netCDF is supplied
        if netcdf_file is not None:
            # Auto-detect a pyroscan_norms.json sitting alongside the scan
            # so the fresh base_pyro can restore its physical reference
            # values. Without this, generic simulation units saved to the
            # netCDF cannot be converted back to physical units.
            netcdf_path = pathlib.Path(netcdf_file)
            for candidate in (
                self.base_directory / "pyroscan_norms.json",
                netcdf_path.parent / "pyroscan_norms.json",
            ):
                if candidate.exists():
                    try:
                        self.base_pyro.read_reference_values(candidate)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to apply reference values from "
                            f"{candidate}: {type(e).__name__}: {e}",
                            stacklevel=2,
                        )
                    break

            convention = getattr(self.base_pyro.norms, output_convention)
            gk_output = PyroScanGKOutput.from_netcdf(netcdf_file)
            gk_output._norms = self.base_pyro.norms
            gk_output.to(convention, convention.context)
            self.gk_output = gk_output
            return

        layout = self.output_layout()

        load_specs = {
            "linear": {
                "scalars": ["growth_rate", "mode_frequency"],
                "extras": ["growth_rate_tolerance"],
                "fluxes": [],
                # Eigenfunctions are selected exactly as the fields are
                "fields": ["eigenfunctions"],
            },
            "nonlinear": {
                "scalars": [],
                "extras": [],
                "fluxes": [],
                "fields": [],
            },
        }

        if self.base_pyro.gk_code == "TGLF":
            load_specs["nonlinear"]["scalars"].extend(["growth_rate", "mode_frequency"])
            load_specs["nonlinear"]["extras"].extend(["growth_rate_tolerance"])

        if load_fluxes:
            load_specs["linear"]["fluxes"].extend(["particle", "heat", "momentum"])
            load_specs["nonlinear"]["fluxes"].extend(["particle", "heat", "momentum"])

        if load_fields:
            load_specs["linear"]["fields"].extend(["phi", "bpar", "apar"])
            load_specs["nonlinear"]["fields"].extend(["phi", "bpar", "apar"])

        regime = "nonlinear" if self.base_pyro.numerics.nonlinear else "linear"
        spec = load_specs[regime]
        tearing = load_tearing_parameter and regime == "linear"
        if tearing:
            from .diagnostics.field_line import FieldLine

            spec["scalars"] += [
                "tearing_parameter",
                "apar_even_fraction",
                "phi_even_fraction",
            ]

        buffers = {
            name: []
            for name in (
                spec["scalars"] + spec["extras"] + spec["fluxes"] + spec["fields"]
            )
        }
        # Fields averaged as |field|**2, which are renamed once stacked
        squared = set()

        for i, pyro in enumerate(self.pyro_dict.values()):
            run_buffers = {name: None for name in buffers}
            try:
                pyro.load_gk_output(
                    output_convention=output_convention,
                    load_fields=load_fields or eigenvalues_from_fields,
                    load_fluxes=load_fluxes,
                    load_moments=load_moments,
                    drop_nan=drop_nan,
                    **kwargs,
                )
                data = pyro.gk_output.data
                kx_min = float(np.min(np.abs(data.kx)))

                # removes growth_rate_tolerance from nonlinear codes with no time TGLF
                if (
                    "mode" not in pyro.gk_output.dims
                    and "growth_rate_tolerance" in spec["extras"]
                ):
                    run_buffers["growth_rate_tolerance"] = None

                if (
                    "mode" not in pyro.gk_output.dims
                    and "growth_rate_tolerance" in spec["extras"]
                ):
                    run_buffers["growth_rate_tolerance"] = (
                        pyro.gk_output.get_growth_rate_tolerance(
                            tolerance_time_range
                        ).sel(kx=kx_min)
                    )

                if tearing and "apar" in pyro.gk_output:
                    # Needs apar's sign along theta, which "average" discards.
                    # Geometry comes from the run's own deck: a reloaded scan's
                    # pyros are copies of the base without the scan parameters
                    run = Pyro(gk_file=pyro.gk_file)
                    run.gk_output = pyro.gk_output
                    field_line = FieldLine(run)
                    for name, value in (
                        (
                            "tearing_parameter",
                            field_line.compute_linear_tearing_parameter(),
                        ),
                        ("apar_even_fraction", field_line.compute_linear_parity()),
                        *(
                            [
                                (
                                    "phi_even_fraction",
                                    field_line.compute_linear_parity("phi"),
                                )
                            ]
                            if "phi" in pyro.gk_output
                            else []
                        ),
                    ):
                        run_buffers[name] = select_kx_ky_time(
                            value.copy(data=value.values * ureg.dimensionless),
                            kx_min=kx_min,
                            time_mode=time_mode,
                        )

                for name in spec["scalars"]:
                    if name in pyro.gk_output:
                        run_buffers[name] = select_kx_ky_time(
                            pyro.gk_output[name],
                            kx_min=kx_min,
                            time_mode=time_mode,
                            tolerance_time_range=tolerance_time_range,
                        )

                for name in spec["fluxes"]:
                    if name in pyro.gk_output:
                        run_buffers[name] = select_kx_ky_time(
                            pyro.gk_output[name],
                            kx_min=kx_min,
                            sum_ky=sum_ky,
                            time_mode=time_mode,
                            tolerance_time_range=tolerance_time_range,
                        )

                # Fields and eigenfunctions are reduced identically: kx and ky
                # are kept unless summed with sum_kx / sum_ky
                for name in spec["fields"]:
                    if name in pyro.gk_output:
                        field = pyro.gk_output[name]
                        # A time mean of a complex field is meaningless, so
                        # "average" averages |field|**2 instead
                        if time_mode == "average" and "time" in field.dims:
                            field = abs(field) ** 2
                            squared.add(name)
                        run_buffers[name] = select_kx_ky_time(
                            field,
                            sum_ky=sum_ky,
                            sum_kx=sum_kx,
                            time_mode=time_mode,
                            tolerance_time_range=tolerance_time_range,
                        )
                for name, value in run_buffers.items():
                    buffers[name].append(value)

            except (
                FileNotFoundError,
                OSError,
                IndexError,
                RuntimeError,
                KeyError,
                ValueError,
            ) as e:
                warnings.warn(
                    f"Unable to load gk_output for {pyro.gk_file} "
                    f"[{type(e).__name__}: {e}]"
                )
                for name in buffers:
                    ref = next((x for x in buffers[name] if x is not None), None)
                    if ref is None:
                        buffers[name].append(None)
                    else:
                        buffers[name].append(ref * np.nan)

            finally:
                if hasattr(pyro, "gk_output"):
                    pyro.gk_output = None

        for name, arrays in buffers.items():
            ref = next((x for x in arrays if x is not None), None)
            if ref is None:
                continue
            buffers[name] = [ref * np.nan if x is None else x for x in arrays]

        if all(all(x is None for x in arrays) for arrays in buffers.values()):
            raise FileNotFoundError("Unable to load any gk_output files in this scan")

        ds = xr.Dataset(coords=layout.coords)
        for name, arrays in buffers.items():
            ds = add_quantity(
                ds,
                name,
                arrays,
                layout.shape,
                layout.coords,
                scan_dims=layout.dims,
                squeeze_dims=layout.squeeze_dims,
            )

        for coord, units in layout.coord_units.items():
            ds[coord] = ds[coord].assign_attrs(units=units)

        # |field|**2 is not the field, so it is stored under its own name
        ds = ds.rename({name: f"{name}_squared" for name in squared if name in ds})

        self.gk_output = PyroScanGKOutput(ds, norms=self.base_pyro.norms)

        self.gk_output.to(getattr(self.base_pyro.norms, output_convention))

    def output_layout(self) -> ScanLayout:
        """
        Map the runs of this scan onto the dimensions of its output Dataset.

        A ``PyroScan`` is a grid: one dimension per varied parameter, whose
        length is the number of values that parameter takes.
        """
        coords = {}
        coord_units = {}
        for name, values in self.parameter_dict.items():
            vals, unit = magnitude_and_units(values)
            # Carry the units on the coordinate itself: a coordinate rebuilt
            # from bare values when each quantity is stacked would otherwise
            # lose them again.
            coords[name] = ((name,), vals, {"units": unit})
            coord_units[name] = unit

        dims = tuple(self.parameter_dict.keys())
        shape = tuple(len(values) for values in self.parameter_dict.values())

        return ScanLayout(
            coords=coords,
            coord_units=coord_units,
            shape=shape,
            dims=dims,
            squeeze_dims=dims,
        )

    @property
    def gk_code(self):
        # NOTE: In previous versions, this would return a GKCode class. Now it only
        #      returns a string.
        #      The setter has been replaced by the function 'convert_gk_code'
        return self.base_pyro.gk_code

    def convert_gk_code(self, gk_code: str, template_file=None) -> None:
        """
        Converts all gyrokinetics codes to the code type 'gk_code'. This can be any
        viable GKInput type (GS2, CGYRO, GENE,...)

        ``file_name`` becomes the new code's default, and every run keeps its run
        directory, so a following ``write`` writes one input file per run for the
        new code. Pass ``base_directory`` to ``write`` to keep the original runs.

        Parameters
        ----------
        gk_code: str
            The gyrokinetics code to convert to.
        template_file: PathLike, default None
            Template used to populate each new input file. If unset, the default
            template for ``gk_code`` is used.
        """
        self.base_pyro.convert_gk_code(gk_code, template_file=template_file)
        for pyro in self.pyro_dict.values():
            pyro.convert_gk_code(gk_code, template_file=template_file)

        self.file_name = GKInput._factory[gk_code].default_file_name
        for name, pyro in self.pyro_dict.items():
            pyro.gk_file = self.base_directory / name / self.file_name

    @property
    def base_directory(self):
        return self._base_directory

    @base_directory.setter
    def base_directory(self, value):
        """
        Sets the base_directory

        """

        self._base_directory = pathlib.Path(value).absolute()
        self.pyroscan_json["base_directory"] = self._base_directory

        # Set base_directory in copies of pyro
        for key, pyro in self.pyro_dict.items():
            pyro.gk_file = self.base_directory / key / pyro.gk_file.name

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        """
        Sets the file_name

        """

        self.pyroscan_json["file_name"] = value
        self._file_name = value

    def outer_product(self):
        """
        Creates generator of outer product for all parameter permutations
        """
        return (
            dict(zip(self.parameter_dict, x))
            for x in product(*self.parameter_dict.values())
        )


def get_from_dict(data_dict, map_list):
    """
    Gets item in dict given location as a list of string
    """
    return reduce(get_attr_or_item, map_list, data_dict)


def get_attr_or_item(obj, value):
    if hasattr(obj, value):
        return getattr(obj, value)
    elif value in obj.keys():
        return obj[value]
    else:
        raise ValueError(f"{obj} has not got {value} as a key or attribute")


def set_in_dict(data_dict, map_list, value):
    """
    Sets item in dict given location as a list of string
    """
    get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = copy.deepcopy(value)


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


class NumpyEncoder(json.JSONEncoder):
    """Numpy/pint-aware JSON encoder. Quantities are stored as
    ``[magnitude, unit_str]`` with unit names left fully qualified
    (including any instance-specific normalisation suffix)."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pathlib.Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, pint.Quantity):
            return [obj.m, str(obj.units)]
        return json.JSONEncoder.default(self, obj)


class PyroScanGKOutput(DatasetWrapper):
    def __init__(self, dataset: xr.Dataset, norms=None):
        data_vars = dataset.data_vars
        coords = dataset.coords
        attrs = dataset.attrs

        # Hand over to underlying dataset
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

        # Used by ``to_netcdf`` to strip run-specific unit suffixes before
        # serialisation. Optional — may be set later by ``PyroScan``.
        self._norms = norms

    def to(self, norms: ConventionNormalisation, *contexts):
        """

        Parameters
        ----------
        norms : ConventionNormalisation
            Normalisation convention to convert to

        Returns
        -------
        GKOutput with units from norms
        """
        for data_var in self.data_vars:
            self[data_var].data = self[data_var].data.to(norms, *contexts)

        # Coordinates with units not supported in xarray need to manually change
        new_coords = {}
        for coord in self.coords:
            quantity = coord_quantity(self[coord])
            if quantity is None:
                continue
            new_coord = quantity.to(norms, *contexts)
            # Use the coordinate's own dims: a non-dimension coordinate (a
            # varied parameter along ``sample``) is not its own dimension.
            new_coords[coord] = (
                self[coord].dims,
                new_coord.m,
                {"units": new_coord.units},
            )

        self.data = self.data.assign_coords(coords=new_coords)

    def convert_physical_units(self, norms: ConventionNormalisation):
        """
        Replace physical units on data_vars and coords with the generic
        simulation units of ``norms``. Needed before writing to netCDF so
        saved unit strings (e.g. ``nref_electron``) are not tied to a
        particular pyro's run-specific normalisation name.
        """
        for data_var in self.data_vars:
            data = self[data_var].data
            if hasattr(data, "convert_physical_units"):
                self[data_var].data = data.convert_physical_units(norms)

        new_coords = {}
        for coord in self.coords:
            quantity = coord_quantity(self[coord])
            if quantity is None:
                continue
            if hasattr(quantity, "convert_physical_units"):
                new_coord = quantity.convert_physical_units(norms)
                new_coords[coord] = (
                    self[coord].dims,
                    new_coord.m,
                    {"units": new_coord.units},
                )

        if new_coords:
            self.data = self.data.assign_coords(coords=new_coords)

    def to_netcdf(self, *args, **kwargs) -> None:
        """
        Serialise to netCDF. Physical units are first converted to generic
        simulation units so the stored unit strings remain portable across
        pyro instances (which otherwise mint run-specific unit names).
        """
        if self._norms is not None:
            convention = getattr(self._norms, "pyrokinetics", self._norms)
            self.convert_physical_units(convention)
        super().to_netcdf(*args, **kwargs)

    def unwrap(self):
        """Return the underlying xarray.Dataset."""
        return self._dataset
