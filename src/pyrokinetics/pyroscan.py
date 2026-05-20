from __future__ import annotations

import copy
import json
import os
import pathlib
import warnings
from contextlib import contextmanager
from functools import reduce
from itertools import product

import numpy as np
import pint
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

    raise ValueError(f"Unknown time_mode={time_mode}")


# ---- xarray selection ----
def select_kx_ky_time(
    da,
    *,
    kx_min,
    sum_ky=False,
    time_mode,
    tolerance_time_range=None,
):
    if "kx" in da.dims:
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
def add_quantity(ds, name, arrays, base_shape, scan_coords):
    if not any(isinstance(a, xr.DataArray) for a in arrays):
        return ds

    last = arrays[-1]
    for dim in scan_coords:
        if dim in last.dims:
            last = last.squeeze(dim, drop=True)
            arrays = [
                a.squeeze(dim, drop=True) if isinstance(a, xr.DataArray) else a
                for a in arrays
            ]

    shape = base_shape + last.shape

    raw = []
    units = None

    for a in arrays:
        data = a.data
        if hasattr(data, "magnitude"):
            raw.append(data.magnitude)
            units = data.units
        else:
            raw.append(np.asarray(data))

    stacked = np.stack(raw).reshape(shape)

    if units is not None:
        stacked = stacked * units

    dims = tuple(scan_coords.keys()) + last.dims

    arr = xr.DataArray(
        stacked,
        dims=dims,
        coords={
            **scan_coords,
            **last.coords,
        },
    )

    arr = arr.reset_coords(drop=True)
    ds[name] = arr

    return ds


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

        # Per-run state.
        #
        # ``_run_records`` is the source of truth for the scan: an ordered list
        # of ``(name, parameters)`` tuples, one per scan point. Per-run Pyros
        # are *not* built at construction time — they are deep-copies of
        # ``base_pyro`` and a 1000-point scan would do 1000 deepcopies up front,
        # most of which are never touched (e.g. when the user only wants
        # ``load_gk_output``). Instead we materialise the dict lazily on first
        # access, and the read path (``load_gk_output``) bypasses it entirely.
        #
        # ``_pyro_dict_cache`` is ``None`` until something forces materialisation
        # (e.g. ``write``, ``convert_gk_code``, ``update_self_parameters`` or a
        # direct read of the public ``pyro_dict`` property).
        self._run_records: list[tuple[str, dict]] = []
        self._pyro_dict_cache: dict | None = None

        # ``pyroscan_json`` must exist before the ``base_directory`` setter runs
        # (it writes into it). Same goes for ``parameter_func``.
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

        # Build the metadata for each scan point. We deepcopy ``parameters``
        # here (rather than holding a reference to the dict yielded by
        # ``outer_product``) so callers can mutate the per-run params dict
        # without surprising side effects. This is cheap — the dicts are tiny.
        self._run_records = [
            (self.format_single_run_name(params), copy.deepcopy(params))
            for params in self.outer_product()
        ]

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
                (
                    f"{param}{self.value_separator}{getattr(value, 'magnitude', value):{self.value_fmt}}"
                    for param, value in parameters.items()
                )
            )

    def create_single_run(self, parameters: dict):
        """
        Create a new Pyro instance from the PyroScan base with new run parameters.

        This deep-copies ``base_pyro`` and is therefore O(size of base_pyro). It is
        used internally when ``pyro_dict`` is materialised; most callers should
        not need to call it directly.
        """
        name = self.format_single_run_name(parameters)
        new_run = copy.deepcopy(self.base_pyro)

        new_run.gk_file = self.base_directory / name / self.file_name
        new_run.run_parameters = copy.deepcopy(parameters)
        return name, new_run

    def _gk_file_for(self, name: str) -> pathlib.Path:
        """Path of the per-run gk_file for scan point ``name``, derived live from
        ``base_directory`` and ``file_name`` so that updating either is reflected
        without touching any per-run Pyros."""
        return self.base_directory / name / self.file_name

    def _materialise_pyro_dict(self) -> dict:
        """
        Build the per-run ``Pyro`` objects on demand.

        Each entry is a deep-copy of ``base_pyro`` with its ``gk_file`` and
        ``run_parameters`` fields set. Parameter values are *not* applied here
        (that happens in ``update_self_parameters`` from ``write``); callers
        relying on parameters being applied should call ``write`` or
        ``update_self_parameters`` themselves, exactly as before.

        This function is the only place in the class that pays the deep-copy
        cost. Avoid calling it on the read path.
        """
        return dict(
            self.create_single_run(params) for _, params in self._run_records
        )

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

    def update_self_parameters(
        self,
    ):
        """
        Updates all pyro object parameters based on pyro_dict values
        """
        for parameter, run_dir, pyro in zip(
            self.outer_product(), self.run_directories, self.pyro_dict.values()
        ):
            # Param value for each run written accordingly
            for param, value in parameter.items():
                # Get attribute name and keys where param is stored in Pyro
                attr_name, keys_to_param = self.parameter_map[param]

                # Get attribute in Pyro storing the parameter
                pyro_attr = getattr(pyro, attr_name)

                # Set the value given the Pyro attribute and location of parameter
                set_in_dict(pyro_attr, keys_to_param, value)

                if param in self.parameter_func.keys():
                    func, kwargs = self.parameter_func[param]
                    func(pyro, **kwargs)

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
        parameter_func: function that take in a pyro object applies modification
        parameter_kwargs: Dictionary of kwargs to apply to function
        """

        self.parameter_func[parameter_key] = (parameter_func, parameter_kwargs)

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
        netcdf_file=None,
        load_fields=True,
        load_fluxes=True,
        load_moments=False,
        sum_ky=True,
        drop_nan=False,
        **kwargs,
    ):
        """
        Loads PyroScanGKOutput into self.gk_output

        Parameters
        ----------
        output_convention: str default 'pyrokinetics'
            ConventionNormalisation to convert output to
        tolerance_time_range: float default 0.8
            Time window over which to calculate growth rate tolerance
        netcdf_file: PathLike default None
            If supplied then load PyroScanGKOutput from existing netCDF
        load_fields (bool, default True) – Flag to load fields or not
        load_fluxes (bool, default True) – Flag to load fluxes or not
        load_moments (bool, default False) – Flag to load moments or not
        drop_nan (bool, default False) – If NaNs are found in the output then that data is dropped. Off by default
        **kwargs – Arguments to pass to the GKOutputReader.
        Returns
        -------
        None
        """
        # Load from netCDF is supplied
        if netcdf_file is not None:
            convention = getattr(self.base_pyro.norms, output_convention)
            gk_output = PyroScanGKOutput.from_netcdf(netcdf_file)
            gk_output.to(convention, convention.context)
            self.gk_output = gk_output
            return

        parameter_dict = {}
        coords = {}
        attrs = {}
        coord_units = {}
        for name, values in self.parameter_dict.items():
            # Unitless scan parameters are stored as plain arrays/lists.
            if isinstance(values, Quantity):
                vals = np.asarray(values.magnitude)
                unit = values.units
            else:
                vals = np.asarray(values)
                unit = ureg.dimensionless

            parameter_dict[name] = ((name,), vals)
            coords[name] = vals
            attrs[name + "_units"] = str(unit)
            coord_units[name] = unit

        output_shape = tuple(len(v) for v in self.parameter_dict.values())

        load_specs = {
            "linear": {
                "scalars": ["growth_rate", "mode_frequency", "eigenfunctions"],
                "extras": ["growth_rate_tolerance"],
                "fluxes": [],
                "fields": [],
            },
            "nonlinear": {
                "scalars": [],
                "extras": [],
                "fluxes": [],
                "fields": [],
            },
        }

        time_policy = {
            "linear": {
                "scalars": "last",
                "fluxes": "last",
                "fields": "last",
            },
            "nonlinear": {
                "scalars": "last",
                "fluxes": "average",
                "fields": "average",
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
        time_policy = time_policy[regime]

        buffers = {
            name: []
            for name in (
                spec["scalars"] + spec["extras"] + spec["fluxes"] + spec["fields"]
            )
        }

        # Reuse ``base_pyro`` for every read instead of looping over per-run
        # Pyros from ``pyro_dict``. This is safe because each per-run Pyro is
        # only ever a deep-copy of ``base_pyro`` (any per-run parameter
        # overrides applied by ``write``/``update_self_parameters`` change the
        # *input* file content, not the output file format) and the
        # gk_output readers consume only ``(path, norm)``, never the Pyro
        # itself. By skipping ``pyro_dict`` we avoid both the up-front
        # deep-copy cost in the constructor and the per-run pint-context
        # overhead that came with each per-run ``SimulationNormalisation``.
        pyro = self.base_pyro
        # Save state so we leave ``base_pyro`` looking exactly as we found it.
        # We mutate ``gk_file`` per scan point and ``pyro.load_gk_output`` sets
        # ``gk_output_file`` as a side effect.
        saved_gk_file = pyro.gk_file
        saved_gk_output_file = pyro.gk_output_file
        for run_name, _ in self._run_records:
            run_buffers = {name: None for name in buffers}
            gk_file = self._gk_file_for(run_name)
            try:
                pyro.gk_file = gk_file
                pyro.load_gk_output(
                    output_convention=output_convention,
                    load_fields=load_fields,
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

                for name in spec["scalars"]:
                    if name in pyro.gk_output:
                        run_buffers[name] = select_kx_ky_time(
                            pyro.gk_output[name],
                            kx_min=kx_min,
                            time_mode=time_policy["scalars"],
                        )

                for name in spec["fluxes"]:
                    if name in pyro.gk_output:
                        run_buffers[name] = select_kx_ky_time(
                            pyro.gk_output[name],
                            kx_min=kx_min,
                            sum_ky=sum_ky,
                            time_mode=time_policy["fluxes"],
                            tolerance_time_range=tolerance_time_range,
                        )

                data = data.isel(ky=[0]).squeeze()
                pyro.gk_output.data = data

                for name in spec["fields"]:
                    if name in pyro.gk_output:
                        run_buffers[name] = select_kx_ky_time(
                            pyro.gk_output[name],
                            kx_min=kx_min,
                            time_mode=time_policy["fields"],
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
                    f"Unable to load gk_output for {gk_file} "
                    f"[{type(e).__name__}: {e}]"
                )
                for name in buffers:
                    ref = next((x for x in buffers[name] if x is not None), None)
                    if ref is None:
                        buffers[name].append(None)
                    else:
                        buffers[name].append(ref * np.nan)

            finally:
                # Free the loaded dataset so memory doesn't accumulate across
                # the scan. ``base_pyro`` is reused on the next iteration, so
                # we must clear ``gk_output`` between runs.
                if hasattr(pyro, "gk_output"):
                    pyro.gk_output = None

        # Restore ``base_pyro`` to its pre-load state. We mutated ``gk_file``
        # and ``pyro.load_gk_output`` set ``gk_output_file``; without this,
        # the user would find their ``base_pyro`` pointing at the last run.
        # Both setters reject ``None``, so guard the restores accordingly.
        if saved_gk_file is not None:
            pyro.gk_file = saved_gk_file
        if saved_gk_output_file is not None:
            pyro.gk_output_file = saved_gk_output_file

        for name, arrays in buffers.items():
            ref = next((x for x in arrays if x is not None), None)
            if ref is None:
                continue
            buffers[name] = [ref * np.nan if x is None else x for x in arrays]

        if all(all(x is None for x in arrays) for arrays in buffers.values()):
            raise FileNotFoundError("Unable to load any gk_output files in this scan")

        ds = xr.Dataset(parameter_dict)
        for name, arrays in buffers.items():
            ds = add_quantity(ds, name, arrays, output_shape, coords)

        for coord, units in coord_units.items():
            ds[coord] = ds[coord].assign_attrs(units=units)

        self.gk_output = PyroScanGKOutput(ds)

        self.gk_output.to(getattr(self.base_pyro.norms, output_convention))

    @property
    def gk_code(self):
        # NOTE: In previous versions, this would return a GKCode class. Now it only
        #      returns a string.
        #      The setter has been replaced by the function 'convert_gk_code'
        return self.base_pyro.gk_code

    def convert_gk_code(self, gk_code: str) -> None:
        """
        Converts all gyrokinetics codes to the code type 'gk_code'. This can be any
        viable GKInput type (GS2, CGYRO, GENE,...)

        Note: this materialises ``pyro_dict`` because each per-run Pyro must be
        converted independently.
        """
        self.base_pyro.convert_gk_code(gk_code)
        for pyro in self.pyro_dict.values():
            pyro.convert_gk_code(gk_code)

    @property
    def pyro_dict(self) -> dict:
        """
        Mapping ``{run_name: Pyro}`` for each scan point.

        The dict is built lazily on first access — see ``_materialise_pyro_dict``
        for why. If you only need to read gk_outputs, use ``load_gk_output`` and
        avoid touching this property: it will skip the deep-copies entirely.
        """
        if self._pyro_dict_cache is None:
            self._pyro_dict_cache = self._materialise_pyro_dict()
        return self._pyro_dict_cache

    @pyro_dict.setter
    def pyro_dict(self, value: dict) -> None:
        """Allow callers to inject a pre-built dict (used by tests and
        downstream code). Setting it bypasses lazy materialisation."""
        self._pyro_dict_cache = value

    @property
    def run_directories(self) -> list:
        """
        Directories that each scan point will be written into.

        Derived live from ``base_directory`` and the run names: this means
        updating ``base_directory`` is reflected immediately without having to
        walk an N-element ``pyro_dict``.
        """
        return [self.base_directory / name for name, _ in self._run_records]

    @run_directories.setter
    def run_directories(self, value: list) -> None:
        # Kept for backward compatibility. The property is normally derived,
        # but external code (and a few historic call sites) may try to set it.
        # We accept the assignment without error; subsequent reads will reflect
        # whatever ``base_directory`` and ``_run_records`` say, which matches
        # the value that was just written in all current call sites.
        pass

    @property
    def base_directory(self):
        return self._base_directory

    @base_directory.setter
    def base_directory(self, value):
        """
        Sets the base_directory.

        Updates the per-run ``gk_file`` paths *only* if ``pyro_dict`` has
        already been materialised — otherwise we leave ``_pyro_dict_cache`` at
        ``None`` so the load path stays cheap. ``run_directories`` is a
        derived property, so it always reflects the new base directory.
        """

        self._base_directory = pathlib.Path(value).absolute()
        self.pyroscan_json["base_directory"] = self._base_directory

        if self._pyro_dict_cache is not None:
            for key, pyro in self._pyro_dict_cache.items():
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
    def __init__(self, dataset: xr.Dataset):
        data_vars = dataset.data_vars
        coords = dataset.coords
        attrs = dataset.attrs

        # Hand over to underlying dataset
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

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
            if hasattr(self[coord], "units"):
                if self[coord].units is None:
                    continue
                new_coord = (self[coord].data * self[coord].units).to(norms, *contexts)
                new_coords[coord] = (
                    coord,
                    new_coord.m,
                    {"units": new_coord.units},
                )

        self.data = self.data.assign_coords(coords=new_coords)

    def unwrap(self):
        """Return the underlying xarray.Dataset."""
        return self._dataset
