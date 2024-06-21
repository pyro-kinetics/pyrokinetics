from __future__ import annotations

import copy
import json
import os
import pathlib
from contextlib import contextmanager
from functools import reduce
from itertools import product

import numpy as np
import pint

from .gk_code import GKInput
from .pyro import Pyro


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
        "p_prime_type",
        "parameter_map",
    ]

    def __init__(
        self,
        pyro,
        parameter_dict=None,
        p_prime_type=0,
        value_fmt=".2f",
        value_separator="_",
        parameter_separator="/",
        file_name=None,
        base_directory=".",
        load_default_parameter_keys=True,
        pyroscan_json=None,
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

        if file_name is not None:
            self.file_name = file_name
        else:
            self.file_name = GKInput._factory[pyro.gk_code].default_file_name

        if load_default_parameter_keys:
            self.load_default_parameter_keys()

        self.run_directories = None

        if isinstance(pyro, Pyro):
            self.base_pyro = pyro
        else:
            raise ValueError("PyroScan takes in a pyro object")

        if parameter_dict is None:
            self.parameter_dict = {}
        else:
            self.parameter_dict = parameter_dict

        self.p_prime_type = p_prime_type

        # Load in pyroscan json if there
        if pyroscan_json is not None:
            with open(pyroscan_json) as f:
                self.pyroscan_json = json.load(f)

            for key, value in self.pyroscan_json.items():
                setattr(self, key, value)
        else:
            self.pyroscan_json = {attr: getattr(self, attr) for attr in self.JSON_ATTRS}

        # Get len of values for each parameter
        self.value_size = [len(value) for value in self.parameter_dict.values()]

        self.pyro_dict = dict(
            self.create_single_run(run) for run in self.outer_product()
        )
        self.run_directories = [pyro.run_directory for pyro in self.pyro_dict.values()]

    def format_single_run_name(self, parameters):
        """
        Concatenate parameter names/values with separator
        """
        return self.parameter_separator.join(
            (
                f"{param}{self.value_separator}{getattr(value, 'magnitude', value):{self.value_fmt}}"
                for param, value in parameters.items()
            )
        )

    def create_single_run(self, parameters: dict):
        """
        Create a new Pyro instance from the PyroScan base with new run parameters
        """
        name = self.format_single_run_name(parameters)
        new_run = copy.deepcopy(self.base_pyro)
        new_run.gk_file = self.base_directory / name / self.file_name
        new_run.run_parameters = copy.deepcopy(parameters)
        return name, new_run

    def write(self, file_name=None, base_directory=None, template_file=None):
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
        with open(json_file, "w+") as f:
            json.dump(self.pyroscan_json, f, cls=NumpyEncoder)

        # Iterate through all runs and write output
        for parameter, run_dir, pyro in zip(
            self.outer_product(), self.run_directories, self.pyro_dict.values()
        ):
            # Param value for each run written accordingly
            for param, value in parameter.items():
                # Get attribute name and keys where param is stored in Pyro
                (attr_name, keys_to_param) = self.parameter_map[param]

                # Get attribute in Pyro storing the parameter
                pyro_attr = getattr(pyro, attr_name)

                if hasattr(value, "units"):
                    dimensional_value = value
                else:
                    units = getattr(
                        get_from_dict(pyro_attr, keys_to_param[:-1])[keys_to_param[-1]],
                        "units",
                        1,
                    )
                    dimensional_value = value * units

                # Set the value given the Pyro attribute and location of parameter
                set_in_dict(pyro_attr, keys_to_param, dimensional_value)

                if param in self.parameter_func.keys():
                    func, kwargs = self.parameter_func[param]
                    func(pyro, **kwargs)

            # Write input file
            pyro.write_gk_file(
                file_name=run_dir / self.file_name, template_file=template_file
            )

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

        {'electron_temp_gradient': ["local_species", ['electron','inverse_lt']] }
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

        # Elongation
        parameter_key = "kappa"
        parameter_attr = "local_geometry"
        parameter_location = ["kappa"]
        self.add_parameter_key(parameter_key, parameter_attr, parameter_location)

    def load_gk_output(self):
        """
        Loads GKOutput as a xarray Sataset

        Returns
        -------
        self.gk_output : xarray DataSet of data
        """
        import xarray as xr

        # xarray DataSet to store data
        ds = xr.Dataset(self.parameter_dict)

        # TODO Need to add property to GKCode checking if it is an eigensolver
        # or initial value run and then set nmodes accordingly
        if self.base_pyro.gk_code == "TGLF":
            nmode = self.base_pyro.gk_input.data.get("nmodes", 2)
            nmode_coords = {"nmode": list(range(1, 1 + nmode))}
            ds = ds.assign_coords(nmode_coords)
        else:
            nmode = np.nan

        if not self.base_pyro.numerics.nonlinear:
            growth_rate = []
            mode_frequency = []
            eigenfunctions = []
            growth_rate_tolerance = []
            particle = []
            heat = []

            # Load gk_output in copies of pyro
            for pyro in self.pyro_dict.values():
                try:
                    pyro.load_gk_output()

                    if "time" in pyro.gk_output.dims:
                        growth_rate.append(pyro.gk_output["growth_rate"].isel(time=-1))
                        mode_frequency.append(
                            pyro.gk_output["mode_frequency"].isel(time=-1)
                        )
                        eigenfunctions.append(
                            pyro.gk_output["eigenfunctions"]
                            .isel(time=-1, kx=0, ky=0, missing_dims="ignore")
                            .drop_vars(["time", "kx", "ky"], errors="ignore")
                        )
                        if "ky" in pyro.gk_output["particle"].coords:
                            particle.append(
                                pyro.gk_output["particle"]
                                .isel(time=-1, missing_dims="ignore")
                                .sum(dim="ky")
                                .drop_vars(["time"])
                            )
                            heat.append(
                                pyro.gk_output["heat"]
                                .isel(time=-1, missing_dims="ignore")
                                .sum(dim="ky")
                                .drop_vars(["time"])
                            )
                        else:
                            particle.append(
                                pyro.gk_output["particle"]
                                .isel(time=-1, missing_dims="ignore")
                                .drop_vars(["time"])
                            )
                            heat.append(
                                pyro.gk_output["heat"]
                                .isel(time=-1, missing_dims="ignore")
                                .drop_vars(["time"])
                            )

                        tolerance = pyro.gk_output[
                            "growth_rate_tolerance"
                        ].data.flatten()[0]

                        growth_rate_tolerance.append(tolerance)

                    elif "mode" in pyro.gk_output.dims:
                        growth_rate.append(pyro.gk_output["growth_rate"])
                        mode_frequency.append(pyro.gk_output["mode_frequency"])
                        eigenfunctions.append(pyro.gk_output["eigenfunctions"])

                    # Remove GKOutput to conserve memory
                    pyro.gk_output = None

                except (FileNotFoundError, OSError, IndexError, RuntimeError, KeyError):
                    growth_rate.append(growth_rate[0] * np.nan)
                    mode_frequency.append(mode_frequency[0] * np.nan)
                    growth_rate_tolerance.append(growth_rate_tolerance[0] * np.nan)
                    particle.append(particle[0] * np.nan)
                    heat.append(heat[0] * np.nan)
                    eigenfunctions.append(eigenfunctions[0] * np.nan)

            # Save eigenvalues

            output_shape = copy.deepcopy(self.value_size)
            coords = list(self.parameter_dict.keys())

            if "nmode" in ds.dims:
                output_shape.append(nmode)
                coords.append("mode")

            growth_rate = np.reshape(growth_rate, output_shape)
            mode_frequency = np.reshape(mode_frequency, output_shape)
            ds["growth_rate"] = (coords, growth_rate)
            ds["mode_frequency"] = (coords, mode_frequency)

            if growth_rate_tolerance:
                growth_rate_tolerance = np.reshape(growth_rate_tolerance, output_shape)
                ds["growth_rate_tolerance"] = (
                    coords,
                    growth_rate_tolerance,
                )

            # Add eigenfunctions
            eig_coords = eigenfunctions[-1].coords
            ds = ds.assign_coords(coords=eig_coords)

            # Reshape eigenfunctions and generate new coordinates
            eigenfunction_shape = self.value_size + list(np.shape(eigenfunctions[-1]))
            eigenfunctions = np.reshape(eigenfunctions, eigenfunction_shape)
            eigenfunctions_coords = tuple(self.parameter_dict.keys()) + eig_coords.dims

            ds["eigenfunctions"] = (eigenfunctions_coords, eigenfunctions)

            # Add fluxes
            if particle:
                particle_coords = particle[-1].coords
                ds = ds.assign_coords(coords=particle_coords)

                # Reshape particle and generate new coordinates
                particle_shape = output_shape + list(np.shape(particle[-1]))
                particle = np.reshape(particle, particle_shape)
                particle_coords = tuple(coords) + particle_coords.dims

                ds["particle"] = (particle_coords, particle)

                heat_coords = heat[-1].coords
                ds = ds.assign_coords(coords=heat_coords)

                # Reshape heat and generate new coordinates
                heat_shape = output_shape + list(np.shape(heat[-1]))
                heat = np.reshape(heat, heat_shape)
                heat_coords = tuple(coords) + heat_coords.dims

                ds["heat"] = (heat_coords, heat)

        self.gk_output = ds

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
        """
        self.base_pyro.convert_gk_code(gk_code)
        for pyro in self.pyro_dict.values():
            pyro.convert_gk_code(gk_code)

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
    r"""
    Numpy encoder for json.dump
    """

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
            return obj.m
        return json.JSONEncoder.default(self, obj)
