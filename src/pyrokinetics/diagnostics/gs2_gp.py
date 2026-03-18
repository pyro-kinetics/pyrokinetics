import itertools
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.special import erf

from pyrokinetics import Pyro, PyroScan
from pyrokinetics.pyroscan import PyroScanGKOutput
from pyrokinetics.units import ureg

pyro = Pyro(gk_code="GS2")  # check units with bahvin


default_unit_dict = {
    "growth_rate": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "mode_frequency": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "kperp2_phi": ureg.dimensionless,  # k perep squared over ky squared
    "kperp2_apa": ureg.dimensionless,
    "kperp2_bpar": ureg.dimensionless,  # Kperp2 is normalised to ky^2 - The others need to be multiplied by kperpe2_phi, thereore need to make sure that model is always loaded
    "totIonFlux": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "totElecFlux": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "totPartFlux": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "totPartFlux": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "apa_phi": ureg.dimensionless,
    "bpar_phi": ureg.dimensionless,
    "Lambda": pyro.norms.pyrokinetics.rhoref,  # fix this
    "theta0fit_sigmas": ureg.dimensionless,
}
### Need to get the correct units for this


default_output_conversion_dict = {
    "_log": lambda x: np.power(10, x),
    "": lambda x: x,
}


class gs2_gp:
    def __init__(
        self,
        pyro: Pyro,
        models_path,
        models,
        model_kernel=[""],  # This is the kernel - rename
        units_dict=default_unit_dict,
        output_conversion_dict=default_output_conversion_dict,
    ):
        """
        If `pyro` is a Pyro object → evaluate single case.
        If `pyro` is a PyroScan → evaluate all cases and combine results.
        """
        self.model_kernel = model_kernel
        self.models_path = models_path
        self.model_names = models
        self.units_dict = units_dict
        self.output_conversion_dict = output_conversion_dict

        # Load models once
        self.load_models(models_path, models, model_kernel)

        # Determine what was passed
        if isinstance(pyro, Pyro):
            self._evaluate_single(pyro)
            self.convert_to_GKoutput()
        elif isinstance(pyro, PyroScan):
            self._evaluate_scan_whole(pyro)
            self.convert_to_GKoutput()
        else:
            raise TypeError(f"Expected Pyro or PyroScan, got {type(pyro)}")

    def strip_suffix(self, name):
        """Return base name without suffix."""
        for suffix in self.output_conversion_dict:
            if suffix and name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def get_conversion_function(self, name):
        """Return conversion function based on model name suffix."""
        for suffix, func in self.output_conversion_dict.items():
            if suffix and name.endswith(suffix):
                return func
        return self.output_conversion_dict.get("", lambda x: x)

    # ------------------------------
    # Single Pyro evaluation
    # ------------------------------
    def _evaluate_single(self, pyro: Pyro):
        """Evaluate models for a single Pyro object."""
        self.pyro = pyro
        self.inputs = torch.tensor([self.model_input()], dtype=torch.float64)
        self.evaluate_all_models()

    def format_single_run_name(self, parameters):
        """
        Concatenate parameter names/values with separator
        """
        return self.parameter_separator.join(
            (
                f"{param}{self.pyro.value_separator}{getattr(value, 'magnitude', value):{self.value_fmt}}"
                for param, value in parameters.items()
            )
        )

    def convert_to_GKoutput(self):
        pyroutput = PyroScanGKOutput(self.models)
        my_convention = self.pyro.norms.pyrokinetics
        pyroutput.to(my_convention)
        # convert xarray from normalisation of pysocan to nomarlisation of
        self.gk_output = pyroutput

    def _evaluate_scan_whole(self, pyroscan: PyroScan):
        if not "ky" in pyroscan.parameter_dict.keys():
            new_dict = pyroscan.parameter_dict
            new_dict["ky"] = (
                np.linspace(0.04, 1, 30) / pyroscan.base_pyro.norms.pyrokinetics.rhoref
            )
            new_pyroscan = PyroScan(
                pyroscan.base_pyro,
                new_dict,
                value_fmt=pyroscan.value_fmt,
                value_separator=pyroscan.value_separator,
                parameter_separator=pyroscan.parameter_separator,
                file_name=pyroscan.file_name,
            )
            new_keys = set(new_pyroscan.parameter_map.keys())
            old_keys = set(pyroscan.parameter_map.keys())
            extra_keys = list(old_keys - new_keys)
            for key in extra_keys:
                # You need to provide the correct parameter_attr and parameter_location for each key
                new_pyroscan.add_parameter_key(
                    parameter_key=key,
                    parameter_attr=pyroscan.parameter_map[key][
                        0
                    ],  # Replace with actual attribute
                    parameter_location=pyroscan.parameter_map[key][
                        1
                    ],  # Replace with actual location
                )

            pyroscan = new_pyroscan
        keys = list(pyroscan.parameter_dict.keys())
        input_dict = {}
        input_array = []
        pyroscan.update_self_parameters()
        for count, combo in enumerate(
            itertools.product(*pyroscan.parameter_dict.values())
        ):
            current = dict(zip(keys, combo))  # easy access to all key–value pairs
            name = pyroscan.format_single_run_name(current)
            pyro_object = pyroscan.pyro_dict[name]
            input_dict[name] = count
            self.pyro = pyro_object
            input_array.append(self.model_input())

        input_tensor = torch.tensor(input_array, dtype=torch.float64)
        all_combined_models = []
        for model_name in self.models_specifics:
            all_models = []
            data_with_units = self.evaluate_model_multi(model_name, input_tensor)
            for count, combo in enumerate(
                itertools.product(*pyroscan.parameter_dict.values())
            ):
                current = dict(zip(keys, combo))  # easy access to all key–value pairs
                name = pyroscan.format_single_run_name(current)
                pyro_model = xr.DataArray(
                    data_with_units[
                        input_dict[name]
                    ],  # Pass the Pint Quantity directly
                    dims=("output"),
                    coords={
                        "output": ["value", "max_value", "min_value"],
                    },
                )
                for key in keys:
                    value = current[key]
                    if hasattr(value, "m"):
                        pyro_model = pyro_model.expand_dims(dim={key: [value.m]})
                        pyro_model[key].attrs["units"] = value.units
                    else:
                        pyro_model = pyro_model.expand_dims(dim={key: [None]})
                        pyro_model[key].attrs["units"] = None
                all_models.append(pyro_model)
            combined = xr.combine_by_coords(all_models)
            all_combined_models.append(xr.Dataset(data_vars={model_name: combined}))
        all_combined_models = xr.merge(all_combined_models)
        self.models = all_combined_models

    def evaluate_model_multi(self, key: str, input_tensor: torch.Tensor):
        model = self.models_specifics[key]
        value_log_tall, error_log_tall = model(input_tensor)
        value_log = np.array(value_log_tall).flatten()
        error_log = np.array(error_log_tall).flatten()
        units = self.models_specifics_units[key]
        max_value_log = value_log + error_log
        min_value_log = value_log - error_log
        value_mag = self.models_specifics_conversion[key](value_log)
        max_value_mag = self.models_specifics_conversion[key](max_value_log)
        min_value_mag = self.models_specifics_conversion[key](min_value_log)
        # print(f"key is {key}")
        # print(f"units are {units}")
        # print(f"conversion {self.models_specifics_conversion[key]}")
        if key == "kperp2_phi_log":
            print("what is going on for kperp_phi_log")
            # print(value_log)
            # print("that was the log value")
            # print("now for the real value")
            # print(value_mag)

        # Hard coding this since I don't know a better way of doing it
        # Multiplies kperp2_apa and kperp2_bpar by kperp2_phi to get correct normalisation
        if key == "kperp2_apa_log" or key == "kperp2_bpar_log":
            model_phi = self.models_specifics["kperp2_phi_log"]
            value_log_tall_phi, error_log_tall_phi = model_phi(input_tensor)
            value_log_phi = np.array(value_log_tall_phi).flatten()
            units_phi = self.models_specifics_units["kperp2_phi_log"]
            value_mag_phi = self.models_specifics_conversion["kperp2_phi_log"](
                value_log_phi
            )
            value_mag *= value_mag_phi
            max_value_mag *= value_mag_phi
            min_value_mag *= value_mag_phi

        if units is not None:
            data_with_units = (
                np.array([value_mag, max_value_mag, min_value_mag]) * units
            )
        else:
            data_with_units = np.array([value_mag, max_value_mag, min_value_mag])
        data_with_units = np.swapaxes(data_with_units, 0, 1)
        return data_with_units

    def model_input(self) -> np.array:
        """Extract parameters from the Pyro object and create a model input tensor."""
        my_convention = self.pyro.norms.pyrokinetics
        self.pyro.to(my_convention)
        numerics = self.pyro.numerics
        geom = self.pyro.local_geometry
        species = self.pyro.local_species

        ky_log = np.log10(numerics["ky"].magnitude)
        q = geom["q"].magnitude
        shat = geom["shat"].magnitude
        beta = numerics["beta"].magnitude
        if "deuterium" in species.names:
            deuterium_temp_gradient = species["deuterium"]["inverse_lt"].magnitude
        else:
            deuterium_temp_gradient = species["ion1"]["inverse_lt"].magnitude
        electron_temp_gradient = species["electron"]["inverse_lt"].magnitude
        electron_dens_gradient = species["electron"]["inverse_ln"].magnitude
        electron_nu = species["electron"]["nu"].magnitude
        return [
            ky_log,
            q,
            shat,
            beta,
            deuterium_temp_gradient,
            electron_temp_gradient,
            electron_dens_gradient,
            electron_nu,
        ]

    def load_models(self, path, kernel_names, model_kernel):
        """Load TorchScript models from a directory."""
        self.models_specifics = {}
        self.models_specifics_units = {}
        self.models_specifics_conversion = {}
        self.models_output_names = {}
        for name in kernel_names:
            for variant in model_kernel:
                model_path = Path(path) / f"{name}.pt"
                # try:
                base_name = self.strip_suffix(name)
                self.models_specifics[base_name] = torch.jit.load(model_path)
                self.models_specifics_units[base_name] = self.units_dict[base_name]
                self.models_specifics_conversion[base_name] = (
                    self.get_conversion_function(name)
                )
                # print(f"✅ Loaded: {model_path}")
                # except FileNotFoundError:
                #    print(f"⚠️ Missing: {model_path}")
                # except Exception as e:
                #    print(f"❌ Error loading {model_path}: {e}")

    def _evaluate_model(self, key: str):
        """Evaluate a TorchScript model, exponentiate outputs, and return xarray DataArray."""
        # try:
        model = self.models_specifics[key]
        value_log, error_log = model(
            self.inputs
        )  # Modify this so that you give it a array of inputs and get an array of outputs - Need to give it a torch.tensor - Talk to andy snowdon

        units = self.models_specifics_units[key]
        max_value_log = value_log + error_log
        min_value_log = value_log - error_log
        value_mag = self.models_specifics_conversion[key](value_log)
        max_value_mag = self.models_specifics_conversion[key](max_value_log)
        min_value_mag = self.models_specifics_conversion[key](min_value_log)
        if key == "kperp2_phi_log":
            print("what is going on for kperp_phi_log")
            print(value_log)
            print("that was the log value")
            print("now for the real value")
            print(value_mag)
            exit()

        # Hard coding this since I don't know a better way of doing it
        # Multiplies kperp2_apa and kperp2_bpar by kperp2_phi to get correct normalisation
        if key == "kperp2_apa_log" or key == "kperp2_bpar_log":
            value_mag *= self.models_specifics_conversion["kperp2_phi_log"](
                self.models_specifics["kperp2_phi_log"](self.inputs)[0]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            max_value_mag *= self.models_specifics_conversion["kperp2_phi_log"](
                self.models_specifics["kperp2_phi_log"](self.inputs)[0]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            min_value_mag *= self.models_specifics_conversion["kperp2_phi_log"](
                self.models_specifics["kperp2_phi_log"](self.inputs)[0]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
        if units is not None:
            data_with_units = (
                np.array([value_mag, max_value_mag, min_value_mag]).flatten() * units
            )
        else:
            data_with_units = np.array(
                [value_mag, max_value_mag, min_value_mag]
            ).flatten()
        new_model = xr.DataArray(
            data_with_units,  # Pass the Pint Quantity directly
            dims=("output"),
            coords={
                "output": ["value", "max_value", "min_value"],
            },
        )

        new_model_dataset = xr.Dataset(data_vars={key: new_model})
        return new_model_dataset

    def evaluate_all_models(self):
        """Evaluate all loaded model variants and store in a single xarray.DataArray."""
        dataarrays = []
        for (
            key
        ) in (
            self.models_specifics
        ):  # I think it should check through the model names right?
            # try:
            new_model = self._evaluate_model(key)

            if new_model is not None:
                # Check for :
                dataarrays.append(new_model)
            else:
                print(f"⚠️ No valid output for {key}")
            # except Exception as e:
            #   print(f"❌ Error evaluating {key}: {e}")

        if not dataarrays:
            raise ValueError("No valid model outputs to concatenate.")

        # Concatenate only valid DataArrays
        self.models = xr.merge(dataarrays)
