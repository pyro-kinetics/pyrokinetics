import itertools
from itertools import product
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from astropy import units as u
from astropy.units import Quantity

from pyrokinetics import Pyro, PyroScan
from pyrokinetics.pyroscan import PyroScanGKOutput

pyro = Pyro(gk_code="GS2")  # check units with bahvin


default_unit_dict = {
    "growth_rate_log": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "mode_frequency_log": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "kperp2_phi_log": 1 / pyro.norms.pyrokinetics.rhoref**2,
    "kperp2_apa_log": 1 / pyro.norms.pyrokinetics.rhoref**2,
    "kperp2_bpar_log": 1
    / pyro.norms.pyrokinetics.rhoref
    ** 2,  # Kperp2 is normalised to ky^2 - The others need to be multiplied by kperpe2_phi, thereore need to make sure that model is always loaded
    "totIonFlux_log": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "totElecFlux_log": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "totPartFlux_log": pyro.norms.pyrokinetics.nref
    * pyro.norms.pyrokinetics.vref
    * (pyro.norms.pyrokinetics.rhoref / pyro.norms.pyrokinetics.lref),
    "apa_phi_log": pyro.norms.pyrokinetics.bref
    * pyro.norms.pyrokinetics.rhoref**2
    / pyro.norms.pyrokinetics.lref,
    "bpar_phi_log": pyro.norms.pyrokinetics.bref
    * pyro.norms.pyrokinetics.rhoref
    / pyro.norms.pyrokinetics.lref,
    "Lambda_log": pyro.norms.pyrokinetics.rhoref,  # fix this
    "theta0fit_output_sigmas_log": pyro.norms.pyrokinetics.rhoref,
}
### Need to get the correct units for this

default_ouput_conversion_dict = {
    "growth_rate_log": lambda x: np.power(10, x) - 0.1,
    "mode_frequency_log": lambda x: np.power(10, x),
    "kperp2_phi_log": lambda x: np.power(10, x),
    "kperp2_apa_log": lambda x: np.power(10, x),
    "kperp2_bpar_log": lambda x: np.power(10, x),  # check the order of opterations here
    "totIonFlux_log": lambda x: np.power(10, x),
    "totElecFlux_log": lambda x: np.power(10, x),
    "totPartFlux_log": lambda x: np.power(10, x),
    "apa_phi_log": lambda x: np.power(10, x),
    "bpar_phi_log": lambda x: np.power(10, x),
    "Lambda_log": lambda x: np.power(10, x),
    "theta0fit_output_sigmas_log": lambda x: np.power(10, x),
}


class gs2_gp:

    def __init__(
        self,
        pyro: Pyro,
        models_path,
        models,
        model_kernel=["M12", "M32", "M52"],  # This is the kernel - rename
        units_dict=default_unit_dict,
        ouput_conversion_dict=default_ouput_conversion_dict,
    ):
        """
        If `pyro` is a Pyro object → evaluate single case.
        If `pyro` is a PyroScan → evaluate all cases and combine results.
        """
        self.model_kernel = model_kernel
        self.models_path = models_path
        self.model_names = models
        self.units_dict = units_dict
        self.ouput_conversion_dict = ouput_conversion_dict

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

    # ------------------------------
    # Single Pyro evaluation
    # ------------------------------
    def _evaluate_single(self, pyro: Pyro):
        """Evaluate models for a single Pyro object."""
        self.pyro = pyro
        self.inputs = torch.tensor([self.model_input()], dtype=torch.float32)
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

        keys = list(pyroscan.parameter_dict.keys())
        input_dict = {}
        input_array = []

        for count, combo in enumerate(
            itertools.product(*pyroscan.parameter_dict.values())
        ):
            current = dict(zip(keys, combo))  # easy access to all key–value pairs
            name = pyroscan.format_single_run_name(current)
            pyro_object = pyroscan.pyro_dict[name]
            input_dict[name] = count
            self.pyro = pyro_object
            input_array.append(self.model_input())

        input_tensor = torch.tensor(input_array, dtype=torch.float32)
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
                        "output": ["value", "error"],
                    },
                )
                for key in keys:
                    value = current[key]
                    pyro_model = pyro_model.expand_dims(dim={key: [value.m]})
                    pyro_model[key].attrs["units"] = value.units
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
        value_mag = self.models_specifics_conversion[key](value_log)
        error_mag = self.models_specifics_conversion[key](error_log)

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
            error_mag *= self.models_specifics_conversion["kperp2_phi_log"](
                self.models_specifics["kperp2_phi_log"](self.inputs)[0]
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
        data_with_units = np.array([value_mag, error_mag]) * units
        data_with_units = np.swapaxes(data_with_units, 0, 1)
        return data_with_units

    def model_input(self) -> np.array:
        """Extract parameters from the Pyro object and create a model input tensor."""
        my_convention = self.pyro.norms.pyrokinetics
        self.pyro.numerics.with_units(my_convention)
        numerics = self.pyro.numerics
        self.pyro.local_geometry.normalise(my_convention)
        geom = self.pyro.local_geometry
        self.pyro.local_species.normalise(my_convention)  #why is this throwing an error
        species = self.pyro.local_species

        ky_log = np.log10(numerics["ky"].magnitude)
        q = geom["q"].magnitude
        shat = geom["shat"].magnitude
        beta = numerics["beta"].magnitude

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
        for name in kernel_names:
            for variant in model_kernel:
                model_path = (
                    Path(path) / f"output_{name}_warping_True_kernel_{variant}.pt"
                )
                try:
                    self.models_specifics[f"{name}_{variant}"] = torch.jit.load(
                        model_path
                    )
                    self.models_specifics_units[f"{name}_{variant}"] = self.units_dict[
                        name
                    ]
                    self.models_specifics_conversion[f"{name}_{variant}"] = (
                        self.ouput_conversion_dict[name]
                    )
                    # print(f"✅ Loaded: {model_path}")
                except FileNotFoundError:
                    print(f"⚠️ Missing: {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")

    def _evaluate_model(self, key: str):
        """Evaluate a TorchScript model, exponentiate outputs, and return xarray DataArray."""
        # try:
        model = self.models_specifics[key]
        value_log, error_log = model(
            self.inputs
        )  # Modify this so that you give it a array of inputs and get an array of outputs - Need to give it a torch.tensor - Talk to andy snowdon

        units = self.models_specifics_units[key]

        value_mag = self.models_specifics_conversion[key](
            value_log.detach().cpu().numpy().squeeze()
        )
        error_mag = self.models_specifics_conversion[key](
            error_log.detach().cpu().numpy().squeeze()
        )

        # Hard coding this since I don't know a better way of doing it
        # Multiplies kperp2_apa and kperp2_bpar by kperp2_phi to get correct normalisation
        if key == "kperp2_apa_log" or key == "kperp2_bpar_log":
            values_kperp2_phi_log, _ = self.models_specifics[self.inputs]
            value_kperp2_phi_mag = self.models_specifics_conversion[key](
                values_kperp2_phi_log.detach().cpu().numpy().squeeze()
            )
            value_mag *= value_kperp2_phi_mag
            error_mag *= value_kperp2_phi_mag

        # 🚨 CRITICAL FIX: Wrap the magnitudes together with the unit 🚨
        data_with_units = np.array([value_mag, error_mag]) * units

        # Pass the Pint Quantity directly to xr.DataArray (no np.array() needed)
        new_model = xr.DataArray(
            data_with_units,  # Pass the Pint Quantity directly
            dims=("output"),
            coords={
                "output": ["value", "error"],
            },
        )

        new_model_dataset = xr.Dataset(data_vars={key: new_model})
        return new_model_dataset

        # except Exception as e:
        #   print(f"An error occurred: {e}")
        # Handle the error appropriately
        # return None or raise

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
                # Check for NaN
                dataarrays.append(new_model)
            else:
                print(f"⚠️ No valid output for {key}")
            # except Exception as e:
            #   print(f"❌ Error evaluating {key}: {e}")

        if not dataarrays:
            raise ValueError("No valid model outputs to concatenate.")

        # Concatenate only valid DataArrays
        self.models = xr.merge(dataarrays)
