import itertools
import re
from itertools import product
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from astropy import units as u
from astropy.units import Quantity

from pyrokinetics import Pyro, PyroScan
from pyrokinetics.pyroscan import PyroScanGKOutput


class gs2_gp:

    def __init__(
        self, pyro: Pyro, models_path, models, model_variants=["M12", "M32", "M52"]
    ):
        """
        If `pyro` is a Pyro object → evaluate single case.
        If `pyro` is a PyroScan → evaluate all cases and combine results.
        """
        self.model_variants = model_variants
        self.models_path = models_path
        self.model_names = models

        # Load models once
        self.load_models(models_path, models, model_variants)

        # Determine what was passed
        if isinstance(pyro, Pyro):
            self._evaluate_single(pyro)
        elif isinstance(pyro, PyroScan):
            self._evaluate_scan(pyro)
        else:
            raise TypeError(f"Expected Pyro or PyroScan, got {type(pyro)}")

    # ------------------------------
    # Single Pyro evaluation
    # ------------------------------
    def _evaluate_single(self, pyro: Pyro):
        """Evaluate models for a single Pyro object."""
        self.pyro = pyro
        self._prepare_inputs()
        self.evaluate_all_models()

    def outer_product(self):
        return (
            dict(zip(self.parameter_dict, x))
            for x in product(*self.parameter_dict.values())
        )

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

    # ------------------------------
    # PyroScan evaluation
    # ------------------------------
    def _evaluate_scan(self, pyroscan):
        """Evaluate models for every Pyro in a PyroScan."""

        all_models = []

        keys = list(pyroscan.parameter_dict.keys())

        run_keys = list(pyroscan.pyro_dict.keys())

        for count, combo in enumerate(
            itertools.product(*pyroscan.parameter_dict.values())
        ):
            current = dict(zip(keys, combo))  # easy access to all key–value pairs

            name = pyroscan.format_single_run_name(current)
            pyro_object = pyroscan.pyro_dict[name]
            self._evaluate_single(pyro_object)
            pyro_model = self.models
            for key in keys:
                value = current[key]
                pyro_model = pyro_model.expand_dims(dim={key: [value.m]})
                pyro_model[key].attrs["units"] = str(
                    value.units
                )  # make sure this is the case for all the ouputs

            all_models.append(pyro_model)
        combined = xr.combine_by_coords(all_models)
        pyroutput = PyroScanGKOutput(combined)
        my_convention = self.pyro.norms.pyrokinetics
        # pyroutput.to(my_convention)
        # convert xarray from normalisation of pysocan to nomarlisation of
        self.scan_ouput = pyroutput

    def load_models(self, path, kernel_names, model_variants):
        """Load TorchScript models from a directory."""
        self.models_specifics = {}

        for name in kernel_names:
            for variant in model_variants:
                model_path = (
                    Path(path) / f"output_{name}_warping_True_kernel_{variant}.pt"
                )
                try:
                    self.models_specifics[f"{name}_{variant}"] = torch.jit.load(
                        model_path
                    )
                    # print(f"✅ Loaded: {model_path}")
                except FileNotFoundError:
                    print(f"⚠️ Missing: {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")

    def _prepare_inputs(self) -> torch.Tensor:
        """Extract parameters from the Pyro object and create a model input tensor."""
        my_convention = self.pyro.norms.pyrokinetics
        # numerics = self.pyro.numerics.with_units(my_convention)
        numerics = self.pyro.numerics
        geom = self.pyro.local_geometry
        # geom = self.pyro.local_geometry.normalise(my_convention)
        # species = self.pyro.local_species.normalise(my_convention)
        species = self.pyro.local_species

        ky_log = np.log(numerics["ky"].magnitude)
        q = geom["q"].magnitude
        shat = geom["shat"].magnitude
        beta = numerics["beta"].magnitude

        deuterium_temp_gradient = species["ion1"]["inverse_lt"].magnitude
        electron_temp_gradient = species["electron"]["inverse_lt"].magnitude
        electron_dens_gradient = species["electron"]["inverse_ln"].magnitude
        electron_nu = species["electron"]["nu"].magnitude

        self.inputs = torch.tensor(
            [
                [
                    ky_log,
                    q,
                    shat,
                    beta,
                    deuterium_temp_gradient,
                    electron_temp_gradient,
                    electron_dens_gradient,
                    electron_nu,
                ]
            ],
            dtype=torch.float32,
        )

    def _evaluate_model(self, key: str):
        """Evaluate a TorchScript model, exponentiate outputs, and return xarray DataArray."""
        # try:
        model = self.models_specifics[key]  # ✅ use the key directly
        value_log, error_log = model(self.inputs)

        value = np.exp(value_log.detach().cpu().numpy().squeeze())
        error = np.exp(error_log.detach().cpu().numpy().squeeze())

        new_model = xr.DataArray(
            np.array([[value, error]]),
            dims=("model", "output"),
            coords={
                "model": [key],
                "output": ["value", "error"],
            },
        )
        #### add units here
        return new_model

        # except Exception as e:
        #     print(f"Error evaluating model '{key}': {e}")
        #     return None

    def evaluate_all_models(self):
        """Evaluate all loaded model variants and store in a single xarray.DataArray."""
        dataarrays = []
        for (
            key
        ) in (
            self.models_specifics
        ):  # I think it should check through the model names right?
            try:
                new_model = self._evaluate_model(key)
                if new_model is not None:
                    # Check for NaN
                    if np.any(np.isnan(new_model.values)):
                        print(f"⚠️ Model {key} produced NaNs, skipping.")
                    else:
                        dataarrays.append(new_model)
                else:
                    print(f"⚠️ No valid output for {key}")
            except Exception as e:
                print(f"❌ Error evaluating {key}: {e}")

        if not dataarrays:
            raise ValueError("No valid model outputs to concatenate.")

        # Concatenate only valid DataArrays
        self.models = xr.concat(dataarrays, dim="model")
