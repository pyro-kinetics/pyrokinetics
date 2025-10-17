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

pyro = Pyro(gk_code="CGYRO")


default_unit_dict = {
    "growth_rate_log": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "mode_frequency_log": pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref,
    "kperp2_phi_log": pyro.norms.pyrokinetics.rhoref,
    "kperp2_apa_log": pyro.norms.pyrokinetics.rhoref,
    "kperp2_bpar_log": pyro.norms.pyrokinetics.rhoref,
    "totIonFlux_log": pyro.norms.pyrokinetics.rhoref,
    "totElecFlux_log": pyro.norms.pyrokinetics.rhoref,
    "totPartFlux_log": pyro.norms.pyrokinetics.rhoref,
    "apa_phi_log": pyro.norms.pyrokinetics.rhoref,
    "bpar_phi_log": pyro.norms.pyrokinetics.rhoref,
}
### Need to get the correct units for this


class gs2_gp:

    def __init__(
        self,
        pyro: Pyro,
        models_path,
        models,
        model_variants=["M12", "M32", "M52"],
        units_dict=default_unit_dict,
    ):
        """
        If `pyro` is a Pyro object → evaluate single case.
        If `pyro` is a PyroScan → evaluate all cases and combine results.
        """
        self.model_variants = model_variants
        self.models_path = models_path
        self.model_names = models
        self.units_dict = units_dict

        # Load models once
        self.load_models(models_path, models, model_variants)

        # Determine what was passed
        if isinstance(pyro, Pyro):
            self._evaluate_single(pyro)
            self.convert_to_GKoutput()
        elif isinstance(pyro, PyroScan):
            self._evaluate_scan(pyro)
            self.convert_to_GKoutput()
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

    def convert_to_GKoutput(self):
        # print(self.models)
        pyroutput = PyroScanGKOutput(self.models)
        my_convention = self.pyro.norms.pyrokinetics
        # print(pyroutput)
        pyroutput.to(my_convention)
        # convert xarray from normalisation of pysocan to nomarlisation of
        self.gk_output = pyroutput

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
                pyro_model[key].attrs["units"] = value.units

            # print(pyro_model)

            all_models.append(pyro_model)
        combined = xr.combine_by_coords(all_models)
        self.models = combined

    def load_models(self, path, kernel_names, model_variants):
        """Load TorchScript models from a directory."""
        self.models_specifics = {}
        self.models_specifics_units = {}
        for name in kernel_names:
            for variant in model_variants:
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
                    # print(f"✅ Loaded: {model_path}")
                except FileNotFoundError:
                    print(f"⚠️ Missing: {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")

    def _prepare_inputs(self) -> torch.Tensor:
        """Extract parameters from the Pyro object and create a model input tensor."""
        my_convention = self.pyro.norms.pyrokinetics
        self.pyro.numerics.with_units(my_convention)
        numerics = self.pyro.numerics
        self.pyro.local_geometry.normalise(my_convention)
        geom = self.pyro.local_geometry
        # self.pyro.local_species.normalise(my_convention)  #why is this throwing an error
        species = self.pyro.local_species

        ky_log = np.log10(numerics["ky"].magnitude)
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
        try:
            model = self.models_specifics[key]
            value_log, error_log = model(self.inputs)

            units = self.models_specifics_units[key]

            # Calculate the Pint Quantities
            # .magnitude extracts the number from the Quantity
            value_mag = np.power(10,value_log.detach().cpu().numpy().squeeze())
            error_mag = np.power(10,error_log.detach().cpu().numpy().squeeze())

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

        except Exception as e:
            print(f"An error occurred: {e}")
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
            try:
                new_model = self._evaluate_model(key)

                if new_model is not None:
                    # Check for NaN
                    dataarrays.append(new_model)
                else:
                    print(f"⚠️ No valid output for {key}")
            except Exception as e:
                print(f"❌ Error evaluating {key}: {e}")

        if not dataarrays:
            raise ValueError("No valid model outputs to concatenate.")

        # Concatenate only valid DataArrays
        self.models = xr.merge(dataarrays)
