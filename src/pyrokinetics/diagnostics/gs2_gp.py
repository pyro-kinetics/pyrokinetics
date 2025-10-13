from pathlib import Path
import numpy as np
import torch
import xarray as xr

from pyrokinetics import Pyro




class gs2_gp:

    def __init__(self, pyro: Pyro, models_path, models,model_varaiants=["M12","M32","M52"]):
        self.pyro = pyro
        self.load_models(models_path,models,model_varaiants) 
        self._prepare_inputs()
        self.evaluate_all_models()
    
    def load_models(self, path, kernel_names,model_variants):
        """Load TorchScript models from a directory."""
        self.models = {}

        for name in kernel_names:
            for variant in model_variants:
                model_path = Path(path) / f"output_{name}_warping_True_kernel_{variant}.pt"
                try:
                    self.models[f"{name}_{variant}"] = torch.jit.load(model_path)
                    print(f"✅ Loaded: {model_path}")
                except FileNotFoundError:
                    print(f"⚠️ Missing: {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")


    def _prepare_inputs(self) -> torch.Tensor:
        """Extract parameters from the Pyro object and create a model input tensor."""
        numerics = self.pyro.numerics
        geom = self.pyro.local_geometry
        species = self.pyro.local_species

        ky_log = np.log(numerics["ky"].magnitude)
        q = geom["q"].magnitude
        shat = geom["shat"].magnitude
        beta = numerics["beta"].magnitude

        deuterium_temp_gradient = species["ion1"]["inverse_lt"].magnitude
        electron_temp_gradient = species["electron"]["inverse_lt"].magnitude
        electron_dens_gradient = species["electron"]["inverse_ln"].magnitude
        electron_nu = species["electron"]["nu"].magnitude

        self.inputs = torch.tensor([
            [ky_log, q, shat, beta,
             deuterium_temp_gradient,
             electron_temp_gradient,
             electron_dens_gradient,
             electron_nu]
        ], dtype=torch.float32)

    def _evaluate_model(self, key: str):
        """Evaluate a TorchScript model, exponentiate outputs, and return xarray DataArray."""
        try:
            model = self.models[key]  # ✅ use the key directly
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
            return new_model

        except Exception as e:
            print(f"Error evaluating model '{key}': {e}")
            return None



    def evaluate_all_models(self):
        """Evaluate all loaded model variants and store in a single xarray.DataArray."""
        dataarrays = []

        for key, model in self.models.items():
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



