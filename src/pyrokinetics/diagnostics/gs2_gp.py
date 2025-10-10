from pathlib import Path

import numpy as np
import torch

from pyrokinetics import Pyro


class gs2_gp:

    def __init__(self, pyro: Pyro, models_path):

        self.pyro = pyro
        self.load_models(models_path)
        self._prepare_inputs()
        self.get_model_frequency()
        self.get_model_growth_rates()

    def load_models(self, path):
        """Load TorchScript models from a directory."""
        kernel_names = [
            "growth_rate_log",
            "mode_frequency_log",
            "kperp2_phi_log",
            "kperp2_apa_log",
            "kperp2_bpar_log",
            "totIonFlux_log",
            "totElecFlux_log",
            "totPartFlux_log",
            "apa_phi_log",
            "bpar_phi_log",
        ]
        model_variants = ["M12", "M32", "M52"]

        self.models = {}
        self.loaded_files = []

        for name in kernel_names:
            for variant in model_variants:
                try:
                    model_path = (
                        Path(path)
                        / "output_"
                        / name
                        / "_warping_True_kernel_"
                        / variant
                        / ".pt"
                    )
                    self.models.setdefault(name, {})[variant] = torch.jit.load(
                        model_path
                    )

                    self.loaded_files.append(model_path)
                    print(f"Loaded: {model_path}")
                except FileNotFoundError:
                    print(f"Warning: {model_path} not found.")
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")

        print("All models loaded")

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

    def get_model_frequency(self):
        self.model_frequency, self.model_frequency_error = self.models[
            "mode_frequency_log"
        ](self.inputs)

    def get_model_growth_rates(self):
        self.model_growth_rate, self.model_growth_rate_error = self.models[
            "growth_rate_log"
        ](self.inputs)
