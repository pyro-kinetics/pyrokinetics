import os
import sys
from scipy.constants import sigma
import torch
from torch._prims_common import validate_no_repeating_dims
from pyrokinetics import Pyro, PyroScan, template_dir
import pyrokinetics
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np
from pathlib import Path
from scipy.special import erf

# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/GP-MODELS/0.0.2"

models = [
    "growth_rate",
    "mode_frequency_log",
    "kperp2_phi_log",
    "kperp2_apa_log",
    "kperp2_bpar_log",
    "totIonFlux_log",
    "totElecFlux_log",
    "totPartFlux",
    "apa_phi_log",
    "bpar_phi_log",
]


pyro = Pyro(gk_file=template_dir / "input.gs2")

pyro.numerics.nky = 1
pyro.numerics.gamma_exb = 0.1
pyro.local_species.electron.domega_drho = 0.0

# Use existing parameter with more realistic ky range
param_1 = "ky"
values_1 = np.arange(0.1, 1.0, 0.1) / pyro.norms.pyrokinetics.rhoref

# Add beta parameter with realistic values
param_2 = "beta"
values_2 = np.arange(0.01, 0.03, 0.01)

param_3 = "gamma_exb"
values_3 = (
    np.arange(0, 1, 0.1) * pyro.norms.pyrokinetics.vref / pyro.norms.pyrokinetics.lref
)


# Dictionary of param and values
param_dict = {param_1: values_1, param_2: values_2, param_3: values_3}
# param_dict = {param_1: values_1}


def enforce_beta_prime(pyro):
    pyro.enforce_consistent_beta_prime()


# If there are kwargs to function then define here
param_2_kwargs = {}

# Switch to TGLF
pyro.gk_code = "TGLF"

# Create PyroScan object with more descriptive naming
pyro_scan_tglf = PyroScan(
    pyro,
    param_dict,
    value_fmt=".4f",  # Increased precision for small beta values
    value_separator="_",
    parameter_separator="_",
    file_name="input.tglf",
)

# Add function to enforce consistent beta prime
pyro_scan_tglf.add_parameter_key(
    parameter_key="beta", parameter_attr="numerics", parameter_location=["beta"]
)

pyro_scan_tglf.add_parameter_key(
    parameter_key="gamma_exb",
    parameter_attr="numerics",
    parameter_location=["gamma_exb"],
)

# Add function to tglf
pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)

my_models = gs2_gp(pyro=pyro_scan_tglf, models_path=models_path, models=models)

print(my_models.gk_output)
print(my_models.gk_output["growth_rate"].coords["ky"].values)
breakpoint()
my_models.evaluate_nonlinear_flux()

print("fluxes are here")

print(my_models.flux_Ion.data)
print(my_models.flux_Elec)
print(my_models.flux_Part)
print("calculated fluxes")
