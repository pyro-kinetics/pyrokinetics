import os
import sys
import torch
from pyrokinetics import Pyro,PyroScan
from pyrokinetics.diagnostics.gs2_gp import gs2_gp
import numpy as np

# load example GS2 input file
Template_Path = "/home/Felix/Documents/Physics_Work/Project_Codes/GP_Model_Eval/GS2/Templates/r1.in"
pyro = Pyro(gk_file=Template_Path)

# load models
models_path = "/home/Felix/Documents/Physics_Work/Project_Codes/8d_3000/"

# models = [
#             "growth_rate_log", "mode_frequency_log", "kperp2_phi_log", "kperp2_apa_log",
#             "kperp2_bpar_log", "totIonFlux_log", "totElecFlux_log", "totPartFlux_log",
#             "apa_phi_log", "bpar_phi_log"
#         ]


models = [
            "growth_rate_log", "mode_frequency_log",
        ]



my_models = gs2_gp(pyro=pyro, models_path=models_path, models=models)

gr = my_models.models
print("gr")
print(gr)

 
pyro.numerics.nky = 1
pyro.numerics.gamma_exb = 0.0
pyro.local_species.electron.domega_drho = 0.0

# Use existing parameter with more realistic ky range
param_1 = "ky" 
values_1 = np.arange(0.1, 0.2, 0.1)

# Add beta parameter with realistic values
param_2 = "beta"
values_2 = np.arange(0.01, 0.30, 0.01)

# Dictionary of param and values
param_dict = {param_1: values_1, param_2: values_2}

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
    parameter_key="beta",
    parameter_attr="numerics", 
    parameter_location=["beta"]
)

# Add function to tglf
pyro_scan_tglf.add_parameter_func(param_2, enforce_beta_prime, param_2_kwargs)


my_models = gs2_gp(pyro=pyro_scan_tglf, models_path=models_path, models=models)

print(my_models.gk_output.sel())
print(my_models.gk_output["growth_rate_log_M52"].ky)
print(my_models.gk_output["growth_rate_log_M52"].beta)