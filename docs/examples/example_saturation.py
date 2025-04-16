from pyrokinetics import Pyro, PyroScan
from pyrokinetics.diagnostics.saturation_rules import SaturationRules
import numpy as np

# Choose convention
output_convention = "pyrokinetics"

# Base file
base_dir = "/common/CSD3/ir-giac2/t3d/resolution/nth128/restart6/outputs"
base_file = "r3-ky0-th00-0.in"

pyro = Pyro(gk_file=f"{base_dir}/{base_file}")

base_ky = pyro.numerics.ky.to(pyro.norms.pyrokinetics).m / 2

# Set up ky and theta0 grid
param_1 = "ky"
param_2 = "th0"

kys = np.array([2, 3, 5, 10, 20, 30, 40, 50, 70, 100, 120, 140]) * base_ky
th0s = np.array([0, 0.1, 0.2, 0.4, 1.2, 3.14])

values_1 = kys
values_2 = th0s

# Create dictionary mapping input files to parameters in scan
runfile_dict = {}
for iky, ky in enumerate(kys):
    for ith0, th0 in enumerate(th0s):
        runfile_dict[
            (f"{param_1}_{values_1[iky]}", f"{param_2}_{values_2[ith0]}")
        ] = f"{base_dir}/r3-ky{iky}-th0{ith0}-0.in"

# Dictionary of param and values
param_dict = {param_1: values_1, param_2: values_2}

# Create PyroScan object
pyro_scan = PyroScan(
    pyro,
    param_dict,
    value_fmt="d",
    value_separator="",
    parameter_separator="-",
    file_name="",
    base_directory=base_dir,
    runfile_dict=runfile_dict,
)

# Add in path to each defined parameter to scan through
pyro_scan.add_parameter_key(
    param_1, "gk_input", ["data", "kt_grids_single_parameters", "n0"]
)

# Load outputs
pyro_scan.load_gk_output(output_convention=output_convention, tolerance_time_range=0.9)

# Create saturation object
saturation = SaturationRules(pyro_scan)

# Inputs for QL model
alpha = 2.5
Q0 = 25

# Must match convention
gamma_exb = 0.04380304982261718

gk_output = saturation.mg_saturation(
    Q0=Q0,
    alpha=alpha,
    gamma_exb=gamma_exb,
    output_convention=output_convention,
    gamma_tolerance=0.3,
    equal_arc_theta=True,
    theta0_dim="th0",
)

print("GK output", gk_output)
print("Heat flux calculation", gk_output["heat"])
print("Gamma flux calculation", gk_output["particle"])
