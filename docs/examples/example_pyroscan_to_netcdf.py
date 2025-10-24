from pyrokinetics import Pyro, PyroScan
import matplotlib.pyplot as plt
import numpy as np

file_name = "input.cgyro"

pyro = Pyro(gk_file=file_name)

# Create reference units and apply to Pyro
units = pyro.norms.units
reference_values = {
    "tref_electron": 1 * units.eV,
    "nref_electron": 1e19 * units.meter**-3,
    "bref_B0": 0.5 * units.tesla,
    "lref_minor_radius": 1.5 * units.meter,
}
pyro.set_reference_values(**reference_values)

# Write to file for later usage
pyro.write_reference_values("pyro_reference_values.json")

# Use existing parameter
param_1 = "ky"
values_1 = np.geomspace(1, 10000, 41) * pyro.numerics.ky.to(pyro.norms.pyrokinetics)

# Dictionary of param and values
param_dict = {param_1: values_1}

# Run directory
base_directory = "./scan"

# Load in file
pyro.write_gk_file(file_name=f"{base_directory}/{file_name}")


# Create PyroScan object
pyro_scan = PyroScan(
    pyro,
    param_dict,
    value_fmt=".3f",
    value_separator="_",
    parameter_separator="_",
    file_name=file_name,
    base_directory=base_directory,
)


def reduce_time_step(pyro):
    if np.abs(pyro.numerics.ky.m) >= 1.0:
        pyro.numerics.delta_time *= 1.0 / np.abs(pyro.numerics.ky.m)
        pyro.numerics.max_time *= 1.0 / np.abs(pyro.numerics.ky.m)


# If there are kwargs to function then define here
param_1_kwargs = {}

# Add function to pyro
pyro_scan.add_parameter_func(param_1, reduce_time_step, param_1_kwargs)

# Load output
pyro_scan.load_gk_output()

# Prepare data for plotting
data = pyro_scan.gk_output
growth_rate = data["growth_rate"]
mode_frequency = data["mode_frequency"]
growth_rate_tolerance = data["growth_rate_tolerance"]
growth_rate = growth_rate.where(growth_rate_tolerance < 0.5)
mode_frequency = mode_frequency.where(growth_rate_tolerance < 0.5)
mode_frequency = mode_frequency.where(growth_rate > 0.0)
growth_rate = growth_rate.where(growth_rate > 0.0)

# Save pyroscan output to file
pyro_scan.gk_output.to_netcdf("pyroscan.cdf")


# Load in new PyroScan
new_pyro = Pyro(gk_file="input.cgyro")

# Load reference values in
new_pyro.read_reference_values("pyro_reference_values.json")

new_pyro_scan = PyroScan(new_pyro, pyroscan_json="scan/pyroscan.json")

# Load outputs using PyroScan method
new_pyro_scan.load_gk_output(netcdf_file="pyroscan.cdf")

# Prepare data for plotting
new_data = new_pyro_scan.gk_output.data
new_growth_rate = new_data["growth_rate"]
new_mode_frequency = new_data["mode_frequency"]
new_growth_rate_tolerance = new_data["growth_rate_tolerance"]
new_growth_rate = new_growth_rate.where(new_growth_rate_tolerance < 0.5)
new_mode_frequency = new_mode_frequency.where(new_growth_rate_tolerance < 0.5)
new_mode_frequency = new_mode_frequency.where(new_growth_rate > 0.0)
new_growth_rate = new_growth_rate.where(new_growth_rate > 0.0)


print("Original data")
print("Simulation units")
print(growth_rate.data)
print("Physical units")
print(growth_rate.data.to("seconds**-1"))

print("")
print("NetCDF data")
print("Simulation units")
print(new_growth_rate.data)
print("Physical units")
print(new_growth_rate.data.to("seconds**-1"))

# Plot to compare
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 9))

ax1.plot(
    growth_rate.ky, growth_rate.data, "o", color="C0", fillstyle="none", markersize=8
)
ax2.plot(
    mode_frequency.ky,
    mode_frequency.data,
    "o",
    color="C0",
    fillstyle="none",
    markersize=8,
)

ax1.plot(
    new_growth_rate.ky,
    new_growth_rate.data,
    "x",
    color="C1",
    fillstyle="none",
    markersize=8,
)
ax2.plot(
    new_mode_frequency.ky,
    new_mode_frequency.data,
    "x",
    color="C1",
    fillstyle="none",
    markersize=8,
)

ax1.grid(True)
ax2.grid(True)

ax1.set_yscale("log")
ax1.set_xscale("log")
ax2.set_yscale("symlog")
ax1.set_ylabel(r"Growth rate: $\gamma (c_{s}/a)$")
ax2.set_ylabel(r"Mode frequency: $\omega (c_{s}/a)$")
ax2.set_xlabel(r"Binormal wavenumber: $k_y\rho_s$")


fig.tight_layout()
plt.show()
