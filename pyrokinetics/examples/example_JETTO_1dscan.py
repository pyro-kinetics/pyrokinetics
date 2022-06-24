from pyrokinetics import Pyro, PyroScan, template_dir
import numpy as np

# Equilibrium file
eq_file = template_dir / "test.geqdsk"

# Kinetics data file
kinetics_file = template_dir / "jetto.cdf"

# Load up pyro object
pyro = Pyro(
    eq_file=eq_file,
    eq_type="GEQDSK",
    kinetics_file=kinetics_file,
    kinetics_type="JETTO",
)

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5, local_geometry="Miller")

# Change GK code to GS2
pyro.gk_code = "GS2"

# GS2 template input file
template_file = template_dir / "input.gs2"

# Write single input file using my own template
pyro.write_gk_file(file_name="test_jetto.gs2", template_file=template_file)

# Use existing parameter
param_1 = "ky"
values_1 = np.arange(0.1, 0.3, 0.1)

# Add new parameter to scan through
param_2 = "my_electron_gradient"
values_2 = np.arange(0.0, 1.5, 0.5)

# Dictionary of param and values
param_dict = {param_1: values_1, param_2: values_2}

# Create PyroScan object
pyro_scan = PyroScan(
    pyro,
    param_dict,
    value_fmt=".3f",
    value_separator="_",
    parameter_separator="_",
    file_name="mygs2.in",
    base_directory="test_GS2",
)

pyro_scan.add_parameter_key(param_2, "local_species", ["electron", "a_lt"])

pyro_scan.write()

pyro.gk_code = "CGYRO"

pyro_scan.write(file_name="input.cgyro", base_directory="test_CGYRO")
