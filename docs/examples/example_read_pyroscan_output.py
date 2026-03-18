"""
When reading Pyroscan outputs you have two different options

1: reconstruct the pyroscan based on the same inputs:

2: read the pyroscan json file directly.

The second option is further subdevived in two methods,

2.A: Read the base Pyro file and use the json for the configuration

2.B: (NEW!) Read a base pyro file automtically from the pyroscan and use the json for the configuration.
This option allow you to reconstruct a pyroscan fully from the folder,
without the original file used to generate the scan. This is especially useful when the
original file used to generate the scan was from a different gyrokinetic code

"""

import numpy as np

from pyrokinetics import Pyro, PyroScan, template_dir

# Method 1:
base_directory = template_dir / "outputs/CGYRO_linear_scan"

gk_file = base_directory / "input.cgyro"

pyro = Pyro(gk_file=gk_file)

param_key = "ky"
param_value = np.array([0.1, 0.2, 0.3])
param_dict = {param_key: param_value}

# Create PyroScan object
pyro_scan = PyroScan(pyro, param_dict, base_directory=base_directory)

pyro_scan.load_gk_output()


# Method 2 A:

base_directory = template_dir / "outputs/CGYRO_linear_scan"

gk_file = base_directory / "input.cgyro"

json_path = template_dir / "outputs/CGYRO_linear_scan/pyroscan.json"

pyro = Pyro(gk_file=gk_file)

pyro_scan = PyroScan(pyro, pyroscan_json=json_path)

pyro_scan.load_gk_output()


# Method 2 B:


json_path = template_dir / "outputs/CGYRO_linear_scan/pyroscan.json"

pyro_scan = PyroScan(pyroscan_json=json_path, load_base_pyro=True)

pyro_scan.load_gk_output()
