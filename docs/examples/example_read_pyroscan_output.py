from pyrokinetics import Pyro, PyroScan, template_dir
import numpy as np


base_directory = template_dir / "outputs/CGYRO_linear_scan"

gk_file = base_directory / "input.cgyro"

pyro = Pyro(gk_file=gk_file)

param_key = "ky"
param_value = np.array([0.1, 0.2, 0.3])
param_dict = {param_key: param_value}

# Create PyroScan object
pyro_scan = PyroScan(pyro, param_dict, base_directory=base_directory)

pyro_scan.load_gk_output()
