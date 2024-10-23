import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics

# Load data
fname = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"
pyro = Pyro(gk_file=fname, gk_code="CGYRO")
pyro.load_gk_output()
diag = Diagnostics(pyro)

# Input data
# time when correlation and displacement are evaluated
time = 1
# binormal position where the correlation is evaluated
dky = pyro.gk_output["ky"].values[1]
yarray = np.linspace(-np.pi / dky, np.pi / dky, 3)
# number of radial grid points for the correlation lenght
Nx = 20
# number of increments for the correlation lenght
ndelta = 10

# Compute field line displacement
delta_r = diag.compute_half_disp(time)
delta_r = np.mean(delta_r)

# Compute the correlation length of radial perturbations
lambda_Bxx = diag.compute_corr_length(time, yarray, Nx=20, ndelta=10)
lambda_Bxx = np.mean(lambda_Bxx)

# Print results
print(f"Mean correlation lenght: {lambda_Bxx}")
print(f"Mean displacement: {delta_r}")
