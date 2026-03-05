import matplotlib.pyplot as plt
import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.field_line import  FieldLine

# Load data
fname = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"
pyro = Pyro(gk_file=fname, gk_code="CGYRO")
pyro.load_gk_output()

# Set input for Poincare map
xarray = np.linspace(6, 8, 5) * pyro.norms.rhoref
yarray = np.linspace(-10, 10, 3) * pyro.norms.rhoref
nturns = 1000
time = 1
rhostar = 0.036

# Generate Poincare map
diag = FieldLine(pyro)
coords = diag.poincare(xarray, yarray, nturns, time, rhostar)

# Simple plot
plt.figure()
plt.plot(coords[0, :].ravel().m, coords[1, :].ravel().m, "k.")

# Plot with colors
ntot = nturns * yarray.shape[0]
colorlist = plt.cm.jet(np.linspace(0, 1, xarray.shape[0]))
plt.figure()
for i, color in enumerate(colorlist):
    plt.plot(coords[0, :, :, i].ravel().m, coords[1, :, :, i].ravel().m, ".", color=color)
plt.show()


radial_diff = diag.radial_diffusion_coefficient(xarray, yarray, nturns, time, rhostar)
print("Radial diffusion", radial_diff)

delta_r = diag.compute_half_displacement(xarray, yarray, time, rhostar)
print("Avg delta r", np.mean(delta_r))

lambda_Bxx = diag.parallel_correlation_length(time)
print("My lambda Bxx", lambda_Bxx)