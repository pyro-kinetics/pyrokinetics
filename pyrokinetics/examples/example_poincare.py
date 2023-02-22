import matplotlib.pyplot as plt
import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics

# Load data
fname = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"
pyro = Pyro(gk_file=fname, gk_code="CGYRO")
pyro.load_gk_output()

# Set input for Poincare map
xarray = np.linspace(6, 8, 5)
yarray = np.linspace(-10, 10, 3)
nturns = 1000
time = 1
rhostar = 0.036

# Generate Poincare map
diag = Diagnostics(pyro)
coords = diag.poincare(xarray, yarray, nturns, time, rhostar)

# Simple plot
plt.figure()
plt.plot(coords[0, :], coords[1, :], 'k.')
plt.show()
