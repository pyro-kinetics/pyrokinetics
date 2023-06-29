from pyrokinetics import Pyro
import matplotlib.pyplot as plt
import numpy as np
import xrft

cgyro_template = "C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_nonlinear_small\input.cgyro"

# Load in file
pyro = Pyro(gk_file=cgyro_template)

pyro.load_gk_output(load_moments=True, load_fluxes=False, load_fields=False)

data = pyro.gk_output.data
density = data['density'].pint.dequantify()
density = density.isel(time=-1).sel(theta=0.0, method="nearest").sel(species='electron')

nkx = len(density.kx)
nky = len(density.ky)

rs_density = xrft.ifft(density, real_dim="ky", true_amplitude=True, true_phase=False)
max_ne = np.max(rs_density.data)
min_ne = np.min(rs_density.data)
levels = np.arange(min_ne, max_ne, (max_ne-min_ne)/256)

ky = pyro.numerics.ky
shat = pyro.local_geometry.shat
length = pyro.cgyro_input["BOX_SIZE"] / (ky * shat)

nx = len(rs_density.freq_kx.data)
x = np.linspace(-1 / 2, 1 / 2, nx) * length

ny = len(rs_density.freq_ky.data)
y = np.linspace(-np.pi / ky, np.pi / ky, ny)


plt.contourf(x, y, rs_density.data.T, levels, cmap=plt.get_cmap("jet"))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$\delta n_e$')
plt.grid()
ax = plt.gca()

ax.set_aspect("equal")
plt.show()
