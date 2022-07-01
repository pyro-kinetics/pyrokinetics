from pyrokinetics import Pyro, template_dir
import matplotlib.pyplot as plt
import numpy as np

# Point to CGYRO input file
cgyro_template = template_dir / "outputs/CGYRO_linear/input.cgyro"

# Load in file
pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

# Get eigenvalues
eigenvalues = data["eigenvalues"]
growth_rate = data["growth_rate"]
mode_freq = data["mode_frequency"]

# Plot growth and mode frequency
growth_rate.plot(x="time")
plt.show()

mode_freq.plot(x="time")
plt.show()

# Plot eigenfunction
phi_eig = np.real(data["eigenfunctions"].sel(field="phi").isel(time=-1))
phi_eig.plot(x="theta")

phi_i_eig = np.imag(data["eigenfunctions"].sel(field="phi").isel(time=-1))
phi_i_eig.plot(x="theta")
plt.show()

# Plot electron energy flux
energy_flux = (
    data["fluxes"]
    .sel(field="phi", species="electron", moment="energy")
    .sum(dim="ky")
    .plot.line()
)
plt.show()

# Plot phi
phi = (
    data["fields"]
    .sel(field="phi")
    .sel(theta=0.0, method="nearest")
    .isel(ky=0)
    .isel(kx=0)
)
phi = np.abs(phi)
phi.plot.line(x="time")

plt.yscale("log")
plt.show()
