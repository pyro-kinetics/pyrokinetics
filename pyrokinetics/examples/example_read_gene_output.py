from pyrokinetics import Pyro
import matplotlib.pyplot as plt
import numpy as np

# Point to GENE input file
gene_template = "parameters_0005"

# Load in file
pyro = Pyro(gk_file=gene_template, gk_code="GENE")

# Load in GENE output data
pyro.load_gk_output(gene_output_number="0005")
data = pyro.gk_output.data

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
    .sel(species="electron", moment="energy")
    .sum(dim=["field"])
    .plot.line()
)
plt.show()

# Plot phi
phi = (
    data["fields"]
    .sel(field="phi")
    .isel(ky=0)
    .isel(kx=0)
    .sel(theta=0.0, method="nearest")
)
phi = np.abs(phi)
phi.plot.line(x="time")

plt.yscale("log")
plt.show()
