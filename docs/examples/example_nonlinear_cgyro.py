from pyrokinetics import Pyro, template_dir
import matplotlib.pyplot as plt
import numpy as np

# Point to CGYRO input file
cgyro_template = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"

# Load in file
pyro = Pyro(gk_file=cgyro_template)

# Load in CGYRO output data
pyro.load_gk_output(load_fields=True, load_fluxes=True, load_moments=True)
data = pyro.gk_output

# Plot electron energy flux as a function of ky
electron_heat_flux_ky = data["heat"].sel(field="phi", species="electron").isel(time=-1)
electron_heat_flux_ky.plot()
plt.show()

# Plot electron energy flux as a function of ky
total_heat_flux_time = data["heat"].sum(dim=["field", "species", "ky"])
total_heat_flux_time.plot()
plt.show()

# Plot phi
phi = data["phi"].sel(theta=0.0, method="nearest", drop=True).isel(time=-1, drop=True)
# Can't log something with units to need to remove them
log_phi = np.log(np.abs(phi).pint.dequantify())
log_phi.plot(x="kx", y="ky")
plt.show()
