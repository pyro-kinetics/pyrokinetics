from pyrokinetics import Pyro
import numpy as np
import matplotlib.pyplot as plt

pyro = Pyro(gk_file="input.tglf", gk_code="TGLF")

pyro.load_gk_output()

# Plot dominant eigenfunction
dominant = pyro.gk_output["eigenfunctions"].isel(mode=0)

np.real(dominant.sel(field="phi")).plot(x="theta", marker="x")
np.imag(dominant.sel(field="phi")).plot(x="theta", marker="o")
plt.show()

np.real(dominant.sel(field="apar")).plot(marker="x")
np.imag(dominant.sel(field="apar")).plot(marker="o")
plt.show()

# Plot subdominant eigenfunction
subdominant = pyro.gk_output["eigenfunctions"].isel(mode=1)
np.real(subdominant.sel(field="phi")).plot(marker="x")
np.imag(subdominant.sel(field="phi")).plot(marker="o")
plt.show()

np.real(subdominant.sel(field="apar")).plot(marker="x")
np.imag(subdominant.sel(field="apar")).plot(marker="o")
plt.show()

# Plot growth rate and frequency
growth_rate = pyro.gk_output["growth_rate"]
growth_rate.plot()
plt.show()

mode_frequency = pyro.gk_output["mode_frequency"]
mode_frequency.plot()
plt.show()
