from pyrokinetics import Pyro, template_dir
import numpy as np
import matplotlib.pyplot as plt

gk_file = template_dir / "outputs/TGLF_linear/input.tglf"

gk_file = template_dir / "../../../pyro_test/TGLF_linear/input.tglf"

pyro = Pyro(gk_file=gk_file)

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
