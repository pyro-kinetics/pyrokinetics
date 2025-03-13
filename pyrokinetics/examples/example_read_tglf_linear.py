from pyrokinetics import Pyro
import numpy as np
import matplotlib.pyplot as plt

pyro = Pyro(gk_file="input.tglf", gk_code="TGLF")

pyro.load_gk_output()

# Plot dominant eigenfunction
try:
  dominant = pyro.gk_output.data["eigenfunctions"].isel(mode=0)

  np.real(dominant.sel(field="phi")).plot(x="theta", marker="x")
  np.imag(dominant.sel(field="phi")).plot(x="theta", marker="o")
  plt.show()

  np.real(dominant.sel(field="apar")).plot(marker="x")
  np.imag(dominant.sel(field="apar")).plot(marker="o")
  plt.show()
except:
  print('no eigenfunctions present, rerun TGLF with WRITE_WAVEFUNCTION_FLAG=T and USE_TRANSPORT_MODEL=F ??')

# Plot subdominant eigenfunction
try:
  subdominant = pyro.gk_output.data["eigenfunctions"].isel(mode=1)
  np.real(subdominant.sel(field="phi")).plot(marker="x")
  np.imag(subdominant.sel(field="phi")).plot(marker="o")
  plt.show()

  np.real(subdominant.sel(field="apar")).plot(marker="x")
  np.imag(subdominant.sel(field="apar")).plot(marker="o")
  plt.show()
except:
  print('no eignenfunctions present, rerun TGLF with WRITE_WAVEFUNCTION_FLAG=T')

# Plot growth rate and frequency
growth_rate = pyro.gk_output.data["growth_rate"]
growth_rate[:,0].plot()
growth_rate[:,1].plot()

plt.show()

mode_frequency = pyro.gk_output.data["mode_frequency"]
mode_frequency[:,0].plot()
mode_frequency[:,1].plot()
plt.show()
