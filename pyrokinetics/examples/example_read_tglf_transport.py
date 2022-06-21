from pyrokinetics import Pyro
import matplotlib.pyplot as plt

pyro = Pyro(gk_file='input.tglf', gk_code='TGLF')

pyro.load_gk_output()

# Plot fields
fields = pyro.gk_output.data['fields']
fields.isel(nmode=0).sel(field='phi').plot(marker='x')
fields.isel(nmode=1).sel(field='phi').plot(marker='o')
plt.show()

# Plot fluxes
fluxes = pyro.gk_output.data['fluxes'].sel(moment='energy').sel(field='phi')
fluxes.sel(species='electron').plot(marker='x')
fluxes.sel(species='ion1').plot(marker='o')
plt.show()

# Plot growth rate/frequency spectrum
growth_rate = pyro.gk_output.data[r'growth_rate']
growth_rate.isel(mode=0).plot(marker='x')
growth_rate.isel(mode=1).plot(marker='o')
plt.show()

mode_frequency = pyro.gk_output.data['mode_frequency']
mode_frequency.isel(mode=0).plot(marker='x')
mode_frequency.isel(mode=1).plot(marker='o')
plt.show()
