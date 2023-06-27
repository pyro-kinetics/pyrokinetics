from pyrokinetics import Pyro
import matplotlib.pyplot as plt

pyro = Pyro(gk_file="input.tglf", gk_code="TGLF")

pyro.load_gk_output()

plt.ion()

# Plot fields
plt.figure(1, figsize=(16, 10), dpi=80)
plt.subplot(2, 3, 1)
fields = pyro.gk_output["fields"]
fields.isel(mode=0).sel(field="phi").plot(marker="x", label="mode 1")
fields.isel(mode=1).sel(field="phi").plot(marker="o", label="mode 2")
plt.show(block=False)
plt.legend()
plt.title("Phi")


# plt.figure(2)
if "apar" in fields.field:
    plt.subplot(2, 3, 2)
    fields.isel(mode=0).sel(field="apar").plot(marker="x", label="mode 1")
    fields.isel(mode=1).sel(field="apar").plot(marker="o", label="mode 2")
    plt.show(block=False)
    plt.legend()
    plt.title("Apar")

# Plot fluxes
# plt.figure(3)
plt.subplot(2, 3, 3)
fluxes = pyro.gk_output["fluxes"].sel(moment="energy").sel(field="phi")
fluxes.sel(species="electron").plot(marker="x", label="electron ES")
fluxes.sel(species="ion1").plot(marker="o", label="ion ES")
fluxes = pyro.gk_output["fluxes"].sel(moment="energy").sel(field="apar")
fluxes.sel(species="electron").plot(marker="+", label="electron EM")
plt.show(block=False)
plt.legend()
plt.title("Fluxes")

# Plot growth rate/frequency spectrum
# plt.figure(4)
plt.subplot(2, 3, 4)
growth_rate = pyro.gk_output[r"growth_rate"]
growth_rate.isel(mode=0).plot(marker="x", label="mode 1")
growth_rate.isel(mode=1).plot(marker="o", label="mode 2")
plt.show(block=False)
plt.legend()
plt.title("Eigenvalue Growth.")

# plt.figure(5)
plt.subplot(2, 3, 5)
mode_frequency = pyro.gk_output["mode_frequency"]
mode_frequency.isel(mode=0).plot(marker="x", label="mode 1")
mode_frequency.isel(mode=1).plot(marker="o", label="mode 2")
plt.show(block=True)
plt.legend()
plt.title("Eigenvalue Freq.")
