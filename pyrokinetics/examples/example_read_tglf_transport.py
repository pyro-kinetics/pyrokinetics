from pyrokinetics import Pyro, template_dir
import matplotlib.pyplot as plt

gk_file = template_dir / "../../../pyro_test/TGLF_transport/input.tglf"

pyro = Pyro(gk_file=gk_file, gk_code="TGLF")

pyro.load_gk_output()

plt.ion()

# Plot fields
plt.figure(1, figsize=(16, 10), dpi=80)
plt.subplot(2, 3, 1)
field = pyro.gk_output["phi"]
field.isel(mode=0).plot(marker="x", label="mode 1")
field.isel(mode=1).plot(marker="o", label="mode 2")
plt.show(block=False)
plt.legend()
plt.title("Phi")


# plt.figure(2)
if "apar" in pyro.gk_output["field"].data:
    plt.subplot(2, 3, 2)
    field = pyro.gk_output["apar"]
    field.isel(mode=0).plot(marker="x", label="mode 1")
    field.isel(mode=1).plot(marker="o", label="mode 2")
    plt.show(block=False)
    plt.legend()
    plt.title("Apar")

# Plot fluxes
# plt.figure(3)
plt.subplot(2, 3, 3)
heat_es = pyro.gk_output["heat"].sel(field="phi")
heat_es.sel(species="electron").plot(marker="x", label="electron ES")
heat_es.sel(species="ion1").plot(marker="o", label="ion ES")
heat_em = pyro.gk_output["heat"].sel(field="apar")
heat_em.sel(species="electron").plot(marker="+", label="electron EM")
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
