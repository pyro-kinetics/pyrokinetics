import numpy as np
from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import Sauter1999, Redl2021
import matplotlib.pyplot as plt
import netCDF4 as nc

# Equilibrium and Kinetics data file
transp_cdf = template_dir / "transp.cdf"
t_interest = 0.19

data = nc.Dataset(transp_cdf)

time = data["TIME"][:]

time_index = np.argmin(np.abs(time - t_interest))

psi = data["PLFLX"][time_index, :]
transp_psi_ns = psi / psi[-1]

downsize = 1
psi_ns = transp_psi_ns[::downsize]

bs_sauter_transp = data["CURBSSAU0"][time_index, :]
bs_nclass_transp = data["CURBSWNC"][time_index, :]

jdotb_nclass = data["PLJBSNC"][time_index, :]
B2_avg = data["GB2"][time_index, :]
jdotb_sauter = bs_sauter_transp * np.sqrt(B2_avg)

# Load up pyro object
pyro = Pyro(
    eq_file=transp_cdf,
    eq_type="TRANSP",
    eq_kwargs={"time": t_interest, "neighbors": 64},
    kinetics_file=transp_cdf,
    kinetics_type="TRANSP",
    kinetics_kwargs={"time": t_interest},
)

redl_jdotb = psi_ns * 0.0
redl_bs = psi_ns * 0.0

sauter_jdotb = psi_ns * 0.0
sauter_bs = psi_ns * 0.0


redl_main_jdotb = psi_ns * 0.0
redl_main_bs = psi_ns * 0.0

sauter_main_jdotb = psi_ns * 0.0
sauter_main_bs = psi_ns * 0.0


for i, psi_n in enumerate(psi_ns[1:]):
    try:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")
    except Exception:
        continue

    pyro.load_local(psi_n=psi_n, local_geometry="MXH")

    redl = Redl2021(pyro)
    redl_jdotb[i + 1] = redl.JbsdotB.to("ampere * tesla / cm**2").m
    redl_bs[i + 1] = redl.Jbs.to("ampere / cm**2").m

    sauter = Sauter1999(pyro)
    sauter_jdotb[i + 1] = sauter.JbsdotB.to("ampere * tesla / cm**2").m
    sauter_bs[i + 1] = sauter.Jbs.to("ampere / cm**2").m

    redl_main = Redl2021(pyro, ion_type="thermal")
    redl_main_jdotb[i + 1] = redl_main.JbsdotB.to("ampere * tesla / cm**2").m
    redl_main_bs[i + 1] = redl_main.Jbs.to("ampere / cm**2").m

    sauter_main = Sauter1999(pyro, ion_type="thermal")
    sauter_main_jdotb[i + 1] = sauter_main.JbsdotB.to("ampere * tesla / cm**2").m
    sauter_main_bs[i + 1] = sauter_main.Jbs.to("ampere / cm**2").m


plt.plot(transp_psi_ns, jdotb_nclass, lw=2, label="NCLASS TRANSP")
plt.plot(transp_psi_ns, jdotb_sauter, lw=2, label="Sauter (1999) TRANSP")
plt.plot(
    psi_ns,
    redl_jdotb,
    ls="--",
    lw=2,
    color="C2",
    label="Redl (2021) Pyro Thermal + Fast ion pressure",
)
plt.plot(
    psi_ns,
    sauter_jdotb,
    ls="--",
    lw=2,
    color="C3",
    label="Sauter (1999) Pyro Thermal + Fast ion pressure",
)
plt.plot(
    psi_ns,
    redl_main_jdotb,
    ls=":",
    lw=2,
    color="C2",
    label="Redl (2021) Pyro Thermal ion pressure",
)
plt.plot(
    psi_ns,
    sauter_main_jdotb,
    ls=":",
    lw=2,
    color="C3",
    label="Sauter (1999) Pyro Thermal ion pressure",
)

plt.grid()
plt.legend()
plt.title(r"$\langle J_{bs} \cdot B\rangle$")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A T cm^{-2}$")
plt.show()

plt.plot(transp_psi_ns, bs_nclass_transp, label="NCLASS TRANSP")
plt.plot(transp_psi_ns, bs_sauter_transp, label="SAUTER (1999) TRANSP")
plt.plot(psi_ns, redl_bs, ls="--", lw=2, label="Redl (2021) Pyro")
plt.plot(psi_ns, sauter_bs, ls="--", lw=2, label="Sauter (1999) Pyro")
plt.title(r"$\frac{\langle J_{bs} \cdot B\rangle}{\langle B^2\rangle^{1/2}}$")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A cm^{-2}$")
plt.grid()
plt.legend()
plt.show()
