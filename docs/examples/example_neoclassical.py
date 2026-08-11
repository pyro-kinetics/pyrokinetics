import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import (
    Redl2021,
    Sauter1999,
    integrate_toroidal_current,
)

# Equilibrium and Kinetics data file
transp_cdf = template_dir / "transp.cdf"
t_interest = 0.19

data = nc.Dataset(transp_cdf)

time = data["TIME"][:]

time_index = np.argmin(np.abs(time - t_interest))

psi = data["PLFLX"][time_index, :]
transp_psi_ns = psi / psi[-1]

# Take every downsize'th flux surface. The cost of this example is dominated by
# the MXH fit in load_local (~1.6 s per surface), not by the current
# calculation (~0.1 s), so thinning the radial grid is what makes it faster.
downsize = 2
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

# Toroidal current density and its decomposition, in both conventions:
# <J_phi> is the plain flux surface average, J_phi^eff is the effective density
# (1 / 2 pi rho) dIp/drho. They differ by roughly the elongation.
jphi_total = psi_ns * 0.0
jphi_bs = psi_ns * 0.0
jphi_ext = psi_ns * 0.0
jphi_psdia = psi_ns * 0.0

jphi_eff_total = psi_ns * 0.0
jphi_eff_bs = psi_ns * 0.0
jphi_eff_ext = psi_ns * 0.0
jphi_eff_psdia = psi_ns * 0.0

# Print every current component on each surface. The external contribution is
# auxiliary + ohmic combined, since a local calculation cannot separate them.
print("\nCurrent components on each flux surface (Redl 2021)\n")
print(
    f"{'psi_n':>6} | {'<J.B>':>10} {'<Jbs.B>':>10} {'<Jext.B>':>10} |"
    f" {'<Jphi>':>9} {'bs':>9} {'ext':>9} {'ps+dia':>9} |"
    f" {'Jphi_eff':>9} | {'Ip':>7}"
)
print(
    f"{'':>6} | {'A T/cm2':>10} {'A T/cm2':>10} {'A T/cm2':>10} |"
    f" {'A/cm2':>9} {'A/cm2':>9} {'A/cm2':>9} {'A/cm2':>9} |"
    f" {'A/cm2':>9} | {'MA':>7}"
)
print("-" * 104)

for i, psi_n in enumerate(psi_ns[1:]):
    try:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")
    except Exception:
        continue

    redl = Redl2021(pyro)
    redl_jdotb[i + 1] = redl.JbsdotB.to("ampere * tesla / cm**2").m
    redl_bs[i + 1] = redl.Jbs.to("ampere / cm**2").m

    jphi_total[i + 1] = redl.Jphi_fsa.to("ampere / cm**2").m
    jphi_bs[i + 1] = redl.Jphi_bs_fsa.to("ampere / cm**2").m
    jphi_ext[i + 1] = redl.Jphi_ext_fsa.to("ampere / cm**2").m
    jphi_psdia[i + 1] = redl.Jphi_psdia_fsa.to("ampere / cm**2").m

    jphi_eff_total[i + 1] = redl.Jphi_eff.to("ampere / cm**2").m
    jphi_eff_bs[i + 1] = redl.Jphi_bs_eff.to("ampere / cm**2").m
    jphi_eff_ext[i + 1] = redl.Jphi_ext_eff.to("ampere / cm**2").m
    jphi_eff_psdia[i + 1] = redl.Jphi_psdia_eff.to("ampere / cm**2").m

    parallel_units = "ampere * tesla / cm**2"
    print(
        f"{psi_n:6.3f} |"
        f" {redl.JdotB.to(parallel_units).m:10.4g}"
        f" {redl.JbsdotB.to(parallel_units).m:10.4g}"
        f" {redl.JextdotB.to(parallel_units).m:10.4g} |"
        f" {jphi_total[i + 1]:9.4g}"
        f" {jphi_bs[i + 1]:9.4g}"
        f" {jphi_ext[i + 1]:9.4g}"
        f" {jphi_psdia[i + 1]:9.4g} |"
        f" {jphi_eff_total[i + 1]:9.4g} |"
        f" {redl.Ip.to('MA').m:7.3f}"
    )

    sauter = Sauter1999(pyro)
    sauter_jdotb[i + 1] = sauter.JbsdotB.to("ampere * tesla / cm**2").m
    sauter_bs[i + 1] = sauter.Jbs.to("ampere / cm**2").m

# Radially integrate each component to get the total current it carries. This
# repeats the flux surface scan, so it is the slow part of this example.
#
# Restrict the scan to surfaces where the model is trustworthy:
#  - near the axis the inverse aspect ratio tends to zero and the trapped
#    fraction degenerates (on this equilibrium it goes negative at psi_n = 0.0009,
#    which makes the bootstrap current NaN)
#  - near the separatrix the MXH fit degrades, and the last rows of the table
#    above flip sign and blow up
# Either would quietly corrupt the integral, so exclude both.
psi_n_min, psi_n_max = 0.01, 0.95
psi_n_all = np.asarray(psi_ns)
psi_n_scan = psi_n_all[(psi_n_all > psi_n_min) & (psi_n_all < psi_n_max)]
integrated = integrate_toroidal_current(pyro, psi_n_scan, local_geometry="MXH")

total = integrated["Ip"][-1].to("MA")
rho_edge = integrated["rho"][-1].to("m")
print(
    f"\nTotal current carried by each component, integrated over "
    f"psi_n = {psi_n_min} to {psi_n_max}\n"
)
for label, key in [
    ("bootstrap", "Ip_bs"),
    ("external (auxiliary + ohmic)", "Ip_ext"),
    ("Pfirsch-Schluter + diamagnetic", "Ip_psdia"),
]:
    current = integrated[key][-1].to("MA")
    print(f"  {label:<32} {current.m:8.4f} MA  ({100 * current.m / total.m:5.1f} %)")

print(f"  {'total':<32} {total.m:8.4f} MA")
print(
    f"  {'Ampere law, same surface':<32} "
    f"{integrated['Ip_ampere'][-1].to('MA').m:8.4f} MA   (independent check)"
)
print(f"\n  outermost surface integrated: rho = {rho_edge.m:.3f} m")


plt.plot(transp_psi_ns, jdotb_nclass, lw=2, label="NCLASS TRANSP")
plt.plot(transp_psi_ns, jdotb_sauter, lw=2, label="Sauter (1999) TRANSP")
plt.plot(
    psi_ns,
    redl_jdotb,
    ls="--",
    lw=2,
    color="C2",
    label="Redl (2021) Pyro",
)
plt.plot(
    psi_ns,
    sauter_jdotb,
    ls="--",
    lw=2,
    color="C3",
    label="Sauter (1999) Pyro",
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

# Decomposition of the toroidal current density. Note the external contribution
# is auxiliary + ohmic combined, as a local calculation cannot separate them.
plt.plot(psi_ns, jphi_total, lw=2, color="k", label="Total")
plt.plot(psi_ns, jphi_bs, ls="--", lw=2, label="Bootstrap")
plt.plot(psi_ns, jphi_ext, ls="--", lw=2, label="External (auxiliary + ohmic)")
plt.plot(psi_ns, jphi_psdia, ls="--", lw=2, label="Pfirsch-Schluter + diamagnetic")
plt.plot(
    psi_ns,
    jphi_bs + jphi_ext + jphi_psdia,
    ls=":",
    lw=2,
    color="C3",
    label="Sum of components",
)
plt.title(r"$\langle J_\phi \rangle$ (Redl 2021)")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A cm^{-2}$")
plt.grid()
plt.legend()
plt.show()

# The same decomposition for the effective toroidal current density,
# J_phi^eff = (1 / 2 pi rho) dIp/drho. This is the definition whose 2 pi rho
# weighted radial integral gives the enclosed plasma current, so these are the
# components that integrate to the totals printed above.
plt.plot(psi_ns, jphi_eff_total, lw=2, color="k", label="Total")
plt.plot(psi_ns, jphi_eff_bs, ls="--", lw=2, label="Bootstrap")
plt.plot(psi_ns, jphi_eff_ext, ls="--", lw=2, label="External (auxiliary + ohmic)")
plt.plot(psi_ns, jphi_eff_psdia, ls="--", lw=2, label="Pfirsch-Schluter + diamagnetic")
plt.plot(
    psi_ns,
    jphi_eff_bs + jphi_eff_ext + jphi_eff_psdia,
    ls=":",
    lw=2,
    color="C3",
    label="Sum of components",
)
plt.title(r"$J_\phi^{eff} = \frac{1}{2\pi\rho}\frac{dI_p}{d\rho}$ (Redl 2021)")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A cm^{-2}$")
plt.grid()
plt.legend()
plt.show()

# Both conventions side by side, showing they differ by roughly the elongation
plt.plot(psi_ns, jphi_total, lw=2, color="k", label=r"$\langle J_\phi \rangle$")
plt.plot(psi_ns, jphi_eff_total, ls="--", lw=2, color="C0", label=r"$J_\phi^{eff}$")
plt.title(r"Toroidal current density: the two conventions")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A cm^{-2}$")
plt.grid()
plt.legend()
plt.show()
