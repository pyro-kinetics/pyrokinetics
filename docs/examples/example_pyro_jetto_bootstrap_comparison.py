import os

import matplotlib.pyplot as plt
import numpy as np
from jetto_tools.binary import read_binary_file

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.neoclassical import Redl2021, Sauter1999

# Equilibrium and Kinetics data file
jetto_file = template_dir / "jetto.jsp"
jetto_data = read_binary_file(jetto_file)

jetto_jss_file = template_dir / "jetto.jss"
jetto_jss = read_binary_file(jetto_jss_file)


# NOTE this may not actually match the JETTO JSP file
jetto_eqdsk = template_dir / "test.geqdsk"

jetto_time = jetto_data["TIME"]
jetto_t_index = -1

jetto_ftrap = jetto_data["FT"][jetto_t_index, :]
jetto_psi = jetto_data["PSI"][jetto_t_index, :]

jetto_psi = jetto_psi - jetto_psi[0]
jetto_psin = jetto_psi / jetto_psi[-1]

x = jetto_data["XVEC1"][jetto_t_index, :]
xb = jetto_data["XVEC2"][jetto_t_index, :]
xb = np.hstack(([0], xb, [1.0]))

jetto_jbsdotb_b0_xb = np.abs(jetto_data["JPBS"][jetto_t_index, :])
jetto_jbsdotb_b0_xb = np.hstack(([0.0], jetto_jbsdotb_b0_xb, [0.0]))
jetto_jbsdotb_b0 = np.interp(x, xb, jetto_jbsdotb_b0_xb)

jetto_curbs = jetto_data["JZBS"][jetto_t_index, :]

# Load up pyro object
pyro = Pyro(
    eq_file=jetto_eqdsk,
    eq_type="GEQDSK",
    kinetics_file=jetto_file,
    kinetics_type="JETTO",
    kinetics_kwargs={"time_index": -1},
)

downsize = 5
psi_ns = jetto_psin[::downsize]

jbsdotb_units = pyro.norms.units("ampere tesla / m**2")
bs_units = pyro.norms.units("ampere / m**2")

redl_jbsdotb = psi_ns * 0.0 * jbsdotb_units
redl_bs = psi_ns * 0.0 * bs_units

sauter_jbsdotb = psi_ns * 0.0 * jbsdotb_units
sauter_bs = psi_ns * 0.0 * bs_units

try:
    B0 = np.abs(jetto_jss["B0"][jetto_t_index]) * pyro.norms.units.tesla
except KeyError:
    print("B0 not in jss file, setting to 3 Tesla")
    B0 = 3.0 * pyro.norms.units.tesla

neo_ftrap = psi_ns * 0.0
pyro_ftrap = psi_ns * 0.0

pyro_jetto_formula = psi_ns * 0.0
pyro_sauter_formula = psi_ns * 0.0
pyro_bp_formula = psi_ns * 0.0


neo_exec = False
run_neo = False
if neo_exec:
    neo_jbsdotb = psi_ns * 0.0 * jbsdotb_units


def read_jbs(filename):
    """
    Read <Jbs/B> / Bunit from a text file header and return it as a float.
    """
    try:
        data = np.loadtxt(filename)
    except FileNotFoundError:
        print("No NEO run found, setting jbs = 0.0")
        return 0.0
    try:
        return data[2]
    except IndexError:
        return 0.0


def read_f_trap(filename):
    """
    Read f_trap from a text file header and return it as a float.
    """
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("#"):
                # Stop once header ends
                break

            if "f_trap" in line:
                # Split on '=' and convert to float
                return float(line.split("=")[1].strip())

    raise ValueError("f_trap not found in file header")


for i, psi_n in enumerate(psi_ns[1:]):
    print(f"{i} out of {len(psi_ns)}")
    try:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")
    except Exception:
        continue

    show_fit = False
    pyro.load_local(psi_n=psi_n, local_geometry="MXH", show_fit=show_fit)
    merge_species = [
        name for name in pyro.local_species.names if "fast" in name if "alpha" in name
    ]
    pyro.local_species.merge_species(
        "deuterium",
        merge_species,
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    redl = Redl2021(pyro)
    redl_jbsdotb[i + 1] = redl.JbsdotB.to("ampere * tesla / m**2")
    redl_bs[i + 1] = redl.Jbs.to("ampere / m**2")
    pyro_ftrap[i + 1] = redl.trapped_fraction

    sauter = Sauter1999(pyro)
    sauter_jbsdotb[i + 1] = sauter.JbsdotB.to("ampere * tesla / m**2")
    sauter_bs[i + 1] = sauter.Jbs.to("ampere / m**2")

    epsilon = pyro.local_geometry.rho / pyro.local_geometry.Rmaj
    delta = pyro.local_geometry.delta
    eps_eff = 0.67 * (1 - 1.4 * delta * np.abs(delta)) * epsilon

    eps_eff_bp = 0.4 * (1 - 1.4 * delta * np.abs(delta)) * epsilon

    pyro_jetto_formula[i + 1] = 1 - (1 - epsilon) ** 2 / (
        (1 - epsilon**2) ** 0.5 * (1 + 1.46 * epsilon**0.5)
    )
    pyro_sauter_formula[i + 1] = 1 - np.sqrt((1 - epsilon) / (1 + epsilon)) * (
        1 - eps_eff
    ) / (1 + 2 * np.sqrt(eps_eff))
    pyro_bp_formula[i + 1] = 1 - np.sqrt((1 - epsilon) / (1 + epsilon)) * (
        1 - eps_eff_bp
    ) / (1 + 2 * np.sqrt(eps_eff_bp))

    if neo_exec:
        pyro.gk_code = "NEO"
        pyro.numerics.npitch = 69
        pyro.numerics.nenergy = 6
        pyro.numerics.ntheta = 33
        i_down = int((i + 1) * downsize / 2) - 1
        if i_down == 1:
            i_down = 0

        if downsize == 2:
            i_down = i
        pyro.write_gk_file(f"psi_n_{i_down}/input.neo", gk_code="NEO")

        if run_neo:
            test = read_jbs(f"psi_n_{i_down}/out.neo.transport")
            if not os.path.exists(f"psi_n_{i_down}/out.neo.transport") or test == 0.0:
                print(f"neo -n 1 -nomp 1 -e psi_n_{i_down}/")
                os.system(f"neo -n 1 -nomp 1 -e psi_n_{i_down}/")

        neo_ftrap[i + 1] = read_f_trap(f"psi_n_{i_down}/out.neo.diagnostic_geo")

        units = pyro.norms.cgyro
        jbs_units = units.qref * units.nref * units.vref * units.bref
        jbs_b0 = read_jbs(f"psi_n_{i_down}/out.neo.transport") * jbs_units

        neo_jbsdotb[i + 1] = np.abs((jbs_b0).to("ampere tesla / meter**2"))

    if neo_exec:
        pyro.gk_code = "NEO"
        i_down = int((i + 1) * downsize / 2) - 1
        if i_down == 1:
            i_down = 0
        if downsize == 2:
            i_down = i
        pyro.write_gk_file(f"psi_n_{i_down}/input.neo", gk_code="NEO")
        if run_neo:
            os.system(f"neo -n 32 -e psi_n_{i_down}")

        neo_ftrap[i + 1] = read_f_trap(f"psi_n_{i_down}/out.neo.diagnostic_geo")


plt.plot(jetto_psin[1:], jetto_ftrap, lw=2, ls="-.", color="C0", label="JETTO")
if neo_exec:
    plt.plot(psi_ns, neo_ftrap, lw=2, ls="--", color="C1", label="NEO")
plt.plot(psi_ns, pyro_ftrap, lw=2, color="C2", label="Pyro")
plt.plot(psi_ns, pyro_sauter_formula, lw=2, ls=":", label="Sauter new")
plt.plot(psi_ns, pyro_bp_formula, lw=2, ls=":", label="Sauter + BP mod")
plt.grid()
plt.legend()
plt.title(r"Trapped particle fraction")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$f_{trap}$")
plt.show()


redl_jbsdotb_b0 = (redl_jbsdotb / B0).to("ampere / m**2")
sauter_jbsdotb_b0 = (sauter_jbsdotb / B0).to("ampere / m**2")

if neo_exec:
    neo_jbsdotb_b0 = (neo_jbsdotb / B0).to("ampere / m**2")

plt.plot(jetto_psin, jetto_jbsdotb_b0, ls="-", lw=2, label="JETTO")
if neo_exec:
    plt.plot(psi_ns, neo_jbsdotb_b0.m, ls="--", lw=2, color="C1", label="NEO")
plt.plot(psi_ns, redl_jbsdotb_b0.m, ls="--", lw=2, color="C2", label="Redl (2021) Pyro")
plt.plot(
    psi_ns, sauter_jbsdotb_b0.m, ls="--", lw=2, color="C3", label="Sauter (1999) Pyro"
)
plt.grid()
plt.legend()
plt.title(r"$\langle J_{bs} \cdot B\rangle / B_0$")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A m^{-2}$")
plt.show()

plt.plot(jetto_psin, jetto_curbs, ls="--", lw=2, label="JETTO")
plt.plot(psi_ns, redl_bs, ls="--", lw=2, label="Redl (2021) Pyro")
plt.plot(psi_ns, sauter_bs, ls="--", lw=2, label="Sauter (1999) Pyro")
plt.title(r"$\frac{\langle J_{bs} \cdot B\rangle}{\langle B^2\rangle^{1/2}}$")
plt.xlabel(r"$\psi_N$")
plt.ylabel(r"$A cm^{-2}$")
plt.grid()
plt.legend()
plt.show()
