from pyrokinetics import Pyro, template_dir
import numpy as np

# Set up Pyro object

eq_file = template_dir / "test.geqdsk"
kinetics_file = template_dir / "jetto.jsp"

pyro = Pyro(
    eq_file=eq_file,
    eq_kwargs={"psi_n_lcfs": 0.9999},
    kinetics_file=kinetics_file,
)

# Add a merge species to the Kinetics object one day to make the loop more general
# pyro.kinetics.merge_species("deuterium", ["all", "other", "ions"]

# Need data on rho_tor grid to map from psi_n to rho_tor
n_rho_tor = 1001
pyro_psi_n = np.linspace(0.0001, 1.0, n_rho_tor * 10)
pyro_rho_tor = pyro.eq.rho_tor(pyro_psi_n)

rho_tor = np.linspace(0, 1.0, n_rho_tor)
psi_n = np.interp(rho_tor, pyro_rho_tor, pyro_psi_n)

# Different species data
species_data = pyro.kinetics.species_data
species = species_data.keys()
n_species = len(species)


# Reference values at each surface
nref = species_data.electron.get_dens(psi_n)
tref = species_data.electron.get_temp(psi_n)
mref = species_data.deuterium.get_mass()
qref = abs(species_data.electron.get_charge(psi_n))
bref = pyro.eq.F(psi_n) / pyro.eq.R_major(psi_n)
lref = pyro.eq.a_minor

# Larmor radius
rho_ref = (np.sqrt(tref * mref) / (qref * bref)).to("meter")

# GENE defn of x
x = lref * rho_tor

# Iterate through all species
for name, species in species_data.items():
    global_data = np.empty((4, n_rho_tor))

    global_data[0, :] = (x / lref).m
    global_data[1, :] = (x / rho_ref).m
    global_data[2, :] = (species.get_temp(psi_n).to("keV")).m
    global_data[3, :] = (species.get_dens(psi_n).to("meter**-3") * 1e-19).m
    np.savetxt(
        f"{name}.dat",
        global_data.T,
        fmt="%.6e",
        header="x/a        x/rho_ref    T            n           ",
    )

# Deuterium merged - electron density but ion temp - good for 2 species global sims
# Ideally have a merged species function
name = "deuterium"
global_data = np.empty((4, n_rho_tor))
global_data[0, :] = (x / lref).m
global_data[1, :] = (x / rho_ref).m
global_data[2, :] = (species_data.deuterium.get_temp(psi_n).to("keV")).m
global_data[3, :] = (species_data.electron.get_dens(psi_n).to("meter**-3") * 1e-19).m
np.savetxt(
    f"{name}_merged.dat",
    global_data.T,
    fmt="%.6e",
    header="x/a        x/rho_ref    T            n           ",
)
