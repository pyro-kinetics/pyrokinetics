"""
example_pyro_alt.py

Demonstrates the creation of a Pyro object, and the various ways in which a GK input
file may be manipulated.
"""
# FIXME Rename this file

from pyrokinetics import PyroAlt as Pyro, template_dir

gs2_file = template_dir / "input.gs2"
eq_file = template_dir / "transp_eq.geqdsk"
kinetics_file = template_dir / "scene.cdf"


# Read a GK input file (with automatic file type inference
pyro = Pyro(gk_input_file=gs2_file)

# Can then access Pyrofied data:
ntheta = pyro.numerics["ntheta"]
shat = pyro.local_geometry["shat"]
ion_density = pyro.local_species["ion1"]["dens"]
assert "deuterium" not in pyro.local_species

# We can replace the local_geometry with one generated from a global equilibrium
psi_n = 0.5
pyro.read_global_eq(eq_file)
pyro.set_local_geometry_from_global_eq(psi_n)
# Or, by providing psi_n at the read step, we can do this in one line
pyro.read_global_eq(eq_file, psi_n=psi_n)
# Show that local_geometry has been updated:
assert pyro.local_geometry["psi_n"] == 0.5
assert pyro.local_geometry["shat"] != shat

# We can also replace local_species with one generated from global kinetics
# If we were to do this without setting from a global equilibrium, we would also
# need to set a_minor. We'll explicitly set it to None to demonstrate usage.
pyro.read_global_kinetics(kinetics_file)
pyro.set_local_species_from_global_kinetics(psi_n, a_minor=None)
# Alternatively, do it in one line:
pyro.read_global_kinetics(kinetics_file, psi_n=psi_n, a_minor=None)
# Show that local_species has been updated:
assert "deuterium" in pyro.local_species
assert "ion1" not in pyro.local_species

# To simplify all of that, we can do it all in the constructor to Pyro
pyro = Pyro(
    gk_input_file=gs2_file,
    eq_file=eq_file,
    kinetics_file=kinetics_file,
    psi_n=psi_n,
)

# We may then write out a new GS2 input file given this new info
pyro.write_gk_file("modified_gs2.in", "GS2")
