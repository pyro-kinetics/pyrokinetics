"""
example_pyro.py

Demonstrates the creation of a Pyro object, and the various ways in which a GK input
file may be manipulated.
"""

from pyrokinetics import Pyro, gk_templates, eq_templates, kinetics_templates

gs2_file = gk_templates["GS2"]
eq_file = eq_templates["GEQDSK"]
kinetics_file = kinetics_templates["SCENE"]


# Read a GK input file (with automatic file type inference
pyro = Pyro(gk_input_file=gs2_file)

# Can then access Pyrofied data:
ntheta = pyro.numerics["ntheta"]
shat = pyro.local_geometry["shat"]
ion_density = pyro.local_species["ion1"]["dens"]
assert "deuterium" not in pyro.local_species

# We can replace the local_geometry with one generated from a global equilibrium
psi_n = 0.5
pyro.load_global_eq(eq_file)
pyro.load_local_geometry(psi_n)
# Show that local_geometry has been updated:
assert pyro.local_geometry["psi_n"] == 0.5
assert pyro.local_geometry["shat"] != shat

# We can also replace local_species with one generated from global kinetics
# If we were to do this without setting from a global equilibrium, we would also
# need to set a_minor. We'll explicitly set it to None to demonstrate usage.
pyro.load_global_kinetics(kinetics_file)
pyro.load_local_species(psi_n, a_minor=None)
# Show that local_species has been updated:
assert "deuterium" in pyro.local_species
assert "ion1" not in pyro.local_species

# We may also read multiple files in with the constructor
pyro = Pyro(
    gk_input_file=gs2_file,
    eq_file=eq_file,
    kinetics_file=kinetics_file,
)
# And we may set local_species and local_geometry together
pyro.load_local(psi_n=psi_n)

# We may then write out a new GS2 input file given this new info
pyro.write_gk_file("modified_gs2.in", "GS2")
# Or we can convert to a different gyrokinetics code
pyro.write_gk_file("modified_gs2_to_cgyro.in", "CGYRO")
