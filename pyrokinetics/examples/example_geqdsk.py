from pyrokinetics import Pyro, template_dir

# Equilibrium file
eq_file = template_dir / "test.geqdsk"

# Kinetics data file
kinetics_file = template_dir / "scene.cdf"

# GK file
gk_file = template_dir / "input.gs2"

# Load up pyro object
pyro = Pyro(
    gk_file=gk_file,
    eq_file=eq_file,
    kinetics_file=kinetics_file,
)

# Generate local Miller parameters at psi_n=0.5
pyro.load_local_geometry(psi_n=0.5, local_geometry="Miller")

pyro.load_local_species(psi_n=0.5)

pyro.write_gk_file(file_name="test_gs2.in")
