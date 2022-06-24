from pyrokinetics import Pyro, template_dir

# Equilibrium file
eq_file = template_dir / "test.geqdsk"

# Kinetics data file
kinetics_file = template_dir / "scene.cdf"

# Load up pyro object
pyro = Pyro(
    eq_file=eq_file,
    eq_type="GEQDSK",
    kinetics_file=kinetics_file,
    kinetics_type="SCENE",
)

# Generate local Miller parameters at psi_n=0.5
pyro.load_local_geometry(psi_n=0.5, local_geometry="Miller")

pyro.load_local_species(psi_n=0.5)

pyro.gk_code = "GS2"

pyro.write_gk_file(file_name="test_gs2.in")
