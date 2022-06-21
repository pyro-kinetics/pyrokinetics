from pyrokinetics import Pyro, template_dir, gk_templates

# Equilibrium file
eq_file = template_dir / "test.geqdsk"

# Kinetics data file
kinetics_file = template_dir / "transp.cdf"

# Load up pyro object
pyro = Pyro(
    gk_file = gk_templates["GS2"],
    eq_file=eq_file,
    kinetics_file=kinetics_file,
)

# Generate local Miller parameters at psi_n=0.5
pyro.load_local_geometry(psi_n=0.5)
pyro.load_local_species(psi_n=0.5)


pyro.write_gk_file(file_name="test_transp.in")
