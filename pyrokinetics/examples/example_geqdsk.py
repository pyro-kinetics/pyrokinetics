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

# Show fit when loading in data
show_fit = True

# Generate local Miller parameters at psi_n=0.5
pyro.load_local_geometry(psi_n=0.5, local_geometry="Miller", show_fit=show_fit)

# Switch to different geometry types and plot fit
pyro.switch_local_geometry(local_geometry="FourierGENE", show_fit=show_fit)

pyro.switch_local_geometry(local_geometry="FourierCGYRO", show_fit=show_fit)

pyro.switch_local_geometry(local_geometry="MXH", show_fit=show_fit)

