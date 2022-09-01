from pyrokinetics import Pyro, template_dir

# Equilibrium file
eq_file = template_dir / "test.geqdsk"
#eq_file = "spr45_600x900.eqdsk"
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
pyro.load_local_geometry(psi_n=0.5, local_geometry="Fourier", show_fit=True)

pyro.switch_local_geometry(local_geometry="Miller", show_fit=True)

pyro.switch_local_geometry(local_geometry="MXH", show_fit=True)

pyro.switch_local_geometry(local_geometry="FourierCGYRO", show_fit=True)

pyro.load_local_species(psi_n=0.5)

pyro.gk_code = "GS2"

pyro.write_gk_file(file_name="test_gs2.in")
