from pyrokinetics import Pyro, eq_templates, gk_templates, kinetics_templates

# Equilibrium file
eq_file = eq_templates["GEQDSK"]

# Kinetics data file
kinetics_file = kinetics_templates["JETTO"]

# CGYRO template input file
gk_file = gk_templates["CGYRO"]

# Load up pyro object
pyro = Pyro(
    eq_file=eq_file,
    kinetics_file=kinetics_file,
    gk_file=gk_file,
)

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5)

# Change GK code to GS2
pyro.convert_gk_code("GS2")

# Dictionary for extra flags
# Nested for GS2 namelist
flags = {
    "gs2_diagnostics_knobs": {
        "write_fields": True,
        "write_kpar": True,
    },
}

pyro.add_flags(flags)

# Write single input file using my own template
pyro.write_gk_file(file_name="test_jetto.gs2")
