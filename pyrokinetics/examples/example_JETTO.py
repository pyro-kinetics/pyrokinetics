from pyrokinetics import Pyro, gk_templates, eq_templates, kinetics_templates

# Equilibrium file
eq_file = eq_templates["GEQDSK"]

# Kinetics data file
kinetics_file = kinetics_templates["JETTO"]

# GENE gk input file
gene_file = gk_templates["GENE"]

# Load up pyro object
pyro = Pyro(
    gk_file=gene_file,
    eq_file=eq_file,
    kinetics_file=kinetics_file,
)

# Generate local parameters at psi_n=0.5
pyro.load_local(psi_n=0.5, local_geometry="Miller")

# GS2 template input file
template_file = gk_templates["GS2"]

# Change GK code to GS2
pyro.convert_gk_code("GS2")

# Write single input file using my own template
pyro.write_gk_file(file_name="test_jetto.gs2", template_file=template_file)

# Select code as CGYRO
pyro.convert_gk_code("CGYRO")

# Write CGYRO input file using default template
pyro.write_gk_file(file_name="test_jetto.cgyro")
