import pyrokinetics

gx_template = pyrokinetics.template_dir / "input.gx"

# Read input
pyro = pyrokinetics.Pyro(gk_file=gx_template, gk_code="GX")

# Write input file
pyro.write_gk_file(file_name="test.gx")

# Load output
pyro.load_gk_output()
