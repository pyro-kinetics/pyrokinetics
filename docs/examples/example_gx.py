import pyrokinetics
import tomllib

gx_template = pyrokinetics.template_dir / "input.gx"

with open(gx_template, "rb") as f:
    data = tomllib.load(f)

# Read input
pyro = pyrokinetics.Pyro(gk_file=gx_template, gk_code="GX")

# Write input file
pyro.write_gk_file(file_name="test.gx")

# Load output
pyro.load_gk_output()
