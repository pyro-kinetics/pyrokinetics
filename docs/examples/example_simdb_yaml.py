import os
import sys
from pathlib import Path

from pyrokinetics import Pyro, template_dir
from pyrokinetics.databases import pyro_to_ids
from pyrokinetics.databases.yaml import SimDBYaml

gk_file = sys.argv[1]

path = Path(gk_file).parent.name
ids_file_name = f"gyrokinetics_local_final_{path}.h5"

# Load data
pyro = Pyro(gk_file=gk_file)
pyro.load_gk_output()

# Initialise yaml file first
manifest = SimDBYaml(template_dir / "template.yaml", pyro)

# Select final time slice
pyro.gk_output.data = pyro.gk_output.data.isel(time=-1)

if os.path.exists(ids_file_name):
    os.remove(ids_file_name)

# Write IDS file
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GS2",
    reference_values={},
    format="hdf5",
    file_name=ids_file_name,
)

manifest.set_alias("bpatel2/set/example/alias")

ids_file_path = Path(ids_file_name).resolve()
manifest.add_output(ids_file_path)

# Add another code
manifest.add_code(
    name="newcode",
    commit="abcdef123",
    repo="https://gitlab.com/example/newcode"
)

# Overwrite workflow metadata if needed
manifest.add_code(name="gs3", commit="999999", repo="https://gitlab.com/new/gs3")


# Add description
meta_data = manifest.get_metadata()
text = "Here I am writing a description of the GK simulation performed and I am writing a lot of text to see how the formatting looks in the final YAML file"
manifest.add_description(text)

# Print manifest before writing
print(manifest)
# Save to file to be parsed by SimDB
manifest.write("manifest_modified.yaml")
