"""
Convert a CGYRO input file to different stella input file formats.

Demonstrates:
  - Reading a CGYRO template and converting to stella v1 (default)
  - Writing stella pre-v1 ("modern") format using a template override
  - Verifying both outputs contain the same physics
"""

from pyrokinetics import Pyro, template_dir

# Read the CGYRO template
cgyro_template = template_dir / "input.cgyro"
pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")

print("=== CGYRO input ===")
print(f"  geometry: shat={pyro.local_geometry.shat:.4f}, q={pyro.local_geometry.q:.4f}")
print(f"  species:  nspec={pyro.local_species.nspec}")
print(f"  numerics: nky={pyro.numerics.nky}, ky={pyro.numerics.ky}")
print()

# --- Write stella v1 format (the new default) ---
# Pass gk_code directly so template_file is used for the new context
pyro.write_gk_file(file_name="cgyro_to_stella_v1.in", gk_code="STELLA")
print("Wrote: cgyro_to_stella_v1.in (stella v1 format)")

import f90nml

data_v1 = f90nml.read("cgyro_to_stella_v1.in")
print(f"  v1 namelists: {list(data_v1.keys())[:6]}...")
print(f"  geometry_option = {data_v1['geometry_options']['geometry_option']}")
print(f"  beta = {data_v1['electromagnetic']['beta']}")
print()

# --- Write stella pre-v1 ("modern") format ---
# Use a fresh Pyro object and pass the modern template when creating the context
stella_modern_template = template_dir / "input.stella"
pyro_modern = Pyro(gk_file=cgyro_template, gk_code="CGYRO")
pyro_modern.write_gk_file(
    file_name="cgyro_to_stella_modern.in",
    gk_code="STELLA",
    template_file=stella_modern_template,
)
print("Wrote: cgyro_to_stella_modern.in (stella pre-v1 format)")

data_mod = f90nml.read("cgyro_to_stella_modern.in")
print(f"  modern namelists: {list(data_mod.keys())[:6]}...")
print(f"  geo_option = {data_mod['geo_knobs']['geo_option']}")
print(f"  beta = {data_mod['parameters_physics']['beta']}")
print()

# --- Verify both files produce the same physics ---
from pyrokinetics.gk_code import GKInputSTELLA
from pyrokinetics.gk_code.stella import StellaFormatVersion

stella_v1 = GKInputSTELLA("cgyro_to_stella_v1.in")
stella_mod = GKInputSTELLA("cgyro_to_stella_modern.in")

print("=== Format verification ===")
print(f"  v1 file detected as:      {stella_v1._format_version.value}")
print(f"  modern file detected as:   {stella_mod._format_version.value}")

geom_v1 = stella_v1.get_local_geometry()
geom_mod = stella_mod.get_local_geometry()
print(f"  v1 shat={geom_v1.shat:.4f}, modern shat={geom_mod.shat:.4f}")

species_v1 = stella_v1.get_local_species()
species_mod = stella_mod.get_local_species()
print(f"  v1 nspec={species_v1.nspec}, modern nspec={species_mod.nspec}")
