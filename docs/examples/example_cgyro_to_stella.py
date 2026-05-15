"""
Convert a CGYRO input file to different stella input file formats.

Demonstrates:
  - Reading a CGYRO template and converting to stella v1 (default)
  - Writing pre-v1 format using a template override
  - Verifying both outputs contain the same physics
"""

from pyrokinetics import Pyro, template_dir
import f90nml
from pyrokinetics.gk_code import GKInputSTELLA

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


data_v1 = f90nml.read("cgyro_to_stella_v1.in")
print(f"  v1 namelists: {list(data_v1.keys())[:6]}...")
print(f"  geometry_option = {data_v1['geometry_options']['geometry_option']}")
print(f"  beta = {data_v1['electromagnetic']['beta']}")
print()

# --- Write pre-v1 format ---
# Use a fresh Pyro object and pass the modern template when creating the context
stella_pre_v1_template = template_dir / "input.stella"
pyro_pre_v1 = Pyro(gk_file=cgyro_template, gk_code="CGYRO")
pyro_pre_v1.write_gk_file(
    file_name="cgyro_to_stella_pre_v1.in",
    gk_code="STELLA",
    template_file=stella_pre_v1_template,
)
print("Wrote: cgyro_to_stella_pre_v1.in (pre-v1 format)")

data_pre_v1 = f90nml.read("cgyro_to_stella_pre_v1.in")
print(f"  pre-v1 namelists: {list(data_pre_v1.keys())[:6]}...")
print(f"  geo_option = {data_pre_v1['geo_knobs']['geo_option']}")
print(f"  beta = {data_pre_v1['parameters_physics']['beta']}")
print()

# --- Verify both files produce the same physics ---
stella_v1 = GKInputSTELLA("cgyro_to_stella_v1.in")
stella_pre = GKInputSTELLA("cgyro_to_stella_pre_v1.in")

print("=== Format verification ===")
print(f"  v1 file detected as:      {stella_v1._format_version.value}")
print(f"  pre-v1 file detected as:   {stella_pre._format_version.value}")

geom_v1 = stella_v1.get_local_geometry()
geom_pre = stella_pre.get_local_geometry()
print(f"  v1 shat={geom_v1.shat:.4f}, pre-v1 shat={geom_pre.shat:.4f}")

species_v1 = stella_v1.get_local_species()
species_pre = stella_pre.get_local_species()
print(f"  v1 nspec={species_v1.nspec}, pre-v1 nspec={species_pre.nspec}")
