"""
Example: ExB PyroScan with consistent parallel velocity gradient (PVG)

In the high-flow regime where the toroidal velocity equals the ExB velocity
(V_tor = V_ExB), the parallel velocity is related to the ExB shear by:

    vpar = (q * R_maj / r) * V_ExB

since

    gamma_par = gamma_exb q R_maj/r

    gamma_par = - d Omega/dr R_maj

therefore

    dOmega_drho = -(q /rho)*gamma_exb

This example shows how to use ``enforce_consistent_pvg`` as a
``parameter_func`` in a :class:`PyroScan`, so that whenever ``gamma_exb`` is
updated during the scan, ``domega_drho`` is automatically kept consistent for
all species.  The approach works with all supported GK codes (GS2, CGYRO,
GENE, TGLF, STELLA, …) because the enforcement operates on the code-agnostic
:class:`LocalSpecies` object.
"""

import numpy as np

from pyrokinetics import Pyro, PyroScan, gk_templates

# ── 1. Load a Pyro object from a template input file ──────────────────────────
pyro = Pyro(gk_file=gk_templates["CGYRO"])

# ── 2. Define the scan parameter and its values ───────────────────────────────
param_key = "gamma_exb"
gamma_exb_values = (
    np.array([0.0, 0.05, 0.10, 0.15, 0.20]) * pyro.numerics.gamma_exb.units
)

param_dict = {param_key: gamma_exb_values}

# ── 3. Create the PyroScan object ─────────────────────────────────────────────
pyro_scan = PyroScan(
    pyro,
    param_dict,
    value_fmt=".3f",
    file_name="input.cgyro",
    base_directory="exb_scan",
)

# ── 4. Register ``enforce_consistent_pvg`` as the parameter function ───
# Each time ``gamma_exb`` is assigned in the scan, this function is called
# immediately afterwards and updates ``domega_drho`` for every species so that
# the PVG effect is consistent with the new ExB shear value.
pyro_scan.add_parameter_func(param_key, Pyro.enforce_consistent_pvg, {})

# ── 5. Write the input files ──────────────────────────────────────────────────
pyro_scan.write()

# ── 6. Verify that domega_drho is consistent for every run ───────────────────
print(f"{'gamma_exb':>12}  {'domega_drho (electron)':>24}  {'expected':>24}  match")
print("-" * 75)
for run_params, run_pyro in zip(
    pyro_scan.outer_product(), pyro_scan.pyro_dict.values()
):
    g_exb = run_pyro.numerics.gamma_exb
    domega = run_pyro.local_species.electron.domega_drho
    q = run_pyro.local_geometry.q
    rho = run_pyro.local_geometry.rho
    expected = -(q / rho) * g_exb
    match = np.isclose(domega.m, expected.m)
    print(f"{g_exb.m:>12.3f}  {domega.m:>24.4f}  {expected.m:>24.4f}  {match}")
