
from pathlib import Path

import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.extent import Extent
from pyrokinetics.pyroscan import PyroScan

json_path = template_dir / "outputs" / "CGYRO_linear_scan"
pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)

pyro_scan.load_gk_output()

Extent(pyro_scan.gk_output)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

np.real(pyro_scan.gk_output["phi"].isel(ky=2)).plot(ax=ax)

ax.axvspan(
    pyro_scan.gk_output["bounds"].sel(bound="lo").isel(ky=2).item(),
    pyro_scan.gk_output["bounds"].sel(bound="hi").isel(ky=2).item(),
    color="orange",
    alpha=0.3,
)