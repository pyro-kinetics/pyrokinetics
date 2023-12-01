"""
Run this using `OMP_NUM_THREADS=1 python example_ideal_ballooning.py`, as some
internal routines are multithreaded, and leaving it as the default will lead
to inefficient use of resources.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics

nshat = 15
nbprime = 20
shat = np.linspace(0.0, 2, nshat)
bprime = np.linspace(0.0, -0.5, nbprime)

pyro = Pyro(gk_file=template_dir / "input.gs2")

gamma = np.empty((nshat, nbprime))


def _fn(args):
    s, b = args
    pyro.local_geometry.shat = s
    pyro.local_geometry.beta_prime = b
    diag = Diagnostics(pyro)
    return diag.ideal_ballooning_solver()


gamma = np.asarray(process_map(_fn, product(shat, bprime))).reshape(nshat, nbprime)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 7))

cs = ax.contourf(abs(bprime), shat, gamma)
cs_0 = ax.contour(
    abs(bprime),
    shat,
    gamma,
    colors="w",
    linewidths=(2,),
    levels=[
        0.0,
    ],
)

ax.clabel(cs_0, fmt="%.1e", colors="w", fontsize=22)
ax.set_ylabel(r"$\hat{s}$")
ax.set_xlabel(r"$|\beta'|$")
fig.colorbar(cs)
ax.set_title("IBM growth rate")
fig.tight_layout()
plt.savefig("ideal_ballooning.png")
plt.show()
