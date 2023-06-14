import os
import matplotlib.pyplot as plt
import numpy as np

from pyrokinetics import Pyro
from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro


def compare_pyro_run(og_pyro, new_pyro, code):
    # Ensure same units
    og_pyro.gk_output.to(og_pyro.norms.pyrokinetics)
    new_pyro.gk_output.to(new_pyro.norms.pyrokinetics)

    og_data = og_pyro.gk_output.data
    new_data = new_pyro.gk_output.data

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    og_max_phi = np.max(np.abs(og_data["phi"].isel(time=-1, kx=0, ky=0)))
    new_max_phi = np.max(np.abs(new_data["phi"].isel(time=-1, kx=0, ky=0)))

    ax1.plot(
        og_data.theta,
        np.abs(og_data["phi"].isel(time=-1, kx=0, ky=0)) / og_max_phi,
        label="Original data",
    )
    ax1.plot(
        new_data.theta,
        np.abs(new_data["phi"].isel(time=-1, kx=0, ky=0)) / new_max_phi,
        ls="--",
        label="Data re-read from IDS",
    )

    ax2.plot(
        og_data.theta,
        np.abs(og_data["apar"].isel(time=-1, kx=0, ky=0)) / og_max_phi,
        label="Original data",
    )
    ax2.plot(
        new_data.theta,
        np.abs(new_data["apar"].isel(time=-1, kx=0, ky=0)) / new_max_phi,
        ls="--",
        label="Data re-read from IDS",
    )

    ax3.plot(
        og_data.theta,
        np.abs(og_data["bpar"].isel(time=-1, kx=0, ky=0)) / og_max_phi,
        label="Original data",
    )
    ax3.plot(
        new_data.theta,
        np.abs(new_data["bpar"].isel(time=-1, kx=0, ky=0)) / new_max_phi,
        ls="--",
        label="Data re-read from IDS",
    )

    fig.suptitle(
        f"{code} Difference between eigenvalues = {og_data.eigenvalues.isel(time=-1, kx=0, ky=0).data.m - new_data.eigenvalues.isel(time=-1, kx=0, ky=0).data.m}"
    )

    ax1.grid()
    ax1.set_ylabel(r"$|\phi|$")
    ax1.legend()
    ax2.grid()
    ax2.set_ylabel(r"$|A_{||}|$")

    ax3.grid()
    ax3.set_ylabel(r"$|B_{||}|$")

    ax3.set_xlabel(r"$\theta$")

    plt.show()


ref_dict = {
    "tref": 1000.0,
    "nref": 1e19,
    "lref": 1.5,
    "bref": 2.0,
}


#
# Point to CGYRO input file
cgyro_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_linear\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

if os.path.exists("cgyro_test.hdf5"):
    os.remove("cgyro_test.hdf5")

# Write IDS file
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS CGYRO",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="cgyro_test.hdf5",
)


# Read IDS file back in
new_pyro = ids_to_pyro("cgyro_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)

# Point to GENE input file
gene_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GENE_linear\parameters"
)

# Load in GENE file and output
pyro = Pyro(gk_file=gene_template)
pyro.load_gk_output()
data = pyro.gk_output


if os.path.exists("gene_test.hdf5"):
    os.remove("gene_test.hdf5")
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GENE",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="gene_test.hdf5",
)

new_pyro = ids_to_pyro("gene_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)


# Point to GS2 input file
gs2_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\GS2_linear\gs2.in"
)

# Load in file and load data
pyro = Pyro(gk_file=gs2_template)
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output


if os.path.exists("gs2_test.hdf5"):
    os.remove("gs2_test.hdf5")

ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GS2",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="gs2_test.hdf5",
)

new_pyro = ids_to_pyro("gs2_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)

"""
# Point to CGYRO input file
cgyro_nl_template = (
    r"C:\\Users\\bpatel2\OneDrive - UKAEA\Documents\pyro_test\CGYRO_run\input.cgyro"
)
# Load in file
pyro = Pyro(gk_file=cgyro_nl_template, gk_code="CGYRO")
# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output
ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS CGYRO NL",
    name="testing",
    ref_dict=ref_dict,
    format="hdf5",
    file_name="cgyro_nl_test.hdf5",
)

new_ids = ids_to_pyro("cgyro_nl_test.hdf5")
"""
