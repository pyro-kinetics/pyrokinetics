import os
import matplotlib.pyplot as plt
import numpy as np

from pyrokinetics import Pyro, template_dir
from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro
from pyrokinetics.normalisation import ureg as units


def compare_pyro_run(og_pyro, new_pyro, code):
    # Ensure same units
    og_pyro.gk_output.to(og_pyro.norms.pyrokinetics)
    new_pyro.gk_output.to(new_pyro.norms.pyrokinetics)

    og_data = og_pyro.gk_output.data
    new_data = new_pyro.gk_output.data

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))

    ax1.plot(
        og_data.theta,
        np.abs(og_data["phi"].isel(time=-1, kx=0, ky=0)),
        label="Original data",
    )
    ax1.plot(
        new_data.theta,
        np.abs(new_data["phi"].isel(time=-1, kx=0, ky=0)),
        ls="--",
        label="Data re-read from IDS",
    )

    eigval_difference = (
        og_data.eigenvalues.isel(time=-1, kx=0, ky=0).data.m
        - new_data.eigenvalues.isel(time=-1, kx=0, ky=0).data.m
    )

    fig.suptitle(f"{code} Difference between eigenvalues = {eigval_difference}")

    ax1.grid()
    ax1.set_ylabel(r"$|\phi|$")
    ax1.legend()
    ax1.set_xlabel(r"$\theta$")

    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))

    if "ky" in og_data["heat"].dims:
        og_data["heat"] = og_data["heat"].sum(dim="ky")
        new_data["heat"] = new_data["heat"].sum(dim="ky")

    ax1.plot(
        og_data.time,
        np.abs(og_data["heat"].isel(field=0).sel(species="electron")),
        label="Original data",
    )
    ax1.plot(
        new_data.time,
        np.abs(new_data["heat"].isel(field=0).sel(species="electron")),
        ls="--",
        label="Data re-read from IDS",
    )

    ax1.grid()
    ax1.set_ylabel(r"$Q$")
    ax1.legend()
    ax1.set_xlabel(r"$t (a/c_s)$")

    plt.show()


def compare_pyro_run_nl(og_pyro, new_pyro, code):
    # Ensure same units
    og_pyro.gk_output.to(og_pyro.norms.pyrokinetics)
    new_pyro.gk_output.to(new_pyro.norms.pyrokinetics)

    og_data = og_pyro.gk_output.data
    new_data = new_pyro.gk_output.data

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))

    ax1.plot(
        og_data.theta,
        np.abs(og_data["phi"].isel(time=-1, kx=128, ky=1)),
        label="Original data",
    )
    ax1.plot(
        new_data.theta,
        np.abs(new_data["phi"].isel(time=-1, kx=128, ky=1)),
        ls="--",
        label="Data re-read from IDS",
    )

    ax1.grid()
    ax1.set_ylabel(r"$|\phi|$")
    ax1.legend()
    ax1.set_xlabel(r"$\theta$")

    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))

    # Can only compare time averaged fluxes in IMAS
    ax1.plot(
        og_data.ky,
        np.abs(og_data["heat"].isel(field=0).sel(species="electron").mean(dim="time")),
        label="Time average original data",
    )
    ax1.plot(
        new_data.ky,
        np.abs(new_data["heat"].isel(time=-1, field=0).sel(species="electron")),
        label="IDS final",
        ls="-.",
    )

    ax1.grid()
    ax1.set_ylabel(r"$Q$")
    ax1.legend()
    ax1.set_xlabel(r"$k_y\rho_s$")
    plt.show()


reference_values = {
    "tref_electron": 1000.0 * units.eV,
    "nref_electron": 1e19 * units.meter ** -3,
    "lref_minor_radius": 1.5 * units.meter,
    "bref_B0": 2.0 * units.tesla,
}


# Point to CGYRO input file
cgyro_template = template_dir / "outputs/CGYRO_linear/input.cgyro"

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
    reference_values=reference_values,
    format="hdf5",
    file_name="cgyro_test.hdf5",
)

# Read IDS file back in
new_pyro = ids_to_pyro("cgyro_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)

# Point to GENE input file
gene_template = template_dir / "outputs/GENE_linear/parameters_0001"

# Load in GENE file and output
pyro = Pyro(gk_file=gene_template)
pyro.load_gk_output()
data = pyro.gk_output

if os.path.exists("gene_test.hdf5"):
    os.remove("gene_test.hdf5")

ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS GENE",
    reference_values=reference_values,
    format="hdf5",
    file_name="gene_test.hdf5",
)

new_pyro = ids_to_pyro("gene_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)

# Point to GS2 input file
gs2_template = template_dir / "outputs/GS2_linear/gs2.in"

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
    reference_values=reference_values,
    format="hdf5",
    file_name="gs2_test.hdf5",
)

new_pyro = ids_to_pyro("gs2_test.hdf5")

compare_pyro_run(pyro, new_pyro, ids.code.name)

# Point to Nonlinear CGYRO input file
cgyro_nl_template = template_dir / "outputs/CGYRO_nonlinear/input.cgyro"

# Load in file
pyro = Pyro(gk_file=cgyro_nl_template, gk_code="CGYRO")

# Load in CGYRO output data
pyro.load_gk_output()
data = pyro.gk_output

if os.path.exists("cgyro_nl_test.hdf5"):
    os.remove("cgyro_nl_test.hdf5")

ids = pyro_to_ids(
    pyro,
    comment="Testing IMAS CGYRO NL",
    reference_values=reference_values,
    format="hdf5",
    file_name="cgyro_nl_test.hdf5",
    time_interval=[0, 1]
)

new_pyro = ids_to_pyro("cgyro_nl_test.hdf5")

compare_pyro_run_nl(pyro, new_pyro, ids.code.name)
