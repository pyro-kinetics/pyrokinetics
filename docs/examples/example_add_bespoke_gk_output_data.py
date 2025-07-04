from pyrokinetics import Pyro
import numpy as np
from pathlib import Path
import struct


def _get_gene_stress_files(
    pyro,
):
    """
    Given a directory name, looks for the files filename/stress_{species}.dat
    If instead given any of the files parameters_####, field_#### or nrg_####,
    looks up the rest of the files in the same directory.
    """

    filename = Path(pyro.gk_file).parent
    species_names = pyro.local_species.names
    files = {}
    prefixes = [f"stress_{species_name}" for species_name in species_names]

    if filename.is_dir():
        # If given a dir name, looks for dir/parameters_0000
        dirname = filename
        dat_matches = np.any([Path(filename / f"{p}.dat").is_file() for p in prefixes])
        if dat_matches:
            suffix = "dat"
            delimiter = "."
        else:
            suffix = "0000"
            delimiter = "_"
    else:
        # If given a file, searches for all similar GENE files in that file's dir
        dirname = filename.parent
        suffix = filename.name.split("_")[-1]
        delimiter = "_"

    # Get all files in the same dir
    for prefix in prefixes:
        if (dirname / f"{prefix}{delimiter}{suffix}").exists():
            files[prefix] = dirname / f"{prefix}{delimiter}{suffix}"

    return files


def _get_fields(pyro, raw_data, downsize=1):
    """
    Sets 3D stresses over time.
    The field coordinates should be (field, species, theta, kx, ky, time)
    """

    # Time data stored as binary (int, double, int)
    time = []
    time_data_fmt = "=idi"
    time_data_size = struct.calcsize(time_data_fmt)

    int_size = 4
    complex_size = 16

    coords = pyro.gk_output.data.coords
    gk_input = pyro.gk_input

    nx = pyro.gk_input.data["box"]["nx0"]
    nz = pyro.gk_input.data["box"]["nz0"]

    nkx = len(coords["kx"])
    nky = len(coords["ky"])
    ntheta = len(coords["theta"])
    ntime = len(coords["time"])
    nfield = len(coords["field"])

    species = [species["name"] for species in pyro.gk_input.data["species"]]
    nspecies = len(species)

    stress_size = nx * nz * nky * complex_size

    sliced_stress = np.empty((nspecies, nfield, nx, nky, nz, ntime), dtype=complex)
    stresses = np.empty((nspecies, nfield, nkx, nky, ntheta, ntime), dtype=complex)

    for i_sp, spec in enumerate(species):
        # Read binary file if present
        if ".h5" not in str(raw_data[f"stress_{spec}"]):
            with open(raw_data[f"stress_{spec}"], "rb") as file:
                for i_time in range(ntime):
                    # Read in time data (stored as int, double int)
                    time_value = float(
                        struct.unpack(time_data_fmt, file.read(time_data_size))[1]
                    )
                    if i_sp == 0:
                        time.append(time_value)
                    for i_stress in range(nfield):
                        file.seek(int_size, 1)
                        binary_stress = file.read(stress_size)
                        raw_stress = np.frombuffer(binary_stress, dtype=np.complex128)
                        sliced_stress[i_sp, i_stress, :, :, :, i_time] = (
                            raw_stress.reshape(
                                (nx, nky, nz),
                                order="F",
                            )
                        )
                        file.seek(int_size, 1)

                    if i_time < ntime - 1:
                        file.seek(
                            (downsize - 1)
                            * (time_data_size + nfield * (2 * int_size + stress_size)),
                            1,
                        )

        # Read .h5 file if binary file absent
        else:
            raise NotImplementedError("Stresses from HDf5 not yet supported")

        # Match pyro convention for ion/electron direction
        sliced_stress = np.conjugate(sliced_stress)

        if not gk_input.is_linear():
            nl_shape = (nspecies, nfield, nkx, nky, ntheta, ntime)
            stresses = sliced_stress.reshape(nl_shape, order="F")

        # Convert from kx to ballooning space
        else:
            try:
                n0_global = gk_input.data["box"]["n0_global"]
                q0 = gk_input.data["geometry"]["q0"]
                phase_fac = -np.exp(-2 * np.pi * 1j * n0_global * q0)
            except KeyError:
                phase_fac = -1
            i_ball = 0

            for i_conn in range(-int(nx / 2) + 1, int((nx - 1) / 2) + 1):
                stresses[:, 0, :, i_ball : i_ball + nz, :] = (
                    sliced_stress[:, i_conn, :, :, :] * (phase_fac) ** i_conn
                )
                i_ball += nz

    # =================================================

    # Overwrite 'time' coordinate as determined in _init_dataset
    coords["time"] = time

    # Original method coords: (species, stress, kx, ky, theta, time)
    # New coords: (field, theta, kx, species, ky, time)
    stresses = stresses.transpose(1, 4, 2, 0, 3, 5)

    # Shift kx component to middle of array
    stresses = np.roll(np.fft.fftshift(stresses, axes=2), -1, axis=2)

    return stresses


pyro = Pyro(gk_file="parameters")

output_convention = "gene"

pyro.load_gk_output(output_convention=output_convention)

# Generate existing data
files = _get_gene_stress_files(pyro)
data = _get_fields(pyro, files)

# Specify coordinates and units
coords = ("field", "theta", "kx", "species", "ky", "time")
units = pyro.norms.units.dimensionless

# Add to GKOutput
pyro.gk_output.add_data(
    name="stress",
    data=data,
    coords=coords,
    units=units,
    output_convention=output_convention,
)

print(pyro.gk_output)
