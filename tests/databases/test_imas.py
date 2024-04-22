from pyrokinetics import Pyro, template_dir

from platform import python_version_tuple

if tuple(int(x) for x in python_version_tuple()[:2]) >= (3, 9):
    from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro

import pytest
import numpy as np
import os
import sys
import pint
from pathlib import Path


def array_similar(x, y, atol=1e-8, rtol=1e-5):
    """
    Ensure arrays are similar, after squeezing dimensions of len 1 and (potentially)
    replacing nans with zeros. Transposes both to same coords.
    """

    # Squeeze out any dims of size 1
    x, y = (
        x.squeeze(drop=True).pint.dequantify(),
        y.squeeze(drop=True).pint.dequantify(),
    )

    return np.allclose(x, y, atol=atol, rtol=rtol)


def assert_close_or_equal(name, left, right, norm=None, atol=1e-8, rtol=1e-5):
    if isinstance(left, (str, list, type(None))) or isinstance(
        right, (str, list, type(None))
    ):
        assert left == right, f"{name}: {left} != {right}"
    else:
        if norm and hasattr(right, "units"):
            try:
                assert np.allclose(
                    left.to(norm), right.to(norm), atol=atol, rtol=rtol
                ), f"{name}: {left.to(norm)} != {right.to(norm)}"
            except pint.DimensionalityError:
                raise ValueError(f"Failure: {name}, {left} != {right}")
        else:
            assert np.allclose(
                left, right, atol=atol, rtol=rtol
            ), f"{name}: {left} != {right}"


@pytest.mark.parametrize(
    "input_path",
    [
        template_dir / "outputs" / "GENE_linear" / "parameters_0001",
        template_dir / "outputs" / "GS2_linear" / "gs2.in",
        template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
        template_dir / "outputs" / "GKW_linear" / "input.dat",
    ],
)
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
def test_pyro_to_imas_roundtrip(tmp_path, input_path):
    pyro = Pyro(gk_file=input_path)

    gk_code = pyro.gk_code

    if gk_code == "GKW":
        output_convention = "GKW"
    else:
        output_convention = "pyrokinetics"

    pyro.load_gk_output(output_convention=output_convention)

    reference_values = {
        "tref_electron": 1000.0 * pyro.norms.units.eV,
        "nref_electron": 1e19 * pyro.norms.units.meter**-3,
        "lref_major_radius": 3.0 * pyro.norms.units.meter,
        "bref_B0": 2.0 * pyro.norms.units.tesla,
    }

    hdf5_file_name = tmp_path / f"test_{gk_code}.h5"

    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)

    # Write IDS file
    pyro_to_ids(
        pyro,
        comment=f"Testing IMAS {gk_code.upper()}",
        reference_values=reference_values,
        format="hdf5",
        file_name=hdf5_file_name,
    )

    # Ensure IDS was written
    assert os.path.exists(hdf5_file_name)

    new_pyro = ids_to_pyro(hdf5_file_name)

    old_gk_output = pyro.gk_output
    new_gk_output = new_pyro.gk_output

    # Test data
    final_time_only = [
        "particle",
        "heat",
        "momentum",
    ]

    for data_var in old_gk_output.data_vars:
        if data_var in final_time_only:
            assert array_similar(
                old_gk_output[data_var].isel(time=-1),
                new_gk_output[data_var].isel(time=-1),
            )
        else:
            assert array_similar(old_gk_output[data_var], new_gk_output[data_var])

    # Test coords
    skip_coords = ["energy", "pitch"]

    for c in old_gk_output.coords:
        if c in skip_coords:
            continue
        dtype = old_gk_output[c].dtype
        if dtype == "float64" or dtype == "complex128":
            assert array_similar(old_gk_output[c], new_gk_output[c])
        else:
            assert np.array_equal(old_gk_output[c], new_gk_output[c])


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
def test_pyro_to_imas_roundtrip_nonlinear(tmp_path):

    input_path = template_dir / "outputs" / "CGYRO_nonlinear" / "input.cgyro"
    pyro = Pyro(gk_file=input_path)
    pyro.load_gk_output()

    gk_code = pyro.gk_code

    reference_values = {
        "tref_electron": 1000.0 * pyro.norms.units.eV,
        "nref_electron": 1e19 * pyro.norms.units.meter**-3,
        "lref_major_radius": 3.0 * pyro.norms.units.meter,
        "bref_B0": 2.0 * pyro.norms.units.tesla,
    }

    hdf5_file_name = tmp_path / f"test_nl_{gk_code}.h5"

    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)

    # Write IDS file
    pyro_to_ids(
        pyro,
        comment=f"Testing IMAS {gk_code.upper()}",
        reference_values=reference_values,
        format="hdf5",
        file_name=hdf5_file_name,
        time_interval=[0.0, 1.0],
    )

    # Ensure IDS was written
    assert os.path.exists(hdf5_file_name)

    new_pyro = ids_to_pyro(hdf5_file_name)

    old_gk_output = pyro.gk_output
    new_gk_output = new_pyro.gk_output

    # Test data
    average_time = [
        "particle",
        "heat",
        "momentum",
    ]

    for data_var in old_gk_output.data_vars:
        if data_var in average_time:
            assert array_similar(
                old_gk_output[data_var].mean(dim="time"),
                new_gk_output[data_var].isel(time=-1),
            )
        else:
            assert array_similar(old_gk_output[data_var], new_gk_output[data_var])

    # Test coords
    skip_coords = ["energy", "pitch"]

    for c in old_gk_output.coords:
        if c in skip_coords:
            continue
        dtype = old_gk_output[c].dtype
        if dtype == "float64" or dtype == "complex128":
            assert array_similar(old_gk_output[c], new_gk_output[c])
        else:
            assert np.array_equal(old_gk_output[c], new_gk_output[c])


# Point to gkw input file
this_dir = Path(__file__).parent
gkw_template = this_dir / "golden_answers/input.dat"

# Load in file
pyro = Pyro(gk_file=gkw_template, gk_code="GKW")
pyro.load_gk_output(output_convention="GKW")

pyro_gk_output = pyro.gk_output.data.isel(time=-1, drop=True)

# Read IDS file in
new_pyro = ids_to_pyro(this_dir / "golden_answers/imas_example.h5")

ids_gk_output = new_pyro.gk_output.data.isel(time=-1, drop=True)

FIXME_ignore_geometry_attrs = [
    "B0",
    "psi_n",
    "r_minor",
    "Fpsi",
    "FF_prime",
    "R",
    "Z",
    "theta",
    "b_poloidal",
    "dpsidr",
    "pressure",
    "dpressure_drho",
    "Z0",
    "R_eq",
    "Z_eq",
    "theta_eq",
    "b_poloidal_eq",
    "Zmid",
    "dRdtheta",
    "dRdr",
    "dZdtheta",
    "dZdr",
    "jacob",
    "unit_mapping",
]


def test_compare_roundtrip_local_geometry():
    for key in pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(
            f"{new_pyro.gk_code} {key}",
            pyro.local_geometry[key],
            new_pyro.local_geometry[key],
        )


numerics_fields = [
    "ntheta",
    "nperiod",
    "nenergy",
    "npitch",
    "nky",
    "nkx",
    "ky",
    "kx",
    "delta_time",
    "max_time",
    "theta0",
    "phi",
    "apar",
    "bpar",
    "beta",
    "nonlinear",
    "gamma_exb",
]


def test_compare_roundtrip_numerics():
    for attr in numerics_fields:
        assert_close_or_equal(
            f"{new_pyro.gk_code} {attr}",
            getattr(pyro.numerics, attr),
            getattr(new_pyro.numerics, attr),
        )


species_fields = [
    "name",
    "mass",
    "z",
    "dens",
    "temp",
    "nu",
    "inverse_lt",
    "inverse_ln",
    "domega_drho",
]


def test_compare_roundtrip_local_species():

    assert pyro.local_species.keys() == new_pyro.local_species.keys()

    for key in pyro.local_species.keys():
        if key in pyro.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"{new_pyro.gk_code} {key}.{field}",
                    pyro.local_species[key][field],
                    new_pyro.local_species[key][field],
                    pyro.norms.gkw,
                )
        else:
            assert_close_or_equal(
                f"{new_pyro.gk_code} {key}",
                pyro.local_species[key],
                new_pyro.local_species[key],
                pyro.norms.gkw,
            )


@pytest.mark.parametrize(
    "coord",
    ["kx", "ky", "theta", "energy", "pitch", "field", "species"],
)
def test_get_coords(coord):

    if coord == "theta":
        atol = 0.0001
    else:
        atol = 1e-8

    dtype = pyro_gk_output[coord].dtype
    if dtype == "float64" or dtype == "complex128":
        assert array_similar(pyro_gk_output[coord], ids_gk_output[coord], atol=atol)
    else:
        assert np.array_equal(pyro_gk_output[coord], ids_gk_output[coord])


@pytest.mark.parametrize(
    "var",
    [
        "phi",
        "apar",
        "bpar",
        "particle",
        "momentum",
        "heat",
        "eigenvalues",
        "eigenfunctions",
        "growth_rate",
        "mode_frequency",
    ],
)
def test_data_vars(var):
    dtype = pyro_gk_output[var].dtype
    if dtype == "complex128":
        assert array_similar(
            np.abs(pyro_gk_output[var]), np.abs(ids_gk_output[var]), rtol=1e-3
        )
    else:
        assert array_similar(pyro_gk_output[var], ids_gk_output[var], rtol=1e-3)
