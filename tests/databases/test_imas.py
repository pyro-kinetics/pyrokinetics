from pyrokinetics import Pyro, template_dir

from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro

import pytest
import numpy as np
import os
import shutil
import sys
import pint
from pathlib import Path
from idspy_dictionaries import ids_gyrokinetics_local
import idspy_toolkit as idspy
from idspy_dictionaries.dataclasses_idsschema import fields


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
        template_dir / "outputs" / "GKW_linear" / "GKW_linear.zip",
        template_dir / "outputs" / "TGLF_linear" / "input.tglf"
    ],
)
def test_pyro_to_imas_roundtrip(tmp_path, input_path):

    if input_path.suffix == ".zip":
        output_convention = "GKW"
        shutil.unpack_archive(input_path, tmp_path / "GKW_output")
        input_path = tmp_path / "GKW_output" / "input.dat"
    else:
        output_convention = "pyrokinetics"

    pyro = Pyro(gk_file=input_path)

    gk_code = pyro.gk_code

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

    skip_vars = ["growth_rate_tolerance"]

    for data_var in old_gk_output.data_vars:
        if data_var in skip_vars:
            continue
        assert array_similar(
            old_gk_output[data_var].isel(time=-1, missing_dims="ignore"),
            new_gk_output[data_var].isel(time=-1, missing_dims="ignore"),
        )

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


@pytest.fixture
def golden_answers() -> Path:
    return Path(__file__).parent / "golden_answers"


@pytest.fixture
def direct_pyro(golden_answers) -> Pyro:
    # Get gkw input file
    gkw_template = golden_answers / "input.dat"

    # Load in file and data directly
    result = Pyro(gk_file=gkw_template, gk_code="GKW")
    result.load_gk_output(output_convention="GKW")
    return result


@pytest.fixture
def pyro_ids_h5(tmp_path, direct_pyro) -> Path:
    output_path = tmp_path / "pyro_ids.h5"
    # Write IDS file
    pyro_to_ids(
        direct_pyro,
        comment="Testing round trip GKW",
        format="hdf5",
        file_name=str(output_path),
    )
    return output_path


@pytest.fixture
def direct_pyro_gk_output(direct_pyro):
    return direct_pyro.gk_output.data.isel(time=-1, drop=True)


@pytest.fixture
def round_pyro(pyro_ids_h5):
    # Read IDS file written by pyro
    return ids_to_pyro(str(pyro_ids_h5))


@pytest.fixture
def round_gk_output(round_pyro):
    return round_pyro.gk_output.data.isel(time=-1, drop=True)


@pytest.fixture
def new_pyro(golden_answers):
    # Read template IDS file
    return ids_to_pyro(str(golden_answers / "imas_example.h5"))


@pytest.fixture
def ids_gk_output(new_pyro):
    return new_pyro.gk_output.data.isel(time=-1, drop=True).isel(ky=10)


@pytest.fixture
def template_ids(golden_answers):
    result = ids_gyrokinetics_local.GyrokineticsLocal()
    result = idspy.hdf5_to_ids(golden_answers / "imas_example.h5", result)
    return result


@pytest.fixture
def pyro_ids(pyro_ids_h5):
    # Read template ids using IDSpy library
    result = ids_gyrokinetics_local.GyrokineticsLocal()
    result = idspy.hdf5_to_ids(str(pyro_ids_h5), result)
    return result


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


def test_write_ids_roundtrip(direct_pyro, new_pyro, round_pyro):

    ids = pyro_to_ids(direct_pyro, comment="Test IDS direct")
    assert isinstance(ids, ids_gyrokinetics_local.GyrokineticsLocal)

    ids = pyro_to_ids(new_pyro, comment="Test IDS new")
    assert isinstance(ids, ids_gyrokinetics_local.GyrokineticsLocal)

    ids = pyro_to_ids(round_pyro, comment="Test IDS round")
    assert isinstance(ids, ids_gyrokinetics_local.GyrokineticsLocal)


def test_compare_roundtrip_local_geometry(direct_pyro, new_pyro, round_pyro):
    for key in direct_pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue

        assert_close_or_equal(
            f"{new_pyro.gk_code} {key}",
            direct_pyro.local_geometry[key],
            new_pyro.local_geometry[key],
        )

        assert_close_or_equal(
            f"{new_pyro.gk_code} {key}",
            direct_pyro.local_geometry[key],
            round_pyro.local_geometry[key],
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


def test_compare_roundtrip_numerics(direct_pyro, new_pyro, round_pyro):
    for attr in numerics_fields:
        assert_close_or_equal(
            f"{new_pyro.gk_code} {attr}",
            getattr(direct_pyro.numerics, attr),
            getattr(new_pyro.numerics, attr),
        )
        assert_close_or_equal(
            f"{new_pyro.gk_code} {attr}",
            getattr(direct_pyro.numerics, attr),
            getattr(round_pyro.numerics, attr),
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


def test_compare_roundtrip_local_species(direct_pyro, new_pyro):

    assert direct_pyro.local_species.keys() == new_pyro.local_species.keys()

    for key in direct_pyro.local_species.keys():
        if key in direct_pyro.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"{new_pyro.gk_code} {key}.{field}",
                    direct_pyro.local_species[key][field],
                    new_pyro.local_species[key][field],
                    direct_pyro.norms.gkw,
                )
        else:
            assert_close_or_equal(
                f"{new_pyro.gk_code} {key}",
                direct_pyro.local_species[key],
                new_pyro.local_species[key],
                direct_pyro.norms.gkw,
            )


@pytest.mark.parametrize(
    "coord",
    ["kx", "ky", "theta", "energy", "pitch", "field", "species"],
)
def test_get_coords(coord, direct_pyro_gk_output, ids_gk_output):

    if coord in ["theta", "ky"]:
        atol = 0.0001
    else:
        atol = 1e-8

    dtype = direct_pyro_gk_output[coord].dtype
    if dtype == "float64" or dtype == "complex128":
        assert array_similar(
            direct_pyro_gk_output[coord], ids_gk_output[coord], atol=atol
        )
    else:
        assert np.array_equal(direct_pyro_gk_output[coord], ids_gk_output[coord])


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
def test_data_vars(var, direct_pyro_gk_output, ids_gk_output):
    dtype = direct_pyro_gk_output[var].dtype
    if dtype == "complex128":
        assert array_similar(
            np.abs(direct_pyro_gk_output[var]), np.abs(ids_gk_output[var]), rtol=1e-3
        )
    else:
        assert array_similar(direct_pyro_gk_output[var], ids_gk_output[var], rtol=1e-3)


# TODO remove shaping coefficients when new test ids is made
skip_attr = [
    "max_repr_length",
    "version",
    "include_full_curvature_drift",
    "include_coriolis_drift",
    "include_centrifugal_effects",
    "collisions_pitch_only",
    "collisions_momentum_conservation",
    "collisions_energy_conservation",
    "collisions_finite_larmor_radius",
    "adiabatic_electrons",
    "potential_energy_norm",
    "potential_energy_gradient_norm",
]

shorten_attr = [
    "shape_coefficients_c",
    "shape_coefficients_s",
    "dc_dr_minor_norm",
    "ds_dr_minor_norm",
]


@pytest.mark.parametrize(
    "base_attr",
    ["normalizing_quantities", "model", "flux_surface", "species_all", "collisions"],
)
def test_ids_comparison(base_attr, pyro_ids, template_ids):
    pyro_attr = getattr(pyro_ids, base_attr)
    template_attr = getattr(template_ids, base_attr)

    field = fields(pyro_attr)

    for f in field:
        if f.name in skip_attr:
            continue
        pyro_data = getattr(pyro_attr, f.name)
        template_data = getattr(template_attr, f.name)

        # Handle different lengths of shaping coefficients
        if f.name in shorten_attr:
            template_data = template_data[:4]

        if f.name == "collisionality_norm":
            assert_close_or_equal(f.name, np.diag(pyro_data), np.diag(template_data))
        else:
            assert_close_or_equal(f.name, pyro_data, template_data)


def test_ids_comparison_species(pyro_ids, template_ids):

    pyro_species = pyro_ids.species
    template_species = template_ids.species

    for i in range(len(pyro_species)):
        pyro_spec = pyro_species[i]
        template_spec = template_species[i]

        field = fields(pyro_spec)
        for f in field:
            if f.name in skip_attr:
                continue

            pyro_data = getattr(pyro_spec, f.name)
            template_data = getattr(template_spec, f.name)

            assert_close_or_equal(f.name, pyro_data, template_data, rtol=1e-4)
