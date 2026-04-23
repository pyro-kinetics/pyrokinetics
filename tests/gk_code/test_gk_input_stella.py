import sys
from pathlib import Path
from typing import Dict, Optional

import f90nml
import numpy as np
import pytest

from pyrokinetics import template_dir
from pyrokinetics.gk_code import GKInputSTELLA
from pyrokinetics.gk_code.stella import StellaFormatVersion
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

docs_dir = Path(__file__).parent.parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO  # noqa

template_file = template_dir / "input.stella"
template_file_v1 = template_dir / "input.stella_v1"
template_file_nl = template_dir / "input.stella_nl"


@pytest.fixture
def default_stella():
    return GKInputSTELLA()


@pytest.fixture
def stella():
    return GKInputSTELLA(template_file)


def modified_stella_input(replacements: Dict[str, Optional[Dict[str, Optional[str]]]]):
    """Return a STELLA input file based on the template file, but with
    updated keys/namelists from replacements. Keys that are `None` will be deleted
    """

    input_file = f90nml.read(template_file)

    for namelist, keys in replacements.items():
        if namelist not in input_file:
            if keys is None:
                continue
            input_file[namelist] = {}

        if keys is None:
            del input_file[namelist]
            continue

        for key, value in keys.items():
            if value is None:
                try:
                    del input_file[namelist][key]
                except KeyError:
                    continue
            else:
                input_file[namelist][key] = value

    return GKInputSTELLA.from_str(str(input_file))


def test_read(stella):
    """Ensure a stella file can be read, and that the 'data' attribute is set"""
    params = ["millergeo_parameters", "stella_diagnostics_knobs", "kt_grids_knobs"]
    assert np.all(np.isin(params, list(stella.data)))


def test_read_str():
    """Ensure a stella file can be read as a string, and that the 'data' attribute is set"""
    params = ["millergeo_parameters", "stella_diagnostics_knobs", "kt_grids_knobs"]
    with open(template_file, "r") as f:
        stella = GKInputSTELLA.from_str(f.read())
        assert np.all(np.isin(params, list(stella.data)))


def test_verify_file_type(stella):
    """Ensure that 'verify_file_type' does not raise exception on STELLA file"""
    stella.verify_file_type(template_file)


@pytest.mark.parametrize(
    "filename", ["input.cgyro", "input.gene", "transp.cdf", "helloworld"]
)
def test_verify_file_type_bad_inputs(stella, filename):
    """Ensure that 'verify' raises exception on non-STELLA file"""
    with pytest.raises(Exception):
        stella.verify_file_type(template_dir / filename)


def test_is_nonlinear(stella):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    assert stella.is_linear()
    assert not stella.is_nonlinear()
    stella.data["kt_grids_knobs"]["grid_option"] = "box"
    assert stella.is_linear()
    assert not stella.is_nonlinear()
    stella.data["parameters_physics"] = {"nonlinear": True}
    assert not stella.is_linear()
    assert stella.is_nonlinear()


def test_add_flags(stella):
    stella.add_flags({"foo": {"bar": "baz"}})
    assert stella.data["foo"]["bar"] == "baz"


def test_get_local_geometry(stella):
    # TODO test it has the correct values
    local_geometry = stella.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(stella):
    local_species = stella.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    # TODO test it has the correct values
    assert local_species["electron"]
    assert local_species["ion1"]


def test_get_numerics(stella):
    # TODO test it has the correct values
    numerics = stella.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, stella):
    """Ensure a stella file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = stella.get_local_geometry()
    local_species = stella.get_local_species()
    numerics = stella.get_numerics()
    # Set output path
    filename = tmp_path / "input.in"
    # Write out a new input file
    stella_writer = GKInputSTELLA()
    stella_writer.set(local_geometry, local_species, numerics)
    stella_writer.write(filename)
    # Ensure a new file exists
    assert Path(filename).exists()
    # Ensure it is a valid file
    GKInputSTELLA().verify_file_type(filename)
    stella_reader = GKInputSTELLA(filename)
    new_local_geometry = stella_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = stella_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = stella_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_stella_linear_box(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8, "jtwist": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    shat = stella.get_local_geometry().shat
    jtwist = 8
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    assert np.isclose(numerics.kx, 2 * np.pi * numerics.ky * shat / jtwist)


def test_stella_linear_box_no_jtwist(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "box"},
        "kt_grids_box_parameters": {"ny": 12, "y0": 2, "nx": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    shat = stella.get_local_geometry().shat
    jtwist = 2 * np.pi * shat
    expected_kx = numerics.ky * jtwist / int(jtwist)
    assert np.isclose(numerics.kx, expected_kx)


def test_stella_linear_range(tmp_path):
    replacements = {
        "kt_grids_knobs": {"grid_option": "range"},
        "kt_grids_range_parameters": {"naky": 12, "aky_min": 2, "aky_max": 8},
    }
    stella = modified_stella_input(replacements)

    numerics = stella.get_numerics()
    assert numerics.nkx == 1
    assert numerics.nky == 12
    expected_ky = np.linspace(2, 8, 12)
    assert np.allclose(numerics.ky.m, expected_ky)


def test_drop_species(tmp_path):
    pyro = example_JETTO.main(tmp_path)
    pyro.gk_code = "STELLA"

    n_species = pyro.local_species.nspec
    stored_species = len(
        [key for key in pyro.gk_input.data.keys() if "species_parameters_" in key]
    )
    assert stored_species == n_species

    pyro.local_species.merge_species(
        base_species="deuterium",
        merge_species=["deuterium", "impurity1"],
        keep_base_species_z=True,
        keep_base_species_mass=True,
    )

    pyro.update_gk_code()
    n_species = pyro.local_species.nspec
    stored_species = len(
        [key for key in pyro.gk_input.data.keys() if "species_parameters_" in key]
    )
    assert stored_species == n_species


# ==================== Stella v1 format tests ====================


@pytest.fixture
def stella_v1():
    return GKInputSTELLA(template_file_v1)


def test_format_detection_pre_v1():
    """Pre-v1 format should be detected from the old template."""
    gk = GKInputSTELLA(template_file)
    assert gk._format_version == StellaFormatVersion.PRE_V1


def test_format_detection_legacy():
    """Legacy format should be detected from the NL template (uses knobs/parameters)."""
    gk = GKInputSTELLA(template_file_nl)
    assert gk._format_version == StellaFormatVersion.LEGACY


def test_format_detection_v1():
    """V1 format should be detected from the v1 template."""
    gk = GKInputSTELLA(template_file_v1)
    assert gk._format_version == StellaFormatVersion.V1


def test_v1_read(stella_v1):
    """Ensure v1 template can be read and has expected namelists."""
    params = ["geometry_miller", "species_options", "kxky_grid_option"]
    assert np.all(np.isin(params, list(stella_v1.data)))


def test_v1_verify_file_type():
    """Ensure verify_file_type accepts v1 format."""
    gk = GKInputSTELLA()
    gk.verify_file_type(template_file_v1)


def test_v1_is_nonlinear(stella_v1):
    """V1 format nonlinear detection."""
    assert stella_v1.is_linear()
    stella_v1.data["kxky_grid_option"]["grid_option"] = "box"
    assert stella_v1.is_linear()
    stella_v1.data["gyrokinetic_terms"]["include_nonlinear"] = True
    assert stella_v1.is_nonlinear()


def test_v1_get_local_geometry(stella_v1):
    local_geometry = stella_v1.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_v1_get_local_species(stella_v1):
    local_species = stella_v1.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    assert local_species["electron"]
    assert local_species["ion1"]


def test_v1_get_numerics(stella_v1):
    numerics = stella_v1.get_numerics()
    assert isinstance(numerics, Numerics)


def test_v1_write_roundtrip(tmp_path, stella_v1):
    """V1 roundtrip: read v1 -> get data -> set -> write -> read back -> compare."""
    local_geometry = stella_v1.get_local_geometry()
    local_species = stella_v1.get_local_species()
    numerics = stella_v1.get_numerics()

    filename = tmp_path / "input.in"
    stella_writer = GKInputSTELLA()
    stella_writer.set(local_geometry, local_species, numerics)
    stella_writer.write(filename)

    assert Path(filename).exists()
    GKInputSTELLA().verify_file_type(filename)

    stella_reader = GKInputSTELLA(filename)
    assert stella_reader._format_version == StellaFormatVersion.V1
    new_geom = stella_reader.get_local_geometry()
    assert np.isclose(local_geometry.shat, new_geom.shat)
    new_species = stella_reader.get_local_species()
    assert local_species.nspec == new_species.nspec
    new_numerics = stella_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time


def test_v1_cross_format(tmp_path):
    """Read pre-v1 format, write as v1, verify same physics."""
    stella_pre_v1 = GKInputSTELLA(template_file)
    geom = stella_pre_v1.get_local_geometry()
    species = stella_pre_v1.get_local_species()
    numerics = stella_pre_v1.get_numerics()

    # Write as v1 (default)
    stella_writer = GKInputSTELLA()
    stella_writer.set(geom, species, numerics)
    filename = tmp_path / "v1_output.in"
    stella_writer.write(filename)

    stella_read = GKInputSTELLA(filename)
    assert stella_read._format_version == StellaFormatVersion.V1
    geom2 = stella_read.get_local_geometry()
    assert np.isclose(geom.shat, geom2.shat)
    assert np.isclose(geom.q, geom2.q)
    assert np.isclose(geom.kappa, geom2.kappa)
    species2 = stella_read.get_local_species()
    assert species.nspec == species2.nspec


def test_v1_y0_positive(tmp_path, stella_v1):
    """In v1 format, y0 should be positive (box size = 1/ky)."""
    geom = stella_v1.get_local_geometry()
    species = stella_v1.get_local_species()
    # Get numerics with units from a real read, then modify
    numerics = stella_v1.get_numerics()
    numerics.nky = 4
    numerics.nkx = 5
    numerics.ky = 0.5
    numerics.kx = 0.1
    numerics.nonlinear = True

    stella_writer = GKInputSTELLA()
    stella_writer.set(geom, species, numerics)

    y0 = stella_writer.data["kxky_grid_box"]["y0"]
    assert y0 > 0, f"v1 format requires y0 > 0, got {y0}"
    assert np.isclose(y0, 1.0 / 0.5), f"Expected y0 = 2.0, got {y0}"


def test_v1_box_grid():
    """Test v1 box grid reading with kxky_grid_box namelist."""
    input_str = """
&geometry_options
 geometry_option = "local"
/
&geometry_miller
 rhoc = 0.5
 shat = 0.8
 qinp = 1.4
 rmaj = 3.0
 rgeo = 3.0
 shift = 0.0
 kappa = 1.0
 kapprim = 0.0
 tri = 0.0
 triprim = 0.0
 betaprim = 0.0
/
&gyrokinetic_terms
 include_nonlinear = .false.
/
&electromagnetic
 include_apar = .false.
 include_bpar = .false.
 beta = 0.0
/
&dissipation_and_collisions_options
 vnew_ref = 0.0
 zeff = 1.0
/
&physics_inputs
 rhostar = 0.0
/
&flow_shear
 g_exb = 0.0
/
&scale_gyrokinetic_terms
 fphi = 1.0
/
&adiabatic_electron_response
 adiabatic_option = "field-line-average-term"
/
&species_options
 nspec = 2
/
&species_parameters_1
 z = 1.0
 mass = 1.0
 dens = 1.0
 temp = 1.0
 tprim = 3.0
 fprim = 1.0
 type = "ion"
/
&species_parameters_2
 z = -1.0
 mass = 0.00027
 dens = 1.0
 temp = 1.0
 tprim = 3.0
 fprim = 1.0
 type = "electron"
/
&kxky_grid_option
 grid_option = "box"
/
&kxky_grid_box
 ny = 12
 y0 = 2.0
 nx = 8
 jtwist = 8
/
&z_grid
 nzed = 24
 nperiod = 1
/
&velocity_grids
 nvgrid = 18
 nmu = 12
/
&time_step
 delt = 0.01
/
&time_trace_options
 nstep = 1000
/
&numerical_algorithms
 stream_implicit = .true.
 mirror_implicit = .true.
/
&diagnostics
 nwrite = 100
/
"""
    stella = GKInputSTELLA.from_str(input_str)
    assert stella._format_version == StellaFormatVersion.V1
    numerics = stella.get_numerics()
    shat = stella.get_local_geometry().shat
    assert numerics.nkx == 5
    assert numerics.nky == 4
    assert np.isclose(numerics.ky.m, 1 / 2)
    assert np.isclose(numerics.kx, 2 * np.pi * numerics.ky * shat / 8)


# Fixture generated by running stella's own convert_inputFile.py
# (AUTOMATIC_TESTS/convert_input_files/convert_inputFile.py) on
# src/pyrokinetics/templates/input.stella. It represents stella's authoritative
# pre-v1 -> v1 namelist translation, independent of pyrokinetics' set() logic.
_STELLA_CONVERTED_V1_FIXTURE = (
    Path(__file__).parent / "fixtures" / "input.stella_v1_from_pre_v1"
)


def test_v1_matches_stella_convert_script():
    """Pyrokinetics must read a v1 file produced by stella's own upgrade
    script and extract the same physics as the pre-v1 source template."""
    pre_v1 = GKInputSTELLA(template_file)
    stella_v1 = GKInputSTELLA(_STELLA_CONVERTED_V1_FIXTURE)

    assert pre_v1._format_version == StellaFormatVersion.PRE_V1
    assert stella_v1._format_version == StellaFormatVersion.V1

    g_pre, g_v1 = pre_v1.get_local_geometry(), stella_v1.get_local_geometry()
    for attr in (
        "rho",
        "q",
        "shat",
        "kappa",
        "delta",
        "s_kappa",
        "s_delta",
        "shift",
        "beta_prime",
        "Rmaj",
    ):
        assert np.isclose(getattr(g_pre, attr), getattr(g_v1, attr)), attr

    s_pre, s_v1 = pre_v1.get_local_species(), stella_v1.get_local_species()
    assert s_pre.nspec == s_v1.nspec
    assert s_pre.names == s_v1.names
    for name in s_pre.names:
        for attr in ("z", "mass", "dens", "temp", "inverse_lt", "inverse_ln"):
            assert np.isclose(
                getattr(s_pre[name], attr), getattr(s_v1[name], attr)
            ), f"{name}.{attr}"

    n_pre, n_v1 = pre_v1.get_numerics(), stella_v1.get_numerics()
    for attr in ("nky", "nkx", "ntheta", "nperiod", "nonlinear", "beta"):
        assert getattr(n_pre, attr) == getattr(n_v1, attr) or np.isclose(
            getattr(n_pre, attr), getattr(n_v1, attr)
        ), attr
    assert np.isclose(n_pre.ky.m, n_v1.ky.m)
