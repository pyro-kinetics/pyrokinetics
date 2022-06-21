from pyrokinetics import Pyro
from pyrokinetics.gk_code import gk_inputs
from pyrokinetics.templates import (
    gk_templates,
    eq_templates,
    kinetics_templates,
    template_dir,
)
from pyrokinetics.local_geometry import LocalGeometry, LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics
from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.kinetics import Kinetics

import pytest
from itertools import product, permutations, combinations_with_replacement


@pytest.mark.parametrize(
    "gk_file,gk_code,expected_gk_code",
    [
        *product([gk_templates["GS2"]], ["GS2", None], ["GS2"]),
        *product([gk_templates["CGYRO"]], ["CGYRO", None], ["CGYRO"]),
        *product([gk_templates["GENE"]], ["GENE", None], ["GENE"]),
    ],
)
def test_pyro_file_type_inference_gk_file(gk_file, gk_code, expected_gk_code):
    """Ensure Pyro can read a gk files with and without hints"""
    pyro = Pyro(gk_file=gk_file, gk_code=gk_code)
    assert pyro.gk_code == expected_gk_code
    assert isinstance(pyro.numerics, Numerics)
    assert isinstance(pyro.local_geometry, LocalGeometry)
    assert isinstance(pyro.local_species, LocalSpecies)


@pytest.mark.parametrize(
    "gk_file,gk_code",
    [
        *product([gk_templates["GS2"]], ["CGYRO", "GENE"]),
        *product([gk_templates["CGYRO"]], ["GS2", "GENE"]),
        *product([gk_templates["GENE"]], ["GS2", "CGYRO"]),
    ],
)
def test_pyro_fails_with_wrong_gk_code(gk_file, gk_code):
    with pytest.raises(Exception):
        Pyro(gk_file=gk_file, gk_code=gk_code)


@pytest.mark.parametrize(
    "eq_file,eq_type,expected_eq_type",
    [
        *product([eq_templates["GEQDSK"]], ["GEQDSK", None], ["GEQDSK"]),
        *product([eq_templates["TRANSP"]], ["TRANSP", None], ["TRANSP"]),
    ],
)
def test_pyro_file_type_inference_eq(eq_file, eq_type, expected_eq_type):
    """Ensure Pyro can read a eq files with and without hints"""
    pyro = Pyro(eq_file=eq_file, eq_type=eq_type)
    assert pyro.eq.eq_type == expected_eq_type


@pytest.mark.parametrize(
    "eq_file,eq_type",
    [
        (eq_templates["GEQDSK"], "TRANSP"),
        (eq_templates["TRANSP"], "GEQDSK"),
    ],
)
def test_pyro_fails_with_wrong_eq_type(eq_file, eq_type):
    with pytest.raises(Exception):
        Pyro(eq_file=eq_file, eq_type=eq_type)


@pytest.mark.parametrize(
    "kinetics_file,kinetics_type,expected_kinetics_type",
    [
        *product([kinetics_templates["SCENE"]], ["SCENE", None], ["SCENE"]),
        *product([kinetics_templates["JETTO"]], ["JETTO", None], ["JETTO"]),
        *product([kinetics_templates["TRANSP"]], ["TRANSP", None], ["TRANSP"]),
    ],
)
def test_pyro_file_type_inference_kinetics_file(
    kinetics_file, kinetics_type, expected_kinetics_type
):
    """Ensure Pyro can read a kinetics files with and without hints"""
    pyro = Pyro(kinetics_file=kinetics_file, kinetics_type=kinetics_type)
    assert pyro.kinetics.kinetics_type == expected_kinetics_type


@pytest.mark.parametrize(
    "kinetics_file,kinetics_type",
    [
        *product([kinetics_templates["SCENE"]], ["JETTO", "TRANSP"]),
        *product([kinetics_templates["JETTO"]], ["SCENE", "TRANSP"]),
        *product([kinetics_templates["TRANSP"]], ["SCENE", "JETTO"]),
    ],
)
def test_pyro_fails_with_wrong_kinetics_type(kinetics_file, kinetics_type):
    with pytest.raises(Exception):
        Pyro(kinetics_file=kinetics_file, kinetics_type=kinetics_type)


@pytest.mark.parametrize(
    "start_gk_code,end_gk_code", [*permutations(["GS2", "CGYRO", "GENE"], 2)]
)
def test_pyro_convert_gk_code(start_gk_code, end_gk_code):
    """Test we can convert from any gk code to any other gk code"""
    pyro = Pyro(gk_file=gk_templates[start_gk_code])
    start_class_name = pyro.gk_input.__class__.__name__
    assert start_gk_code in start_class_name
    assert end_gk_code not in start_class_name
    pyro.convert_gk_code(end_gk_code)
    end_class_name = pyro.gk_input.__class__.__name__
    assert end_gk_code in end_class_name
    assert start_gk_code not in end_class_name


@pytest.mark.parametrize("eq_type", ["GEQDSK", "TRANSP"])
def test_pyro_load_global_eq(eq_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_geometry = pyro.local_geometry
    pyro.load_global_eq(eq_templates[eq_type])
    assert isinstance(pyro.eq, Equilibrium)
    assert pyro.eq_file.samefile(eq_templates[eq_type])
    assert pyro.eq_type == eq_type
    # Ensure local_geometry was not overwritten
    assert pyro.local_geometry is local_geometry


@pytest.mark.parametrize("kinetics_type", ["TRANSP", "SCENE", "JETTO"])
def test_pyro_load_global_kinetics(kinetics_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_species = pyro.local_species
    pyro.load_global_kinetics(kinetics_templates[kinetics_type])
    assert isinstance(pyro.kinetics, Kinetics)
    assert pyro.kinetics_file.samefile(kinetics_templates[kinetics_type])
    assert pyro.kinetics_type == kinetics_type
    # Ensure local_species was not overwritten
    assert pyro.local_species is local_species


@pytest.mark.parametrize("eq_type", ["GEQDSK", "TRANSP"])
def test_pyro_load_local_geometry(eq_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_geometry = pyro.local_geometry
    pyro.load_global_eq(eq_templates[eq_type])
    pyro.load_local_geometry(psi_n=0.5)
    assert isinstance(pyro.local_geometry, LocalGeometryMiller)
    # Ensure local_geometry was overwritten
    assert pyro.local_geometry is not local_geometry


@pytest.mark.parametrize("kinetics_type", ["TRANSP", "SCENE", "JETTO"])
def test_pyro_load_local_species(kinetics_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_species = pyro.local_species
    pyro.load_global_kinetics(kinetics_templates["SCENE"])
    pyro.load_local_species(psi_n=0.5, a_minor=0.7)
    assert isinstance(pyro.local_species, LocalSpecies)
    # Ensure local_species was overwritten
    assert pyro.local_species is not local_species


@pytest.mark.parametrize(
    "eq_type,kinetics_type",
    [*product(["GEQDSK", "TRANSP"], ["TRANSP", "SCENE", "JETTO"])],
)
def test_pyro_load_local(eq_type, kinetics_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_geometry = pyro.local_geometry
    local_species = pyro.local_species
    pyro.load_global_eq(eq_templates[eq_type])
    pyro.load_global_kinetics(kinetics_templates[kinetics_type])
    pyro.load_local(psi_n=0.5)
    assert isinstance(pyro.local_geometry, LocalGeometryMiller)
    assert isinstance(pyro.local_species, LocalSpecies)
    # Ensure local_species and local_geometry were overwritten
    assert pyro.local_geometry is not local_geometry
    assert pyro.local_species is not local_species


@pytest.mark.parametrize("gk_code", ["GS2", "CGYRO", "GENE"])
def test_pyro_read_gk_file(gk_code):
    pyro = Pyro()
    pyro.read_gk_file(gk_templates[gk_code])
    # Ensure the correct file data now exists
    assert pyro.gk_file.samefile(gk_templates[gk_code])
    assert pyro.file_name == gk_templates[gk_code].name
    assert pyro.run_directory.samefile(template_dir)
    assert pyro.gk_code == gk_code
    # Ensure that the correct geometry/species/numerics are set
    assert isinstance(pyro.gk_input, gk_inputs.get_type(gk_code))
    assert isinstance(pyro.local_species, LocalSpecies)
    assert isinstance(pyro.local_geometry, LocalGeometryMiller)
    assert isinstance(pyro.numerics, Numerics)


@pytest.mark.parametrize(
    "start_gk_code,end_gk_code",
    [*combinations_with_replacement(["GS2", "CGYRO", "GENE"], 2)],
)
def test_pyro_write_gk_file(tmp_path, start_gk_code, end_gk_code):
    # Read file, get results
    pyro = Pyro(gk_file=gk_templates[start_gk_code])
    gk_input = pyro.gk_input
    numerics = pyro.numerics
    local_species = pyro.local_species
    local_geometry = pyro.local_geometry
    # Write out to new gk_code (potentially the same we started with)
    output_dir = tmp_path / "pyrokinetics_write_gk_file_test"
    output_dir.mkdir()
    output_file = output_dir / f"{end_gk_code}.out"
    pyro.write_gk_file(output_file, end_gk_code)
    # Ensure file exists and is of the correct type
    assert output_file.exists()
    readback = gk_inputs[output_file]
    assert isinstance(readback, gk_inputs.get_type(end_gk_code))
    # Ensure that the original gk_input results are unchanged
    assert pyro.gk_input is gk_input
    assert pyro.numerics is numerics
    assert pyro.local_species is local_species
    assert pyro.local_geometry is local_geometry


# TODO Test load_gk_ouput. Need output test files.
