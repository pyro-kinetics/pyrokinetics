from pyrokinetics import Pyro
from pyrokinetics.gk_code import gk_inputs, gk_output_readers, GKInput
from pyrokinetics.templates import (
    gk_templates,
    eq_templates,
    kinetics_templates,
    template_dir,
)
from pyrokinetics.local_geometry import (
    LocalGeometry,
    LocalGeometryMiller,
    local_geometries,
)
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics
from pyrokinetics.equilibrium import Equilibrium, equilibrium_readers
from pyrokinetics.kinetics import Kinetics, kinetics_readers

import xarray as xr
import f90nml
import pytest
from itertools import product, permutations, combinations_with_replacement
from pathlib import Path


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
    assert pyro.eq_file == eq_templates[eq_type]
    assert pyro.eq_type == eq_type
    # Ensure local_geometry was not overwritten
    assert pyro.local_geometry is local_geometry


@pytest.mark.parametrize("kinetics_type", ["TRANSP", "SCENE", "JETTO"])
def test_pyro_load_global_kinetics(kinetics_type):
    pyro = Pyro(gk_file=gk_templates["CGYRO"])
    local_species = pyro.local_species
    pyro.load_global_kinetics(kinetics_templates[kinetics_type])
    assert isinstance(pyro.kinetics, Kinetics)
    assert pyro.kinetics_file == kinetics_templates[kinetics_type]
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
    assert pyro.gk_file == gk_templates[gk_code]
    assert pyro.file_name == gk_templates[gk_code].name
    assert pyro.run_directory == template_dir
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


@pytest.mark.parametrize(
    "gk_code,path",
    [
        ["GS2", template_dir / "outputs" / "GS2_linear" / "gs2.out.nc"],
        ["CGYRO", template_dir / "outputs" / "CGYRO_linear"],
        ["GENE", template_dir / "outputs" / "GENE_linear" / "parameters_0001"],
        ["GENE", template_dir / "outputs" / "GENE_linear" / "nrg_0001"],
    ],
)
def test_pyro_load_gk_output_with_path(gk_code, path):
    pyro = Pyro(gk_code=gk_code)
    pyro.load_gk_output(path)
    assert isinstance(pyro.gk_output, xr.Dataset)


@pytest.mark.parametrize(
    "input_path",
    [
        template_dir / "outputs" / "GS2_linear" / "gs2.in",
        template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
        template_dir / "outputs" / "GENE_linear" / "parameters_0001",
    ],
)
def test_pyro_load_gk_output_without_path(input_path):
    pyro = Pyro(gk_file=input_path)
    pyro.load_gk_output()
    assert isinstance(pyro.gk_output, xr.Dataset)


def test_pyro_context_switching():
    # Create empty Pyro
    pyro = Pyro()
    # Switch to new context
    pyro.gk_code = "GS2"
    # Check that the template file has been read
    assert pyro.gk_file == gk_templates["GS2"]
    gk_input = pyro.gk_input
    local_geometry = pyro.local_geometry
    local_species = pyro.local_species
    numerics = pyro.numerics
    # Switch to CGYRO, ensure context is different
    pyro.gk_code = "CGYRO"
    assert pyro.gk_file == gk_templates["CGYRO"]
    assert pyro.gk_input is not gk_input
    assert pyro.local_geometry is not local_geometry
    assert pyro.local_species is not local_species
    assert pyro.numerics is not numerics
    # Switch back to GS2, ensure nothing has been overwritten
    pyro.gk_code = "GS2"
    assert pyro.gk_file == gk_templates["GS2"]
    assert pyro.gk_input is gk_input
    assert pyro.local_geometry is local_geometry
    assert pyro.local_species is local_species
    assert pyro.numerics is numerics


@pytest.mark.parametrize("gk_code", ["GS2", "CGYRO", "GENE"])
def test_file_name(gk_code):
    pyro = Pyro(gk_file=gk_templates[gk_code])
    assert pyro.file_name == gk_templates[gk_code].name


@pytest.mark.parametrize("gk_code", ["GS2", "CGYRO", "GENE"])
def test_run_directory(gk_code):
    pyro = Pyro(gk_file=gk_templates[gk_code])
    assert pyro.run_directory == gk_templates[gk_code].parent


def test_gs2_input():
    # Ensure it returns None unless we've read GS2
    pyro = Pyro()
    assert pyro.gs2_input is None
    pyro.gk_code = "CGYRO"
    assert pyro.gs2_input is None
    pyro.gk_code = "GS2"
    assert isinstance(pyro.gs2_input, f90nml.Namelist)


def test_cgyro_input():
    # Ensure it returns None unless we've read CGYRO
    pyro = Pyro()
    assert pyro.cgyro_input is None
    pyro.gk_code = "GS2"
    assert pyro.cgyro_input is None
    pyro.gk_code = "CGYRO"
    assert isinstance(pyro.cgyro_input, dict)


def test_gene_input():
    # Ensure it returns None unless we've read GENE
    pyro = Pyro()
    assert pyro.gene_input is None
    pyro.gk_code = "GS2"
    assert pyro.gene_input is None
    pyro.gk_code = "GENE"
    assert isinstance(pyro.gene_input, f90nml.Namelist)


def test_convert_gk_code():
    # Create empty Pyro
    pyro = Pyro()
    # Switch to new context
    pyro.gk_code = "GS2"
    # Check that the template file has been read
    assert pyro.gk_file == gk_templates["GS2"]
    gk_input = pyro.gk_input
    local_geometry = pyro.local_geometry
    local_species = pyro.local_species
    numerics = pyro.numerics
    # Switch to CGYRO, ensure context is different
    pyro.gk_code = "CGYRO"
    assert pyro.gk_file == gk_templates["CGYRO"]
    assert pyro.gk_input is not gk_input
    assert pyro.local_geometry is not local_geometry
    assert pyro.local_species is not local_species
    assert pyro.numerics is not numerics
    # Convert back to GS2, ensure things have been overwritten
    pyro.convert_gk_code("GS2")
    assert pyro.gk_file == gk_templates["GS2"]
    assert pyro.gk_input is not gk_input
    assert pyro.local_geometry is not local_geometry
    assert pyro.local_species is not local_species
    assert pyro.numerics is not numerics


def test_check_gk_code():
    # Create empty Pyro
    pyro = Pyro()
    # Ensure check_gk_code fails
    pyro.check_gk_code(raises=False)
    with pytest.raises(RuntimeError) as exc_info:
        pyro.check_gk_code(raises=True)
    assert "local_geometry" in str(exc_info.value)
    assert "local_species" in str(exc_info.value)
    assert "numerics" in str(exc_info.value)
    assert "gk_input" in str(exc_info.value)
    # Read in file, ensure it passes
    pyro.read_gk_file(gk_templates["GS2"])
    assert pyro.check_gk_code(raises=False)
    pyro.check_gk_code(raises=True)


def test_gk_input():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.gk_input is None
    # Return GKInput if nothing is there
    pyro.read_gk_file(gk_templates["GS2"])
    assert isinstance(pyro.gk_input, GKInput)
    # Allow assignment of GKInput types
    pyro.gk_input = gk_inputs["CGYRO"]
    assert isinstance(pyro.gk_input, GKInput)
    # Disallow assignemnt of other types
    with pytest.raises(TypeError):
        pyro.gk_input = 5
    # Returns to none when gk_code is None
    pyro.gk_code = None
    assert pyro.gk_input is None


def test_numerics():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.numerics is None
    # Return GKInput if nothing is there
    pyro.read_gk_file(gk_templates["GS2"])
    assert isinstance(pyro.numerics, Numerics)
    # Allow assignment of Numerics
    pyro.numerics = Numerics()
    assert isinstance(pyro.numerics, Numerics)
    # Disallow assignment of other types
    with pytest.raises(TypeError):
        pyro.numerics = 5
    # Returns to none when gk_code is None
    pyro.gk_code = None
    assert pyro.numerics is None


def test_local_geometry():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.local_geometry is None
    assert pyro.local_geometry_type is None
    # Read in from global eq
    pyro.load_global_eq(eq_templates["GEQDSK"])
    pyro.load_local_geometry(psi_n=0.5)
    assert isinstance(pyro.local_geometry, LocalGeometry)
    assert pyro.local_geometry_type == "Miller"
    local_geometry_from_global = pyro.local_geometry
    # Read in from gyrokinetics, ensure it's different (should be a deep copy)
    pyro.read_gk_file(gk_templates["GS2"])
    assert isinstance(pyro.local_geometry, LocalGeometryMiller)
    assert pyro.local_geometry is not local_geometry_from_global
    assert pyro.local_geometry_type == "Miller"
    # Switch back, should still have the old one
    pyro.gk_code = None
    assert pyro.local_geometry is local_geometry_from_global
    assert pyro.local_geometry_type == "Miller"
    # Can assign with string (creates new empty one)
    pyro.local_geometry = "Miller"
    assert pyro.local_geometry is not local_geometry_from_global
    assert pyro.local_geometry_type == "Miller"
    # Can set to None
    pyro.local_geometry = None
    assert pyro.local_geometry is None
    assert pyro.local_geometry_type is None
    # Can't set to something which isn't LocalGeometry
    with pytest.raises(Exception):
        pyro.local_geometry = 5
    with pytest.raises(Exception):
        pyro.local_geometry = "UnknownType"


def test_local_species():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.local_species is None
    # Read in from global eq/kinetics
    pyro.load_global_eq(eq_templates["GEQDSK"])
    pyro.load_global_kinetics(kinetics_templates["SCENE"])
    pyro.load_local(psi_n=0.5)
    assert isinstance(pyro.local_species, LocalSpecies)
    local_species_from_global = pyro.local_species
    # Read in from gyrokinetics, ensure it's different (should be a deep copy)
    pyro.read_gk_file(gk_templates["GS2"])
    assert isinstance(pyro.local_species, LocalSpecies)
    assert pyro.local_species is not local_species_from_global
    # Switch back, should still have the old one
    pyro.gk_code = None
    assert pyro.local_species is local_species_from_global
    # Can't set to something which isn't LocalSpecies
    with pytest.raises(Exception):
        pyro.local_species = 17
    # Can't set to None
    with pytest.raises(Exception):
        pyro.local_species = None


def test_gk_file():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.gk_file is None
    # Return GKInput if nothing is there
    pyro.read_gk_file(gk_templates["GS2"])
    assert pyro.gk_file == gk_templates["GS2"]
    # Allow assignment of pathlike types
    pyro.gk_file = "hello world"
    assert pyro.gk_file == Path("hello world")
    assert isinstance(pyro.gk_file, Path)
    # Disallow assignemnt of other types
    with pytest.raises(TypeError):
        pyro.gk_file = None
    # Returns to none when gk_code is None
    pyro.gk_code = None
    assert pyro.gk_file is None


def test_eq_file():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.gk_file is None
    # Return GKInput if nothing is there
    pyro.load_global_eq(eq_templates["GEQDSK"])
    assert pyro.eq_file == eq_templates["GEQDSK"]
    # Allow assignment of pathlike types
    pyro.eq_file = "hello world"
    assert pyro.eq_file == Path("hello world")
    assert isinstance(pyro.eq_file, Path)
    # Disallow assignemnt of other types
    with pytest.raises(TypeError):
        pyro.eq_file = None


def test_kinetics_file():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.kinetics_file is None
    # Return GKInput if nothing is there
    pyro.load_global_kinetics(kinetics_templates["SCENE"])
    assert pyro.kinetics_file == kinetics_templates["SCENE"]
    # Allow assignment of pathlike types
    pyro.kinetics_file = "hello world"
    assert pyro.kinetics_file == Path("hello world")
    assert isinstance(pyro.kinetics_file, Path)
    # Disallow assignemnt of other types
    with pytest.raises(TypeError):
        pyro.eq_file = None


def test_gk_output():
    pyro = Pyro()
    # Return None if nothing loaded
    assert pyro.gk_output is None
    assert pyro.gk_output_file is None
    # Load in data...
    pyro.read_gk_file(template_dir / "outputs" / "GS2_linear" / "gs2.in")
    pyro.load_gk_output()
    # Contains Dataset
    assert isinstance(pyro.gk_output, xr.Dataset)
    assert isinstance(pyro.gk_output_file, Path)
    assert pyro.gk_output_file == template_dir / "outputs" / "GS2_linear" / "gs2.out.nc"
    # Can assign new Datasets (not recommended for users!)
    pyro.gk_output = xr.Dataset()
    # Can assign new pathlike
    pyro.gk_output_file = "hello world"
    # Returns to None when gk_code is None
    pyro.gk_code = None
    assert pyro.gk_output is None
    assert pyro.gk_output_file is None


@pytest.mark.parametrize("kinetics_type", ["TRANSP", "JETTO", "SCENE"])
def test_kinetics_type(kinetics_type):
    pyro = Pyro(kinetics_file=kinetics_templates[kinetics_type])
    assert pyro.kinetics_type == kinetics_type


@pytest.mark.parametrize("eq_type", ["TRANSP", "GEQDSK"])
def test_eq_type(eq_type):
    pyro = Pyro(eq_file=eq_templates[eq_type])
    assert pyro.eq_type == eq_type


# The following monkeypatch fixtures modify the global 'factory'/'reader' objects
# gk_inputs, gk_output_readers, local_geometries, equilibrium_readers, and
# kinetics_readers. This simulates the user adding their own plugins at runtime.


@pytest.fixture
def mock_gk_inputs(monkeypatch):
    class MyGKInput(gk_inputs.get_type("GS2")):
        pass

    monkeypatch.setitem(gk_inputs, "MyGKInput", MyGKInput)


@pytest.fixture
def mock_gk_output_readers(monkeypatch):
    class MyGKOutput(gk_output_readers.get_type("GS2")):
        pass

    monkeypatch.setitem(gk_output_readers, "MyGKOutput", MyGKOutput)


@pytest.fixture
def mock_local_geometries(monkeypatch):
    class MyLocalGeometry(local_geometries.get_type("Miller")):
        pass

    monkeypatch.setitem(local_geometries, "MyLocalGeometry", MyLocalGeometry)


@pytest.fixture
def mock_equilibrium(monkeypatch):
    class MyEquilibrium(equilibrium_readers.get_type("GEQDSK")):
        pass

    monkeypatch.setitem(equilibrium_readers, "MyEquilibrium", MyEquilibrium)


@pytest.fixture
def mock_kinetics(monkeypatch):
    class MyKinetics(kinetics_readers.get_type("SCENE")):
        pass

    monkeypatch.setitem(kinetics_readers, "MyKinetics", MyKinetics)


def test_supported_gk_inputs(mock_gk_inputs):
    pyro = Pyro()
    assert isinstance(pyro.supported_gk_inputs, list)
    assert "MyGKInput" in pyro.supported_gk_inputs


def test_supported_gk_output_readers(mock_gk_output_readers):
    pyro = Pyro()
    assert isinstance(pyro.supported_gk_output_readers, list)
    assert "MyGKOutput" in pyro.supported_gk_output_readers


def test_supported_local_geometries(mock_local_geometries):
    pyro = Pyro()
    assert isinstance(pyro.supported_local_geometries, list)
    assert "MyLocalGeometry" in pyro.supported_local_geometries


def test_supported_equilibrium_types(mock_equilibrium):
    pyro = Pyro()
    assert isinstance(pyro.supported_equilibrium_types, list)
    assert "MyEquilibrium" in pyro.supported_equilibrium_types


def test_supported_kinetics_types(mock_kinetics):
    pyro = Pyro()
    assert isinstance(pyro.supported_kinetics_types, list)
    assert "MyKinetics" in pyro.supported_kinetics_types
