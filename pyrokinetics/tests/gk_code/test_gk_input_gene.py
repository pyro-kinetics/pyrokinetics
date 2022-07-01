from pyrokinetics.gk_code import GKInputGENE
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics

from pathlib import Path
import numpy as np
import pytest

template_file = template_dir.joinpath("input.gene")


@pytest.fixture
def default_gene():
    return GKInputGENE()


@pytest.fixture
def gene():
    return GKInputGENE(template_file)


def test_read(gene):
    """Ensure a gene file can be read, and that the 'data' attribute is set"""
    params = ["general", "box", "geometry"]
    assert np.all(np.isin(params, list(gene.data)))


def test_read_str():
    """Ensure a gene file can be read as a string, and that the 'data' attribute is set"""
    params = ["general", "box", "geometry"]
    with open(template_file, "r") as f:
        gene = GKInputGENE.from_str(f.read())
        assert np.all(np.isin(params, list(gene.data)))


def test_verify(gene):
    """Ensure that 'verify' does not raise exception on GENE file"""
    gene.verify(template_file)


@pytest.mark.parametrize(
    "filename", ["input.gs2", "input.cgyro", "transp.cdf", "helloworld"]
)
def test_verify_bad_inputs(gene, filename):
    """Ensure that 'verify' raises exception on non-GENE file"""
    with pytest.raises(Exception):
        gene.verify(template_dir.joinpath(filename))


def test_is_nonlinear(gene):
    """Expect template file to be linear. Modify it so that it is nonlinear."""
    gene.data["nonlinear"] = 0
    assert gene.is_linear()
    assert not gene.is_nonlinear()
    gene.data["nonlinear"] = 1
    assert not gene.is_linear()
    assert gene.is_nonlinear()


def test_add_flags(gene):
    gene.add_flags({"foo": {"bar": "baz"}})
    assert gene.data["foo"]["bar"] == "baz"


def test_get_local_geometry(gene):
    # TODO test it has the correct values
    local_geometry = gene.get_local_geometry()
    assert isinstance(local_geometry, LocalGeometryMiller)


def test_get_local_species(gene):
    local_species = gene.get_local_species()
    assert isinstance(local_species, LocalSpecies)
    assert local_species.nspec == 2
    assert len(gene.data["species"]) == 2
    # Ensure you can index gene.data["species"] (doesn't work on some f90nml versions)
    assert gene.data["species"][0]
    assert gene.data["species"][1]
    assert local_species["electron"]
    assert local_species["ion1"]
    # TODO test it has the correct values


def test_get_numerics(gene):
    # TODO test it has the correct values
    numerics = gene.get_numerics()
    assert isinstance(numerics, Numerics)


def test_write(tmp_path, gene):
    """Ensure a gene file can be written, and that no info is lost in the process"""
    # Get template data
    local_geometry = gene.get_local_geometry()
    local_species = gene.get_local_species()
    numerics = gene.get_numerics()

    # Set output path
    filename = tmp_path / "input.in"

    # Write out a new input file
    gene_writer = GKInputGENE()
    gene_writer.set(local_geometry, local_species, numerics)

    # Ensure you can index gene.data["species"] (doesn't work on some f90nml versions)
    assert len(gene_writer.data["species"]) == 2
    assert gene_writer.data["species"][0]
    assert gene_writer.data["species"][1]

    # Write to disk
    gene_writer.write(filename)

    # Ensure a new file exists
    assert Path(filename).exists()

    # Ensure it is a valid file
    GKInputGENE().verify(filename)
    gene_reader = GKInputGENE(filename)
    new_local_geometry = gene_reader.get_local_geometry()
    assert local_geometry.shat == new_local_geometry.shat
    new_local_species = gene_reader.get_local_species()
    assert local_species.nspec == new_local_species.nspec
    new_numerics = gene_reader.get_numerics()
    assert numerics.delta_time == new_numerics.delta_time
