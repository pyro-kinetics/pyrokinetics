from pyrokinetics import Pyro
from pyrokinetics.templates import (
    gk_gs2_template,
    gk_cgyro_template,
    gk_gene_template,
    eq_geqdsk_template,
    eq_transp_template,
    kinetics_scene_template,
    kinetics_jetto_template,
    kinetics_transp_template,
)

import pytest
from itertools import product


@pytest.mark.parametrize(
    "gk_file,gk_code,expected_gk_code",
    [
        *product([gk_gs2_template], ["GS2", None], ["GS2"]),
        *product([gk_cgyro_template], ["CGYRO", None], ["CGYRO"]),
        *product([gk_gene_template], ["GENE", None], ["GENE"]),
    ],
)
def test_pyro_file_type_inference_gk_file(gk_file, gk_code, expected_gk_code):
    """Ensure Pyro can read a gk files with and without hints"""
    pyro = Pyro(gk_file=gk_file, gk_code=gk_code)
    assert pyro.gk_code.code_name == expected_gk_code
    assert hasattr(pyro, "numerics")
    assert hasattr(pyro, "local_geometry")
    assert hasattr(pyro, "local_species")


@pytest.mark.parametrize(
    "gk_file,gk_code",
    [
        *product([gk_gs2_template], ["CGYRO", "GENE"]),
        *product([gk_cgyro_template], ["GS2", "GENE"]),
        *product([gk_gene_template], ["GS2", "CGYRO"]),
    ],
)
def test_pyro_fails_with_wrong_gk_code(gk_file, gk_code):
    with pytest.raises(Exception):
        Pyro(gk_file=gk_file, gk_code=gk_code)


@pytest.mark.parametrize(
    "eq_file,eq_type,expected_eq_type",
    [
        *product([eq_geqdsk_template], ["GEQDSK", None], ["GEQDSK"]),
        *product([eq_transp_template], ["TRANSP", None], ["TRANSP"]),
    ],
)
def test_pyro_file_type_inference_eq(eq_file, eq_type, expected_eq_type):
    """Ensure Pyro can read a eq files with and without hints"""
    pyro = Pyro(eq_file=eq_file, eq_type=eq_type)
    assert pyro.eq.eq_type == expected_eq_type


@pytest.mark.parametrize(
    "eq_file,eq_type",
    [
        (eq_geqdsk_template, "TRANSP"),
        (eq_transp_template, "GEQDSK"),
    ],
)
def test_pyro_fails_with_wrong_eq_type(eq_file, eq_type):
    with pytest.raises(Exception):
        Pyro(eq_file=eq_file, eq_type=eq_type)


@pytest.mark.parametrize(
    "kinetics_file,kinetics_type,expected_kinetics_type",
    [
        *product([kinetics_scene_template], ["SCENE", None], ["SCENE"]),
        *product([kinetics_jetto_template], ["JETTO", None], ["JETTO"]),
        *product([kinetics_transp_template], ["TRANSP", None], ["TRANSP"]),
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
        *product([kinetics_scene_template], ["JETTO", "TRANSP"]),
        *product([kinetics_jetto_template], ["SCENE", "TRANSP"]),
        *product([kinetics_transp_template], ["SCENE", "JETTO"]),
    ],
)
def test_pyro_fails_with_wrong_kinetics_type(kinetics_file, kinetics_type):
    with pytest.raises(Exception):
        Pyro(kinetics_file=kinetics_file, kinetics_type=kinetics_type)
