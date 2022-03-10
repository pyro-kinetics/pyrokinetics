from pyrokinetics import PyroAlt as Pyro
from pyrokinetics import template_dir
from pyrokinetics.local_geometry import LocalGeometry
from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.numerics import Numerics
from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.kinetics import Kinetics

import pytest


@pytest.mark.parametrize(
    "gk_input_file,gk_input_type",
    [("input.gs2", "GS2"), ("input.gs2", None)],
)
def test_init_with_gk_file(gk_input_file, gk_input_type):
    filename = template_dir / gk_input_file
    pyro = Pyro(gk_input_file=filename, gk_input_type=gk_input_type)
    assert isinstance(pyro.gk_input_data, dict)
    assert str(pyro.gk_input_file) == str(template_dir / gk_input_file)
    if "gs2" in gk_input_file:
        assert pyro.gk_input_type == "GS2"
    assert isinstance(pyro.local_geometry, LocalGeometry)
    assert isinstance(pyro.local_species, LocalSpecies)
    assert isinstance(pyro.numerics, Numerics)


@pytest.mark.parametrize(
    "eq_file,eq_type",
    [
        ("transp_eq.geqdsk", "GEQDSK"),
        ("transp_eq.geqdsk", None),
        ("transp_eq.cdf", "TRANSP"),
        ("transp_eq.cdf", None),
    ],
)
def test_init_with_eq_file(eq_file, eq_type):
    pyro = Pyro(eq_file=template_dir / eq_file, eq_type=eq_type)
    assert str(pyro.eq_file) == str(template_dir / eq_file)
    if "geqdsk" in eq_file:
        assert pyro.eq_type == "GEQDSK"
    else:
        assert pyro.eq_type == "TRANSP"
    assert isinstance(pyro.eq, Equilibrium)


@pytest.mark.parametrize(
    "kinetics_file,kinetics_type",
    [
        ("scene.cdf", "SCENE"),
        ("scene.cdf", None),
        ("jetto.cdf", "JETTO"),
        ("jetto.cdf", None),
        ("transp.cdf", "TRANSP"),
        ("transp.cdf", None),
    ],
)
def test_init_with_kinetics_file(kinetics_file, kinetics_type):
    pyro = Pyro(kinetics_file=template_dir / kinetics_file, kinetics_type=kinetics_type)
    assert str(pyro.kinetics_file) == str(template_dir / kinetics_file)
    if "scene" in kinetics_file:
        assert pyro.kinetics_type == "SCENE"
    if "jetto" in kinetics_file:
        assert pyro.kinetics_type == "JETTO"
    if "transp" in kinetics_file:
        assert pyro.kinetics_type == "TRANSP"
    assert isinstance(pyro.kinetics, Kinetics)


@pytest.mark.parametrize(
    "eq_file,eq_type",
    [
        ("transp_eq.geqdsk", "GEQDSK"),
        ("transp_eq.cdf", "TRANSP"),
    ],
)
def test_eq_to_local_geometry(eq_file, eq_type):
    pyro = Pyro(eq_file=template_dir / eq_file, eq_type=eq_type, psi_n=0.5)
    assert isinstance(pyro.local_geometry, LocalGeometry)


@pytest.mark.parametrize(
    "kinetics_file,kinetics_type",
    [
        ("scene.cdf", "SCENE"),
        ("jetto.cdf", "JETTO"),
        ("transp.cdf", "TRANSP"),
    ],
)
def test_kinetics_to_local_species(kinetics_file, kinetics_type):
    pyro = Pyro(
        kinetics_file=template_dir / kinetics_file,
        kinetics_type=kinetics_type,
        psi_n=0.5,
        a_minor=1,
    )
    assert isinstance(pyro.local_species, LocalSpecies)
