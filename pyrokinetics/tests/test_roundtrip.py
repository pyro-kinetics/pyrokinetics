from pyrokinetics import Pyro, template_dir
from pyrokinetics.templates import gk_templates
from pyrokinetics.examples import example_SCENE
import numpy as np
import pint
from itertools import product

import pytest


def assert_close_or_equal(name, left, right, norm=None):
    if isinstance(left, (str, list, type(None))) or isinstance(
        right, (str, list, type(None))
    ):
        assert left == right, f"{name}: {left} != {right}"
    else:
        if norm and not isinstance(right, float):
            try:
                assert np.allclose(
                    left, right.to(norm)
                ), f"{name}: {left} != {right.to(norm)}"
            except pint.DimensionalityError:
                raise ValueError(f"Failure: {name}, {left} != {right}")
        else:
            if name not in ["GS2 s_zeta", "GS2 zeta", "GS2 bunit_over_b0"]:
                assert np.allclose(left, right, atol=1e-4), f"{name}: {left} != {right}"
            else:
                assert True


@pytest.fixture(scope="module")
def setup_roundtrip(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("roundtrip")
    pyro = example_SCENE.main(tmp_path)

    # Rename the ion species in the original pyro object
    pyro.local_species["names"] = ["electron", "ion1", "ion2"]
    pyro.local_species["ion1"] = pyro.local_species.pop("deuterium")
    pyro.local_species["ion1"].name = "ion1"
    pyro.local_species["ion2"] = pyro.local_species.pop("tritium")
    pyro.local_species["ion2"].name = "ion2"

    gs2 = Pyro(gk_file=tmp_path / "test_scene.gs2", gk_code="GS2")
    cgyro = Pyro(gk_file=tmp_path / "test_scene.cgyro", gk_code="CGYRO")
    gene = Pyro(gk_file=tmp_path / "test_scene.gene", gk_code="GENE")
    tglf = Pyro(gk_file=tmp_path / "test_scene.tglf", gk_code="TGLF")
    return {
        "pyro": pyro,
        "gs2": gs2,
        "cgyro": cgyro,
        "gene": gene,
        "tglf": tglf,
    }


@pytest.mark.parametrize(
    "gk_code",
    ["gs2", "cgyro", "gene", "tglf"],
)
def test_compare_roundtrip(setup_roundtrip, gk_code):
    pyro = setup_roundtrip["pyro"]
    code = setup_roundtrip[gk_code]

    FIXME_ignore_geometry_attrs = [
        "B0",
        "psi_n",
        "r_minor",
        "a_minor",
        "f_psi",
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
        "beta_prime",
    ]

    for key in pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(
            f"{code.gk_code} {key}",
            pyro.local_geometry[key],
            code.local_geometry[key],
        )

    species_fields = [
        "name",
        "mass",
        "z",
        "dens",
        "temp",
        "vel",
        "nu",
        "a_lt",
        "a_ln",
        "a_lv",
    ]

    assert pyro.local_species.keys() == code.local_species.keys()

    with pyro.norms.units.as_system(pyro.norms.pyrokinetics), pyro.norms.units.context(
        pyro.norms.context
    ):
        for key in pyro.local_species.keys():
            if key in pyro.local_species["names"]:
                for field in species_fields:
                    assert_close_or_equal(
                        f"{code.gk_code} {key}.{field}",
                        pyro.local_species[key][field],
                        code.local_species[key][field],
                        pyro.norms,
                    )
            else:
                assert_close_or_equal(
                    f"{code.gk_code} {key}",
                    pyro.local_species[key],
                    code.local_species[key],
                    pyro.norms,
                )


@pytest.mark.parametrize(
    "gk_file,gk_code",
    [
        *product([gk_templates["GS2"]], ["CGYRO", "GENE", "TGLF"]),
        *product([gk_templates["CGYRO"]], ["GS2", "GENE", "TGLF"]),
        *product([gk_templates["GENE"]], ["GS2", "CGYRO", "TGLF"]),
        *product([gk_templates["TGLF"]], ["GS2", "CGYRO", "GENE"]),
    ],
)
def test_switch_gk_codes(gk_file, gk_code):

    pyro = Pyro(gk_file=gk_file)

    original_gk_code = pyro.gk_code

    pyro.gk_code = gk_code
    assert pyro.gk_code == gk_code

    pyro.gk_code = original_gk_code
    assert pyro.gk_code == original_gk_code
