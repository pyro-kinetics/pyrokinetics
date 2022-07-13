from pyrokinetics import Pyro
from pyrokinetics.examples import example_SCENE
import numpy as np


def assert_close_or_equal(name, left, right):
    if isinstance(left, (str, list, type(None))) or isinstance(
        right, (str, list, type(None))
    ):
        assert left == right
    else:
        assert np.allclose(left, right), f"{name}: {left} != {right}"


def test_compare_cgyro_gs2_gene(tmp_path):
    pyro = example_SCENE.main(tmp_path)

    gs2 = Pyro(gk_file=tmp_path / "test_scene.gs2", gk_code="GS2")
    cgyro = Pyro(gk_file=tmp_path / "test_scene.cgyro", gk_code="CGYRO")
    gene = Pyro(gk_file=tmp_path / "test_scene.gene", gk_type="GENE")
    tglf = Pyro(gk_file=tmp_path / "test_scene.tglf", gk_type="TGLF")

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
    ]

    for key in pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(
            f"gs2 {key}", pyro.local_geometry[key], gs2.local_geometry[key]
        )
        assert_close_or_equal(
            f"cgyro {key}", pyro.local_geometry[key], cgyro.local_geometry[key]
        )
        assert_close_or_equal(
            f"gene {key}", pyro.local_geometry[key], gene.local_geometry[key]
        )
        assert_close_or_equal(
            f"tglf {key}", pyro.local_geometry[key], tglf.local_geometry[key]
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

    # Rename the ion species in the original pyro object
    pyro.local_species["names"] = ["electron", "ion1", "ion2"]
    pyro.local_species["ion1"] = pyro.local_species.pop("deuterium")
    pyro.local_species["ion1"].name = "ion1"
    pyro.local_species["ion2"] = pyro.local_species.pop("tritium")
    pyro.local_species["ion2"].name = "ion2"

    assert pyro.local_species.keys() == cgyro.local_species.keys()
    FIXME_ignore_species_attrs = [
        "tref",
        "nref",
        "mref",
        "vref",
        "lref",
        "Bref",
    ]

    for key in pyro.local_species.keys():
        if key in FIXME_ignore_species_attrs:
            continue

        if key in pyro.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"gs2 {key}.{field}",
                    pyro.local_species[key][field],
                    gs2.local_species[key][field],
                )
                assert_close_or_equal(
                    f"cgyro {key}.{field}",
                    pyro.local_species[key][field],
                    cgyro.local_species[key][field],
                )
                assert_close_or_equal(
                    f"gene {key}.{field}",
                    gs2.local_species[key][field],
                    gene.local_species[key][field],
                )
                assert_close_or_equal(
                    f"tglf {key}.{field}",
                    gs2.local_species[key][field],
                    tglf.local_species[key][field],
                )
        else:
            assert_close_or_equal(
                f"gs2 {key}", pyro.local_species[key], gs2.local_species[key]
            )
            assert_close_or_equal(
                f"cgyro {key}", pyro.local_species[key], cgyro.local_species[key]
            )
            assert_close_or_equal(
                f"gene {key}", pyro.local_species[key], gene.local_species[key]
            )
            assert_close_or_equal(
                f"tglf {key}", pyro.local_species[key], tglf.local_species[key]
            )
