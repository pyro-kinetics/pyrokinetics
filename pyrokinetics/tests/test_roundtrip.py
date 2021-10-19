from pyrokinetics import Pyro
from pyrokinetics.examples import example_SCENE
import numpy as np


def assert_close_or_equal(name, left, right):
    if isinstance(left, (str, list, type(None))):
        assert left == right
    else:
        assert np.allclose(left, right), f"{name}: {left} != {right}"


def test_compare_cgyro_gs2(tmp_path):
    example_SCENE.main(tmp_path)

    gs2 = Pyro(gk_file=tmp_path / "test_scene.gs2", gk_type="GS2")
    cgyro = Pyro(gk_file=tmp_path / "test_scene.cgyro", gk_type="CGYRO")

    FIXME_ignore_geometry_attrs = ["beta_prime", "B0"]

    for key in gs2.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(key, gs2.local_geometry[key], cgyro.local_geometry[key])

    species_fields = [
        "name",
        "mass",
        "z",
        "dens",
        "temp",
        "vel",
        # "nu",
        "a_lt",
        "a_ln",
        "a_lv",
    ]

    for key in gs2.local_species.keys():
        if key in gs2.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"{key}.{field}",
                    gs2.local_species[key][field],
                    cgyro.local_species[key][field],
                )
        else:
            assert_close_or_equal(key, gs2.local_species[key], cgyro.local_species[key])
