from pyrokinetics import Pyro, template_dir
from pyrokinetics.templates import gk_templates
import numpy as np
import pint
from itertools import product

import pytest

import sys
import pathlib

docs_dir = pathlib.Path(__file__).parent.parent / "docs"
sys.path.append(str(docs_dir))
from examples import example_JETTO, example_PFILE  # noqa


def assert_close_or_equal(name, left, right, norm=None):
    if isinstance(left, (str, list, type(None))) or isinstance(
        right, (str, list, type(None))
    ):
        assert left == right, f"{name}: {left} != {right}"
    else:
        if norm and hasattr(right, "units"):
            try:
                assert np.allclose(
                    left.to(norm), right.to(norm), atol=1e-4
                ), f"{name}: {left.to(norm)} != {right.to(norm)}"
            except pint.DimensionalityError:
                raise ValueError(f"Failure: {name}, {left} != {right}")
        else:
            assert np.allclose(left, right, atol=1e-4), f"{name}: {left} != {right}"


@pytest.fixture(scope="module")
def setup_roundtrip(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("roundtrip")
    pyro = example_JETTO.main(tmp_path)

    # Rename the ion species in the original pyro object
    pyro.local_species["names"] = ["electron", "ion1", "ion2"]
    pyro.local_species["ion1"] = pyro.local_species.pop("deuterium")
    pyro.local_species["ion1"].name = "ion1"
    pyro.local_species["ion2"] = pyro.local_species.pop("impurity1")
    pyro.local_species["ion2"].name = "ion2"

    gs2 = Pyro(gk_file=tmp_path / "test_jetto.gs2", gk_code="GS2")
    cgyro = Pyro(gk_file=tmp_path / "test_jetto.cgyro", gk_code="CGYRO")
    gene = Pyro(gk_file=tmp_path / "test_jetto.gene", gk_code="GENE")
    tglf = Pyro(gk_file=tmp_path / "test_jetto.tglf", gk_code="TGLF")
    gkw = Pyro(gk_file=tmp_path / "test_jetto.gkw", gk_code="GKW")
    stella = Pyro(gk_file=tmp_path / "test_jetto.stella", gk_code="STELLA")

    return {
        "pyro": pyro,
        "gs2": gs2,
        "cgyro": cgyro,
        "gene": gene,
        "tglf": tglf,
        "gkw": gkw,
        "stella": stella,
    }


@pytest.mark.parametrize(
    "gk_code_a, gk_code_b",
    [
        ["gs2", "cgyro"],
        ["gene", "tglf"],
        ["cgyro", "gene"],
        ["tglf", "gs2"],
        ["gkw", "gene"],
        ["stella", "gs2"],
    ],
)
def test_compare_roundtrip(setup_roundtrip, gk_code_a, gk_code_b):
    pyro = setup_roundtrip["pyro"]
    code_a = setup_roundtrip[gk_code_a]
    code_b = setup_roundtrip[gk_code_b]

    FIXME_ignore_geometry_attrs = [
        "B0",
        "psi_n",
        "r_minor",
        "a_minor",
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
        "Zmid",
        "dRdtheta",
        "dRdr",
        "dZdtheta",
        "dZdr",
        "bunit_over_b0",
        "jacob",
        "unit_mapping",
    ]

    species_fields = [
        "name",
        "mass",
        "z",
        "dens",
        "temp",
        "nu",
        "inverse_lt",
        "inverse_ln",
    ]

    assert pyro.local_species.keys() == code_a.local_species.keys()
    assert code_a.local_species.keys() == code_b.local_species.keys()

    for key in pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(
            f"{code_a.gk_code} {key}",
            pyro.local_geometry[key],
            code_a.local_geometry[key],
            pyro.norms,
        )
        assert_close_or_equal(
            f"{code_a.gk_code} {key}",
            code_a.local_geometry[key],
            code_b.local_geometry[key],
            pyro.norms,
        )

    for key in pyro.local_species.keys():
        if key in pyro.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"{code_a.gk_code} {key}.{field}",
                    pyro.local_species[key][field],
                    code_a.local_species[key][field],
                    pyro.norms,
                )
                assert_close_or_equal(
                    f"{code_a.gk_code} {key}.{field}",
                    code_a.local_species[key][field],
                    code_b.local_species[key][field],
                    pyro.norms,
                )
        else:
            assert_close_or_equal(
                f"{code_a.gk_code} {key}",
                pyro.local_species[key],
                code_a.local_species[key],
                pyro.norms,
            )
            assert_close_or_equal(
                f"{code_a.gk_code} {key}",
                code_a.local_species[key],
                code_b.local_species[key],
                pyro.norms,
            )


@pytest.mark.parametrize("gk_code_out", ["CGYRO", "GS2", "STELLA", "GENE", "GKW"])
def test_switch_code_nl_tglf(gk_code_out, tmp_path):
    # Create directory to work in
    tglf_input = template_dir / "outputs" / "TGLF_transport" / "input.tglf"

    pyro = Pyro(gk_file=tglf_input)

    gk_file_out = pathlib.Path(tmp_path / f"input.{gk_code_out.lower()}")
    pyro.write_gk_file(
        file_name=gk_file_out,
        gk_code=gk_code_out,
        template_file=gk_templates[gk_code_out],
    )

    assert gk_file_out.is_file()


@pytest.mark.parametrize(
    "gk_file,gk_code",
    [
        *product([gk_templates["GS2"]], ["CGYRO", "GENE", "TGLF"]),
        *product([gk_templates["CGYRO"]], ["GS2", "GENE", "TGLF"]),
        *product([gk_templates["GENE"]], ["GS2", "CGYRO", "TGLF"]),
        *product([gk_templates["TGLF"]], ["GS2", "CGYRO", "GENE"]),
        *product([gk_templates["GKW"]], ["GS2", "CGYRO", "GENE"]),
        *product([gk_templates["STELLA"]], ["GS2", "CGYRO", "GENE"]),
    ],
)
def test_switch_gk_codes(gk_file, gk_code):
    pyro = Pyro(gk_file=gk_file)

    original_gk_code = pyro.gk_code

    # GKW should raise error as R_major/a_minor is not defined anywhere
    if original_gk_code == "GKW":
        with pytest.raises(Exception):
            pyro.gk_code = gk_code
        pyro.norms.set_ref_ratios(aspect_ratio=3.0)

    pyro.gk_code = gk_code
    assert pyro.gk_code == gk_code

    pyro.gk_code = original_gk_code
    assert pyro.gk_code == original_gk_code

    original_pyro = Pyro(gk_file=gk_file)

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

    for attr in numerics_fields:
        assert_close_or_equal(
            f"{original_pyro.gk_code} {attr}",
            getattr(pyro.numerics, attr),
            getattr(original_pyro.numerics, attr),
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

    local_geometry_ignore = [
        "unit_mapping",
    ]

    assert pyro.local_species.keys() == original_pyro.local_species.keys()

    for key in pyro.local_geometry.keys():
        if key in local_geometry_ignore:
            continue
        assert_close_or_equal(
            f"{original_pyro.gk_code} {key}",
            pyro.local_geometry[key],
            original_pyro.local_geometry[key],
            pyro.norms,
        )

    for key in pyro.local_species.keys():
        if key in pyro.local_species["names"]:
            for field in species_fields:
                assert_close_or_equal(
                    f"{original_pyro.gk_code} {key}.{field}",
                    pyro.local_species[key][field],
                    original_pyro.local_species[key][field],
                    pyro.norms,
                )
        else:
            assert_close_or_equal(
                f"{original_pyro.gk_code} {key}",
                pyro.local_species[key],
                original_pyro.local_species[key],
                pyro.norms,
            )


@pytest.fixture(scope="module")
def setup_roundtrip_exb(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("roundtrip_exb")
    pyro = example_PFILE.main(tmp_path)

    gs2 = Pyro(gk_file=tmp_path / "test_pfile.gs2", gk_code="GS2")
    cgyro = Pyro(gk_file=tmp_path / "test_pfile.cgyro", gk_code="CGYRO")
    gene = Pyro(gk_file=tmp_path / "test_pfile.gene", gk_code="GENE")
    tglf = Pyro(gk_file=tmp_path / "test_pfile.tglf", gk_code="TGLF")
    gkw = Pyro(gk_file=tmp_path / "test_pfile.gkw", gk_code="GKW")
    stella = Pyro(gk_file=tmp_path / "test_pfile.stella", gk_code="STELLA")

    return {
        "pyro": pyro,
        "gs2": gs2,
        "cgyro": cgyro,
        "gene": gene,
        "tglf": tglf,
        "gkw": gkw,
        "stella": stella,
    }


@pytest.mark.parametrize(
    "gk_code_a, gk_code_b",
    [
        ["gs2", "cgyro"],
        ["gene", "tglf"],
        ["cgyro", "gene"],
        ["tglf", "gs2"],
        ["gkw", "gene"],
        ["stella", "gs2"],
    ],
)
def test_compare_roundtrip_exb(setup_roundtrip_exb, gk_code_a, gk_code_b):
    pyro = setup_roundtrip_exb["pyro"]
    code_a = setup_roundtrip_exb[gk_code_a]
    code_b = setup_roundtrip_exb[gk_code_b]

    assert np.isclose(pyro.numerics.gamma_exb.m, -0.08743732140255926, atol=1e-4)

    assert_close_or_equal(
        f"{code_a.gk_code} gamma_exb",
        pyro.numerics.gamma_exb,
        code_a.numerics.gamma_exb,
        pyro.norms,
    )

    assert_close_or_equal(
        f"{code_a.gk_code} gamma_exb",
        code_a.numerics.gamma_exb,
        code_b.numerics.gamma_exb,
        pyro.norms,
    )

    if "stella" not in [gk_code_a, gk_code_b]:
        assert_close_or_equal(
            f"{code_a.gk_code} domega_drho",
            pyro.local_species.electron.domega_drho,
            code_a.local_species.electron.domega_drho,
            pyro.norms,
        )

        assert_close_or_equal(
            f"{code_a.gk_code} domega_drho",
            code_a.local_species.electron.domega_drho,
            code_b.local_species.electron.domega_drho,
            pyro.norms,
        )

        assert np.isclose(
            pyro.local_species.electron.domega_drho.m, 0.5490340792538756, atol=1e-4
        )


@pytest.fixture(scope="module")
def setup_roundtrip_mxh(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("roundtrip_mxh")
    pyro = example_JETTO.main(tmp_path, geometry_type="MXH")

    # Rename the ion species in the original pyro object
    pyro.local_species["names"] = ["electron", "ion1", "ion2"]
    pyro.local_species["ion1"] = pyro.local_species.pop("deuterium")
    pyro.local_species["ion1"].name = "ion1"
    pyro.local_species["ion2"] = pyro.local_species.pop("impurity1")
    pyro.local_species["ion2"].name = "ion2"

    cgyro = Pyro(gk_file=tmp_path / "test_jetto.cgyro", gk_code="CGYRO")
    gene = Pyro(gk_file=tmp_path / "test_jetto.gene", gk_code="GENE")

    return {
        "pyro": pyro,
        "cgyro": cgyro,
        "gene": gene,
    }


@pytest.mark.parametrize(
    "gk_code_a, gk_code_b",
    [
        ["gene", "cgyro"],
    ],
)
def test_compare_roundtrip_mxh(setup_roundtrip_mxh, gk_code_a, gk_code_b):
    pyro = setup_roundtrip_mxh["pyro"]
    code_a = setup_roundtrip_mxh[gk_code_a]
    code_b = setup_roundtrip_mxh[gk_code_b]

    FIXME_ignore_geometry_attrs = [
        "B0",
        "psi_n",
        "a_minor",
        "Fpsi",
        "FF_prime",
        "R",
        "Z",
        "theta",
        "b_poloidal",
        "dpsidr",
        "pressure",
        "dRdtheta",
        "dRdr",
        "dZdtheta",
        "dZdr",
        "jacob",
        "unit_mapping",
        "thetaR",
        "dthetaR_dr",
        "dthetaR_dtheta",
    ]

    for key in pyro.local_geometry.keys():
        if key in FIXME_ignore_geometry_attrs:
            continue
        assert_close_or_equal(
            f"{code_a.gk_code} {key}",
            pyro.local_geometry[key],
            code_a.local_geometry[key],
            pyro.norms,
        )
        assert_close_or_equal(
            f"{code_a.gk_code} {key}",
            code_a.local_geometry[key],
            code_b.local_geometry[key],
            pyro.norms,
        )
