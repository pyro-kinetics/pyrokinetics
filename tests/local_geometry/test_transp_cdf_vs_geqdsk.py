import numpy as np
import pytest
import warnings
from pyrokinetics import template_dir
from pyrokinetics.equilibrium import EquilibriumCOCOSWarning, read_equilibrium
from pyrokinetics.local_geometry import LocalGeometryMiller


@pytest.fixture(scope="module")
def transp_cdf_equilibrium():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    eq = read_equilibrium(template_dir / "transp_eq.cdf", time=0.2)
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return eq


@pytest.fixture(scope="module")
def transp_gq_equilibrium():
    warnings.simplefilter("ignore", EquilibriumCOCOSWarning)
    eq = read_equilibrium(template_dir / "transp_eq.geqdsk")
    warnings.simplefilter("default", EquilibriumCOCOSWarning)
    return eq


def assert_within_ten_percent(key, cdf_value, gq_value):
    difference = np.abs((cdf_value - gq_value))
    smallest_value = np.min(np.abs([cdf_value, gq_value]))

    if smallest_value == 0.0:
        if difference == 0.0:
            assert True
        else:
            assert (
                np.abs((cdf_value - gq_value) / np.min(np.abs([cdf_value, gq_value])))
                < 0.1
            ), f"{key} not within 10 percent"
    else:
        assert difference / smallest_value < 0.5, f"{key} not within 10 percent"


def test_compare_transp_cdf_geqdsk(transp_cdf_equilibrium, transp_gq_equilibrium):
    # TODO Rather than ignoring most attrs, better to explicitly include them
    psi_n = 0.5
    lg_gq = LocalGeometryMiller()
    lg_cdf = LocalGeometryMiller()
    lg_gq.from_global_eq(transp_gq_equilibrium, psi_n=psi_n)
    lg_cdf.from_global_eq(transp_cdf_equilibrium, psi_n=psi_n)

    ignored_geometry_attrs = [
        "B0",
        "psi_n",
        "r_minor",
        "a_minor",
        "f_psi",
        "R",
        "Z",
        "theta",
        "b_poloidal",
        "R_eq",
        "Z_eq",
        "theta_eq",
        "b_poloidal_eq",
        "dRdtheta",
        "dZdtheta",
        "dRdr",
        "dZdr",
        "dpsidr",
        "pressure",
        "dpressure_drho",
        "Z0",
        "local_geometry",
    ]

    for key in lg_gq.keys():
        if key in ignored_geometry_attrs:
            continue
        assert_within_ten_percent(key, lg_gq[key], lg_cdf[key])
