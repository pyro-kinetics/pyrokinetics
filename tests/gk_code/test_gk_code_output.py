import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.units import ureg


def assert_eigenvalue_close(pyro, right):
    left = pyro.gk_output["eigenvalues"].isel(time=-1).data
    np.testing.assert_allclose(
        ureg.Quantity(left).magnitude,
        ureg.Quantity(right).magnitude,
        rtol=1e-5,
        atol=1e-8,
    )


def assert_eigenvalue_close_tglf(pyro, right):
    left = pyro.gk_output["eigenvalues"].isel(mode=0).data
    np.testing.assert_allclose(
        ureg.Quantity(left).magnitude,
        ureg.Quantity(right).magnitude,
        rtol=1e-5,
        atol=1e-8,
    )


def test_gk_codes_output():
    # Test eigenvalue from GS2
    gs2 = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in", gk_code="GS2")
    gs2.load_gk_output()
    gs2_expected = 0.05031906 + 0.03799969j
    assert_eigenvalue_close(gs2, gs2_expected)

    # Test eigenvalue from CGYRO
    cgyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_linear/input.cgyro", gk_code="CGYRO"
    )
    cgyro.load_gk_output()
    cgyro_expected = 1.16463862 - 4.6837147j
    assert_eigenvalue_close(cgyro, cgyro_expected)

    # Test eigenvalue from GENE
    gene = Pyro(
        gk_file=template_dir / "outputs/GENE_linear/parameters_0001", gk_code="GENE"
    )
    gene.load_gk_output()
    # TODO Is this correct?
    gene_expected = -12.20707227 + 1.84839224j  # True for the last time-step
    assert_eigenvalue_close(gene, gene_expected)

    # Test eigenvalue from TGLF
    tglf = Pyro(gk_file=template_dir / "outputs/TGLF_linear/input.tglf", gk_code="TGLF")
    tglf.load_gk_output()
    # TODO Is this correct?
    tglf_expected = -0.048426 + 0.056637j
    assert_eigenvalue_close_tglf(tglf, tglf_expected)


@pytest.mark.parametrize("downsize", (2, 3, 4))
def test_cgyro_linear_output_downsize(downsize):
    # Test time values from linear CGYRO (can't do fields due to normalisation)
    pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_linear/input.cgyro", gk_code="CGYRO"
    )
    pyro.load_gk_output()
    full_data = pyro.gk_output

    pyro.load_gk_output(downsize=downsize)
    downsize_data = pyro.gk_output

    np.testing.assert_allclose(
        ureg.Quantity(full_data["time"][::downsize].data).magnitude,
        ureg.Quantity(downsize_data["time"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )


@pytest.mark.parametrize("downsize", (2, 3, 4))
def test_cgyro_nonlinear_output_downsize(downsize):
    # Test time/phi values from nonlinear CGYRO
    pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_nonlinear/input.cgyro", gk_code="CGYRO"
    )
    pyro.load_gk_output()
    full_data = pyro.gk_output

    pyro.load_gk_output(downsize=downsize)
    downsize_data = pyro.gk_output

    np.testing.assert_allclose(
        ureg.Quantity(full_data["time"][::downsize].data).magnitude,
        ureg.Quantity(downsize_data["time"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ureg.Quantity(full_data["phi"][..., ::downsize].data).magnitude,
        ureg.Quantity(downsize_data["phi"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )


@pytest.mark.parametrize("downsize", (2, 3, 4))
def test_gene_linear_output_downsize(downsize):
    # Test time values from linear CGYRO (can't do fields due to normalisation)
    pyro = Pyro(
        gk_file=template_dir / "outputs/GENE_linear/parameters_0001", gk_code="GENE"
    )

    pyro.load_gk_output()
    full_data = pyro.gk_output

    pyro.load_gk_output(downsize=downsize)
    downsize_data = pyro.gk_output

    np.testing.assert_allclose(
        ureg.Quantity(full_data["time"][::downsize].data).magnitude,
        ureg.Quantity(downsize_data["time"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )
