from pyrokinetics import Pyro
from pyrokinetics import template_dir
import numpy as np


def assert_eigenvalue_close(pyro, right):
    left = pyro.gk_output["eigenvalues"].isel(time=-1).data
    assert np.allclose(left, right), f"{pyro.gk_code} eigenvalue: {left} != {right}"


def assert_eigenvalue_close_tglf(pyro, right):
    left = pyro.gk_output["eigenvalues"].isel(mode=0).data
    assert np.allclose(left, right), f"{pyro.gk_code} eigenvalue: {left} != {right}"


def test_gk_codes_output():

    # Test eigenvalue from GS2
    gs2 = Pyro(gk_file=template_dir / "outputs/GS2_linear/gs2.in", gk_code="GS2")
    gs2.load_gk_output()
    gs2_expected = 0.04998472 + 0.03785109j
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
    gene_expected = 12.20707227 + 1.84839224j  # True for the last time-step
    assert_eigenvalue_close(gene, gene_expected)

    # Test eigenvalue from TGLF
    tglf = Pyro(gk_file=template_dir / "outputs/TGLF_linear/input.tglf", gk_code="TGLF")
    tglf.load_gk_output()
    # TODO Is this correct?
    tglf_expected = 0.048426 + 0.056637j
    assert_eigenvalue_close_tglf(tglf, tglf_expected)
