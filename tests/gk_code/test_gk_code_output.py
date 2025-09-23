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
    gs2_expected = 0.05033878 + 0.03802604j
    assert_eigenvalue_close(gs2, gs2_expected)

    # Test eigenvalue from CGYRO
    cgyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_linear/input.cgyro", gk_code="CGYRO"
    )
    cgyro.load_gk_output()
    cgyro_expected = 1.164639 - 4.683715j
    assert_eigenvalue_close(cgyro, cgyro_expected)

    # Test eigenvalue from GENE
    gene = Pyro(
        gk_file=template_dir / "outputs/GENE_linear/parameters_0001", gk_code="GENE"
    )
    gene.load_gk_output()
    # TODO Is this correct?
    gene_expected = -12.93966796 + 1.93411654j
    assert_eigenvalue_close(gene, gene_expected)

    # Test eigenvalue from TGLF
    tglf = Pyro(gk_file=template_dir / "outputs/TGLF_linear/input.tglf", gk_code="TGLF")
    tglf.load_gk_output()
    # TODO Is this correct?
    tglf_expected = 0.048426 + 0.056637j
    assert_eigenvalue_close_tglf(tglf, tglf_expected)


@pytest.mark.parametrize("downsample", ({"time_idx": slice(None, None, 3)},))
def test_cgyro_linear_output_downsample(downsample):
    # Test time values from linear CGYRO (can't do fields due to normalisation)
    pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_linear/input.cgyro", gk_code="CGYRO"
    )
    pyro.load_gk_output()

    sel_dict = {}
    for key, value in downsample.items():
        if isinstance(value, int):
            sel_dict[key[:-4]] = [value]
        else:
            sel_dict[key[:-4]] = value

    full_data = pyro.gk_output.data.isel(**sel_dict)

    ds_pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_linear/input.cgyro", gk_code="CGYRO"
    )
    ds_pyro.load_gk_output(downsample=downsample)
    downsample_data = ds_pyro.gk_output

    np.testing.assert_allclose(
        ureg.Quantity(full_data["time"].data).magnitude,
        ureg.Quantity(downsample_data["time"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ureg.Quantity(full_data["phi"].data).magnitude,
        ureg.Quantity(downsample_data["phi"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "downsample",
    (
        {"time_idx": slice(None, None, 2), "theta_idx": 8},
        {"kx_idx": slice(None, None, 4), "theta_idx": slice(2, 14, 2)},
        {"ky_idx": 4, "kx_idx": 16},
    ),
)
def test_cgyro_nonlinear_output_downsample(downsample):
    # Test time/phi values from nonlinear CGYRO
    pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_nonlinear/input.cgyro", gk_code="CGYRO"
    )
    pyro.load_gk_output(load_moments=True)
    sel_dict = {}
    for key, value in downsample.items():
        if isinstance(value, int):
            sel_dict[key[:-4]] = [value]
        else:
            sel_dict[key[:-4]] = value

    full_data = pyro.gk_output.data.isel(**sel_dict)

    ds_pyro = Pyro(
        gk_file=template_dir / "outputs/CGYRO_nonlinear/input.cgyro", gk_code="CGYRO"
    )
    ds_pyro.load_gk_output(load_moments=True, downsample=downsample)
    downsample_data = ds_pyro.gk_output.data

    np.testing.assert_allclose(
        ureg.Quantity(full_data["time"].data).magnitude,
        ureg.Quantity(downsample_data["time"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ureg.Quantity(full_data["phi"].data).magnitude,
        ureg.Quantity(downsample_data["phi"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ureg.Quantity(full_data["heat"].data).magnitude,
        ureg.Quantity(downsample_data["heat"].data).magnitude,
        atol=1e-8,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        ureg.Quantity(full_data["density"].data).magnitude,
        ureg.Quantity(downsample_data["density"].data).magnitude,
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
