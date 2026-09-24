import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, PyroScan, template_dir
from pyrokinetics.diagnostics.extent import Extent


@pytest.mark.parametrize(
    "gk_code, gk_file",
    [
        ("GS2", template_dir / "outputs" / "GS2_linear" / "gs2.in"),
        ("CGYRO", template_dir / "outputs" / "CGYRO_linear" / "input.cgyro"),
        ("GENE", template_dir / "outputs" / "GENE_linear" / "parameters_0001"),
        ("TGLF", template_dir / "outputs" / "TGLF_linear" / "input.tglf"),
    ],
)
def test_pyro_output(gk_code, gk_file):
    pyro = Pyro(gk_file=gk_file, gk_code=gk_code)
    pyro.load_gk_output()
    output = pyro.gk_output

    Extent(output)
    result = output["extent"]
    bounds = output["bounds"]

    assert "theta" not in result.dims
    assert "field" in result.dims
    assert "phi" in result.field.values
    assert "theta" not in bounds.dims
    assert "field" in bounds.dims
    assert "phi" in bounds.field.values
    theta_units = getattr(output["eigenfunctions"].theta.data, "units", None)
    if theta_units is not None:
        assert result.data.units == theta_units
        assert bounds.data.units == theta_units


def test_pyroscan_output():
    json_path = template_dir / "outputs" / "CGYRO_linear_scan"
    pyro_scan = PyroScan(pyroscan_json=json_path / "pyroscan.json", load_base_pyro=True)
    pyro_scan.load_gk_output()
    output = pyro_scan.gk_output

    Extent(output)
    result = output["extent"]
    bounds = output["bounds"]

    assert "theta" not in result.dims
    assert "field" in result.dims
    assert "phi" in result.field.values
    assert "theta" not in bounds.dims
    assert "field" in bounds.dims
    assert "phi" in bounds.field.values
    theta_units = getattr(output["eigenfunctions"].theta.data, "units", None)
    if theta_units is not None:
        assert result.data.units == theta_units
        assert bounds.data.units == theta_units


def test_tglf_eigenfunctions_without_fields():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "TGLF_linear" / "input.tglf",
        gk_code="TGLF",
    )
    pyro.load_gk_output(load_fields=False)
    output = pyro.gk_output
    assert not any(name in output for name in ("phi", "apar", "bpar"))

    Extent(output)

    eigenfunctions = output["eigenfunctions"]
    assert set(output["extent"].dims) == set(eigenfunctions.dims) - {"theta"}
    assert "mode" in output["extent"].dims
    assert np.all(np.isfinite(output["extent"].data))
    assert np.all(output["extent"].data > 0)
    assert_allclose(
        output["extent"].data,
        (output["bounds"].sel(bound="hi") - output["bounds"].sel(bound="lo")).data,
    )

    output.data = output.data.drop_vars("eigenfunctions")
    with pytest.raises(ValueError, match="no eigenfunctions"):
        Extent(output)
