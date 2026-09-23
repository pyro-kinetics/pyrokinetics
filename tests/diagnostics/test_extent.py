import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, PyroScan, template_dir
from pyrokinetics.dataset_wrapper import DatasetWrapper
from pyrokinetics.diagnostics.extent import Extent
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics.units import ureg


@pytest.mark.parametrize(
    "gk_code, gk_file",
    [
        ("GS2", template_dir / "outputs" / "GS2_linear" / "gs2.in"),
        ("CGYRO", template_dir / "outputs" / "CGYRO_linear" / "input.cgyro"),
        ("GENE", template_dir / "outputs" / "GENE_linear" / "parameters_0001"),
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
    theta_units = getattr(output["phi"].theta.data, "units", None)
    if theta_units is not None:
        assert result.data.units == theta_units
        assert bounds.data.units == theta_units



def test_pyroscan_output(tmp_path):
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
    theta_units = getattr(output["phi"].theta.data, "units", None)
    if theta_units is not None:
        assert result.data.units == theta_units
        assert bounds.data.units == theta_units
