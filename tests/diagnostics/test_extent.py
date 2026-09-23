import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.dataset_wrapper import DatasetWrapper
from pyrokinetics.diagnostics.parity import Parity
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics.units import ureg


def test_real_gk_output_and_netcdf_roundtrip(tmp_path):
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    output = pyro.gk_output
    original_data = output.data.copy(deep=True)

    Parity(output, field="phi", center=0.2)
    result, bounds = output["parity"]

    assert pyro.gk_output is output
    xr.testing.assert_identical(output.data.drop_vars("parity"), original_data)
    assert result.pint.units == ureg.dimensionless
    assert "theta" not in result.dims
    assert result.attrs["source_variable"] == "phi"
