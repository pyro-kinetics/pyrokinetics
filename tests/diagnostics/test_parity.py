import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.dataset_wrapper import DatasetWrapper
from pyrokinetics.diagnostics.parity import Parity
from pyrokinetics.gk_code.gk_output import GKOutput
from pyrokinetics.units import ureg


def make_gk_output(field):
    output = GKOutput.__new__(GKOutput)
    DatasetWrapper.__init__(output, data_vars={"phi": field})
    return output


@pytest.fixture
def theta():
    return np.linspace(-np.pi, np.pi, 257)


@pytest.mark.parametrize(
    "profile, expected",
    [
        (lambda theta: np.ones_like(theta), 1.0),
        (np.sin, -1.0),
        (np.cos, 1.0),
        (lambda theta: np.cos(theta) + 1j * np.sin(theta), 0.0),
    ],
)
def test_reflection_parity(theta, profile, expected):
    field = xr.DataArray(profile(theta), dims="theta", coords={"theta": theta})
    output = make_gk_output(field)

    assert Parity(output) is not output
    result = output["parity"]

    assert result.pint.units == ureg.dimensionless
    assert result.dims == ()
    assert_allclose(result.data.magnitude, expected, atol=1e-14)


@pytest.mark.parametrize("center", [-1.3, 0.4, 2.1])
@pytest.mark.parametrize("odd", [False, True])
def test_shifted_profiles(center, odd):
    offsets = np.linspace(-3, 3, 301)
    profile = np.exp(-(offsets**2))
    if odd:
        profile *= offsets
    field = xr.DataArray(
        profile * (1 + 2j), dims="theta", coords={"theta": center + offsets}
    )
    output = make_gk_output(field)

    Parity(output, center=center)
    result = output["parity"]

    assert_allclose(result.data.magnitude, -1 if odd else 1, atol=1e-12)
    assert result.attrs["reflection_center"] == center


def test_other_dimensions_are_preserved(theta):
    profiles = np.stack([np.ones_like(theta), np.sin(theta)])
    field = xr.DataArray(
        profiles[:, :, None] * np.array([1, 3j, 1e-200])[None, None, :],
        dims=("kx", "theta", "mode"),
        coords={"kx": [0.1, 0.2], "theta": theta, "mode": [0, 1, 2]},
    )
    field = field.copy(data=field.data * ureg.volt)
    output = make_gk_output(field)

    Parity(output)
    result = output["parity"]

    assert result.dims == ("kx", "mode")
    xr.testing.assert_equal(result.kx, field.kx)
    xr.testing.assert_equal(result["mode"], field["mode"])
    assert_allclose(
        result.data.magnitude, [[1, 1, 1], [-1, -1, -1]], atol=1e-14
    )


def test_recalculation_replaces_existing_result(theta):
    field = xr.DataArray(np.sin(theta), dims="theta", coords={"theta": theta})
    output = make_gk_output(field)

    Parity(output)
    assert_allclose(output["parity"].data.magnitude, -1, atol=1e-12)
    Parity(output, center=np.pi / 2)
    assert_allclose(output["parity"].data.magnitude, 1, atol=1e-12)
    assert output["parity"].attrs["reflection_center"] == np.pi / 2


def test_zero_and_missing_profiles_are_nan():
    field = xr.DataArray(
        [[0, 0, 0], [1, np.nan, 1], [1, np.inf, 1], [1, 1, 1]],
        dims=("kx", "theta"),
        coords={"theta": [-1, 0, 1]},
    )
    output = make_gk_output(field)

    Parity(output)

    assert_allclose(
        output["parity"].data.magnitude,
        [np.nan, np.nan, np.nan, 1],
        equal_nan=True,
    )


@pytest.mark.parametrize("coordinate", [[0], [0, 0, 1], [1, 0, -1], [0, np.nan, 1]])
def test_invalid_theta_coordinate(coordinate):
    field = xr.DataArray(
        np.ones(len(coordinate)), dims="theta", coords={"theta": coordinate}
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        Parity(make_gk_output(field))


@pytest.mark.parametrize("center", [np.nan, np.inf, 1j, -np.pi, np.pi, 10])
def test_invalid_center(theta, center):
    field = xr.DataArray(np.cos(theta), dims="theta", coords={"theta": theta})
    with pytest.raises(ValueError, match="center"):
        Parity(make_gk_output(field), center=center)


def test_requires_gk_output(theta):
    field = xr.DataArray(np.cos(theta), dims="theta", coords={"theta": theta})
    with pytest.raises(TypeError, match="GKOutput"):
        Parity(field)
    with pytest.raises(TypeError, match="GKOutput"):
        Parity(field.to_dataset(name="phi"))


def test_field_validation(theta):
    output = make_gk_output(
        xr.DataArray(np.cos(theta), dims="theta", coords={"theta": theta})
    )
    with pytest.raises(ValueError, match="must be one of"):
        Parity(output, field="eigenfunctions")
    with pytest.raises(ValueError, match="does not contain"):
        Parity(output, field="apar")


def test_real_gk_output_and_netcdf_roundtrip(tmp_path):
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    output = pyro.gk_output
    original_data = output.data.copy(deep=True)

    Parity(output, field="phi", center=0.2)
    result = output["parity"]

    assert pyro.gk_output is output
    xr.testing.assert_identical(output.data.drop_vars("parity"), original_data)
    assert result.pint.units == ureg.dimensionless
    assert "theta" not in result.dims
    assert result.attrs["source_variable"] == "phi"

    path = tmp_path / "gk_output_with_parity.nc"
    output.to_netcdf(path)
    restored = GKOutput.from_netcdf(path)
    xr.testing.assert_identical(restored["parity"], result)
