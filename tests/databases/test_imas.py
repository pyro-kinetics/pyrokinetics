from pyrokinetics import Pyro, template_dir

from pyrokinetics.databases.imas import pyro_to_ids, ids_to_pyro

import pytest
import numpy as np
import os


def array_similar(x, y, nan_to_zero: bool = False) -> bool:
    """
    Ensure arrays are similar, after squeezing dimensions of len 1 and (potentially)
    replacing nans with zeros. Transposes both to same coords.
    """
    # Deal with changed nans
    if nan_to_zero:
        x, y = np.nan_to_num(x), np.nan_to_num(y)
    # Squeeze out any dims of size 1
    x, y = (
        x.squeeze(drop=True).pint.dequantify(),
        y.squeeze(drop=True).pint.dequantify(),
    )

    return np.allclose(x, y)


@pytest.mark.parametrize(
    "input_path",
    [
        template_dir / "outputs" / "GENE_linear" / "parameters_0001",
        template_dir / "outputs" / "GS2_linear" / "gs2.in",
        template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
    ],
)
@pytest.mark.skipif(True, reason="skipping until IMAS properly merged")
def test_pyro_to_imas_roundtrip(tmp_path, input_path):
    pyro = Pyro(gk_file=input_path)
    pyro.load_gk_output()

    gk_code = pyro.gk_code

    ref_dict = {
        "tref": 1000.0,
        "nref": 1e19,
        "lref": 1.5,
        "bref": 2.0,
    }

    hdf5_file_name = tmp_path / f"test_{gk_code}.hdf5"

    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)

    # Write IDS file
    pyro_to_ids(
        pyro,
        comment=f"Testing IMAS {gk_code.upper()}",
        name="testing",
        ref_dict=ref_dict,
        format="hdf5",
        file_name=hdf5_file_name,
    )

    # Ensure IDS was written
    assert os.path.exists(hdf5_file_name)

    new_pyro = ids_to_pyro(hdf5_file_name)

    old_gk_output = pyro.gk_output
    new_gk_output = new_pyro.gk_output

    # Test data
    final_time_only = ["particle", "heat", "momentum", "eigenfucntions"]

    for data_var in old_gk_output.data_vars:
        if data_var in final_time_only:
            assert array_similar(
                old_gk_output[data_var].isel(time=-1),
                new_gk_output[data_var].isel(time=-1),
            )
        else:
            assert array_similar(old_gk_output[data_var], new_gk_output[data_var])

    # Test coords
    skip_coords = ["energy", "pitch"]

    for c in old_gk_output.coords:
        if c in skip_coords:
            continue
        dtype = old_gk_output[c].dtype
        if dtype == "float64" or dtype == "complex128":
            assert array_similar(old_gk_output[c], new_gk_output[c])
        else:
            assert np.array_equal(old_gk_output[c], new_gk_output[c])
