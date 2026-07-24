from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.field_line import FieldLine


def test_displacement():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_nonlinear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    diag = FieldLine(pyro)
    dky = pyro.gk_output["ky"].values[1]
    xarray = np.array([0.0])
    yarray = np.linspace(-np.pi / dky, np.pi / dky, 3)
    time = 1.0
    rhostar = 0.036
    disp = diag.compute_half_displacement(xarray, yarray, time, rhostar).m
    filename = Path(__file__).parent / "golden_answers" / "displacement.npy"
    data = np.asarray(np.load(filename))
    assert_allclose(disp, data, rtol=1e-5, atol=1e-8)
