import numpy as np
import pytest
from numpy.testing import assert_allclose
from pathlib import Path

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics


def call_poincare(pyro):
    xarray = np.linspace(6, 8, 5)
    yarray = np.linspace(-10, 10, 3)
    nturns = 1000
    time = 1
    rhos = 0.036
    diag = Diagnostics(pyro)
    coords = diag.poincare(xarray, yarray, nturns, time, rhos)
    return coords


def test_linear_poincare():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_linear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    with pytest.raises(RuntimeError):
        call_poincare(pyro)


def test_poincare():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_nonlinear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    coords = call_poincare(pyro)
    filename = Path(__file__).parent / "golden_answers" / "poincare.npy"
    data = np.asarray(np.load(filename))
    assert_allclose(coords, data, rtol=1e-5, atol=1e-8)
