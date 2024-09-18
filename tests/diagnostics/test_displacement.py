import numpy as np
import pytest
from numpy.testing import assert_allclose
from pathlib import Path

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics


def test_displacement():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_nonlinear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    diag = Diagnostics(pyro)
    disp = diag.compute_half_disp(1)
    filename = Path(__file__).parent / "golden_answers" / "displacement.npy"
    data = np.asarray(np.load(filename))
    assert_allclose(disp, data, rtol=1e-5, atol=1e-8)
