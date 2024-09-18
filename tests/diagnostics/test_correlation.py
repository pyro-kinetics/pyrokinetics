import numpy as np
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
    dky = pyro.gk_output["ky"][1]
    yarray = np.linspace(-np.pi / dky, np.pi / dky, 3)
    corr = diag.compute_corr_length(1, yarray, Nx=20, ndelta=10)
    filename = Path(__file__).parent / "golden_answers" / "correlation.npy"
    data = np.asarray(np.load(filename))
    assert_allclose(corr, data, rtol=1e-5, atol=1e-8)
