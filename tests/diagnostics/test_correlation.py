from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.field_line import FieldLine


def test_correlation():
    pyro = Pyro(
        gk_file=template_dir / "outputs" / "CGYRO_nonlinear" / "input.cgyro",
        gk_code="CGYRO",
    )
    pyro.load_gk_output()
    diag = FieldLine(pyro)
    corr = diag.parallel_correlation_length(time=1.0).m
    filename = Path(__file__).parent / "golden_answers" / "correlation.npy"
    data = np.asarray(np.load(filename))
    assert_allclose(corr, data, rtol=1e-5, atol=1e-8)
