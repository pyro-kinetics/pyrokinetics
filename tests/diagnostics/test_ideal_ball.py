import numpy as np
import pytest
from numpy.testing import assert_allclose
from pathlib import Path

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics import Diagnostics


@pytest.fixture(scope="module")
def gamma_benchmark():
    return np.load(Path(__file__).parent / "golden_answers/ideal_ball.npy")


def test_linear_poincare(gamma_benchmark):
    nshat = 5
    nbprime = 5
    shat = np.linspace(0.0, 2, nshat)
    bprime = np.linspace(0.0, -0.5, nbprime)

    pyro = Pyro(gk_file=template_dir / "input.gs2")

    gamma = np.empty((nshat, nbprime))

    for i_s, s in enumerate(shat):
        for i_b, b in enumerate(bprime):
            pyro.local_geometry.shat = s
            pyro.local_geometry.beta_prime = b
            diag = Diagnostics(pyro)
            gamma[i_s, i_b] = diag.ideal_ballooning_solver()

    assert_allclose(gamma_benchmark, gamma, rtol=1e-3, atol=1e-5)
