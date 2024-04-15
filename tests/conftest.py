"""
Defines fixtures to be used throughout the test suite.
"""

import numpy as np
import pytest

from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.normalisation import SimulationNormalisation

@pytest.fixture(scope="session")
def generate_miller():
    """
    Create a LocalGeometryMiller.

    Used throughout tests/local_geometry.
    """

    def generate(theta, Rmaj=3.0, rho=0.5, kappa=1.0, delta=0.0, Z0=0.0, dict={}):
        miller = LocalGeometryMiller()

        miller.Rmaj = Rmaj
        miller.Z0 = Z0
        miller.rho = rho
        miller.kappa = kappa
        miller.delta = delta
        miller.dpsidr = 1.0
        miller.shift = 0.0
        miller.theta = theta

        if dict:
            for key, value in dict.items():
                miller[key] = value

        norms = SimulationNormalisation("generate_miller")
        miller.normalise(norms)

        miller.R_eq, miller.Z_eq = miller.get_flux_surface(theta)
        miller.R = miller.R_eq
        miller.Z = miller.Z_eq

        miller.b_poloidal_eq = miller.get_b_poloidal(
            theta=miller.theta,
        )
        (
            miller.dRdtheta,
            miller.dRdr,
            miller.dZdtheta,
            miller.dZdr,
        ) = miller.get_RZ_derivatives(miller.theta)

        return miller

    return generate


@pytest.fixture(scope="session")
def array_similar():
    """
    Ensure arrays are similar, after squeezing dimensions of len 1 and (potentially)
    replacing nans with zeros. Transposes both to same coords.

    Used throughout tests/gk_code
    """

    def test_arrays(x, y, nan_to_zero: bool = False) -> bool:
        # Deal with changed nans
        if nan_to_zero:
            x, y = np.nan_to_num(x), np.nan_to_num(y)
        # Squeeze out any dims of size 1
        x, y = x.squeeze(drop=True), y.squeeze(drop=True)
        # transpose both to the same shape
        # only transpose the coords that exist in both
        coords = x.coords
        x, y = x.transpose(*coords), y.transpose(*coords)
        return np.allclose(x, y)

    return test_arrays
