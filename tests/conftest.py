"""
Defines fixtures to be used throughout the test suite.
"""

from functools import cache
from typing import Optional

import numpy as np
import pytest

from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.gk_code import GKInput
from pyrokinetics.kinetics import Kinetics
from pyrokinetics.local_geometry import LocalGeometryMiller
from pyrokinetics.normalisation import SimulationNormalisation
from pyrokinetics.typing import PathLike
from pyrokinetics.units import ureg


@pytest.fixture(scope="session")
def generate_miller():
    """
    Create a LocalGeometryMiller.

    Used throughout tests/local_geometry.
    """

    def generate(theta, Rmaj=3.0, rho=0.5, kappa=1.0, delta=0.0, Z0=0.0, dict={}):
        data = dict.copy()
        inputs = {
            "Rmaj": Rmaj,
            "rho": rho,
            "kappa": kappa,
            "delta": delta,
            "Z0": Z0,
            "shift": 0.0,
            "dpsidr": 1.0,
        }
        for key, val in inputs.items():
            if key not in data:
                data[key] = val
        miller = LocalGeometryMiller(
            theta=theta,
            **data,
        )

        norms = SimulationNormalisation("generate_miller")
        miller.normalise(norms)
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
        # Ensure x and y have the same units, then strip them
        x, y = ureg.Quantity(x.data), ureg.Quantity(y.data)
        y = y.to(x.units)
        x, y = x.magnitude, y.magnitude
        return np.allclose(x, y)

    return test_arrays


_eq_from_file = Equilibrium.from_file.__func__


@classmethod
@cache
def _read_equilibrium_cache(
    cls, path: PathLike, file_type: Optional[str] = None, **kwargs
) -> Equilibrium:
    """Alternative to ``Equilibrium.from_file``.

    Avoids repeatedly building ``Equilibrium`` instances that we've seen before"""
    return _eq_from_file(cls, path, file_type=file_type, **kwargs)


@pytest.fixture(autouse=True)
def _cache_equilibria(monkeypatch):
    """Avoids repeatedly building ``Equilibrium`` instances throughout the tests"""
    monkeypatch.setattr(Equilibrium, "from_file", _read_equilibrium_cache)


_k_from_file = Kinetics.from_file.__func__


@classmethod
@cache
def _read_kinetics_cache(
    cls, path: PathLike, file_type: Optional[str] = None, **kwargs
) -> Kinetics:
    """Alternative to ``Kinetics.from_file``.

    Avoids repeatedly building ``Kinetics`` instances that we've seen before"""
    return _k_from_file(cls, path, file_type=file_type, **kwargs)


@pytest.fixture(autouse=True)
def _cache_kinetics(monkeypatch):
    """Avoids repeatedly building ``Kinetics`` instances throughout the tests"""
    monkeypatch.setattr(Kinetics, "from_file", _read_kinetics_cache)


_gk_input_from_file = GKInput.from_file.__func__


@classmethod
@cache
def _read_gk_input_cache(
    cls, path: PathLike, file_type: Optional[str] = None, **kwargs
) -> GKInput:
    """Alternative to ``GKInput.from_file``.

    Avoids repeatedly building ``GKInput`` instances that we've seen before"""
    return _gk_input_from_file(cls, path, file_type=file_type, **kwargs)


@pytest.fixture(autouse=True)
def _cache_gk_input(monkeypatch):
    """Avoids repeatedly building ``GKInput`` instances throughout the tests"""
    monkeypatch.setattr(GKInput, "from_file", _read_gk_input_cache)
