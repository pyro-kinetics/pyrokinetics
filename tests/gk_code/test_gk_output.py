import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyrokinetics.gk_code.gk_output import (
    Eigenfunctions,
    Eigenvalues,
    Fields,
    Fluxes,
    Moments,
)

# TODO Difficult to test units handling, not currently included.


@pytest.mark.parametrize("cls", (Fields, Fluxes, Moments, Eigenvalues, Eigenfunctions))
def test_gk_output_args_adding_dims(cls):
    """Check that default dims are successfully overwritten"""
    val = np.linspace(0.0, 10.0, 24).reshape((2, 3, 4))
    dims = ("foo", "bar", "baz")
    names = tuple(f.name for f in dataclasses.fields(cls))
    instance = cls(**{name: val for name in names}, dims=dims)
    assert instance.dims == dims


@pytest.mark.parametrize("cls", (Fields, Fluxes, Moments, Eigenvalues, Eigenfunctions))
def test_gk_output_args_iteration(cls):
    """Check that iteration produces all field names"""
    val = np.linspace(0.0, 10.0, 24).reshape((2, 3, 4))
    dims = ("foo", "bar", "baz")
    names = tuple(f.name for f in dataclasses.fields(cls))
    instance = cls(**{name: val for name in names}, dims=dims)
    # Check iter
    instance_names = tuple(x for x in instance)
    for name in names:
        assert name in instance_names
    # Check values()
    vals = tuple(x for x in instance.values())
    for v in vals:
        assert_array_equal(v, val)
    # Check items()
    items = {k: v for k, v in instance.items()}
    for k, v in items.items():
        assert k in names
        assert_array_equal(v, val)


@pytest.mark.parametrize("cls", (Fields, Fluxes, Moments, Eigenvalues, Eigenfunctions))
def test_gk_output_args_shape(cls):
    """Check that shape function returns expected value"""
    val = np.linspace(0.0, 10.0, 24).reshape((2, 3, 4))
    dims = ("foo", "bar", "baz")
    names = tuple(f.name for f in dataclasses.fields(cls))
    instance = cls(**{name: val for name in names}, dims=dims)
    assert_array_equal(instance.shape, val.shape)


@pytest.mark.parametrize("cls", (Fields, Fluxes, Moments, Eigenvalues, Eigenfunctions))
def test_gk_output_args_get_and_set(cls):
    """Check that iteration produces all field names"""
    val = np.linspace(0.0, 10.0, 24).reshape((2, 3, 4))
    dims = ("foo", "bar", "baz")
    names = tuple(f.name for f in dataclasses.fields(cls))
    instance = cls(**{name: val for name in names}, dims=dims)
    assert_array_equal(instance[names[0]], val)
    # Note: no checking is done on __setitem__
    instance[names[0]] = "hello world"
    assert instance[names[0]] == "hello world"


@pytest.mark.parametrize("cls", (Fields, Fluxes, Moments))
def test_gk_output_args_missing_fields(cls):
    """Check that only non-None fields appear in coords"""
    val = np.linspace(0.0, 10.0, 24).reshape((2, 3, 4))
    dims = ("foo", "bar", "baz")
    names = tuple(f.name for f in dataclasses.fields(cls))
    instance = cls(**{names[0]: val}, dims=dims)
    # Check coords
    assert_array_equal(instance.coords, (names[0],))
    # Check iter -- should only include the first field
    instance_names = tuple(x for x in instance)
    assert len(instance_names) == 1
    assert instance_names[0] == names[0]
    # Check values()
    vals = tuple(x for x in instance.values())
    assert len(vals) == 1
    assert_array_equal(vals[0], val)
    # Check items()
    items = {k: v for k, v in instance.items()}
    assert len(items) == 1
    assert_array_equal(items[names[0]], val)
