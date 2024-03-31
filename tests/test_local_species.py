import itertools
from typing import Dict, List

from pyrokinetics.local_species import LocalSpecies
from pyrokinetics.normalisation import ureg as units

import numpy as np
import pytest

_UNITS = dict(
    mass=units.mref_deuterium,
    z=units.elementary_charge,
    dens=units.nref_electron,
    temp=units.tref_electron,
    omega0=units.vref_most_probable / units.lref_minor_radius,
    nu=units.vref_most_probable / units.lref_minor_radius,
    inverse_lt=units.lref_minor_radius**-1,
    inverse_ln=units.lref_minor_radius**-1,
    domega_drho=units.vref_most_probable / units.lref_minor_radius**2,
)


def _species_data(**kwargs: float) -> Dict[str, float]:
    """Make a dict of ``species_data`` with a default value of ``1.0`` in all fields."""
    result = {key: 1.0 * unit for key, unit in _UNITS.items()}
    changes = {key: value * _UNITS[key] for key, value in kwargs.items()}
    result.update(changes)
    return result


@pytest.fixture
def simple_local_species() -> LocalSpecies:
    """Creates a ``LocalSpecies`` with simple data to test merge functions."""
    local_species = LocalSpecies()
    local_species.add_species(
        name="electron",
        species_data=_species_data(mass=2.5e-4, z=-1.0, dens=1.0, inverse_ln=8.0 / 3.0),
    )
    local_species.add_species(
        name="deuterium",
        species_data=_species_data(mass=1.0, z=1.0, dens=2.0 / 3.0, inverse_ln=3.0),
    )
    local_species.add_species(
        name="carbon12",
        species_data=_species_data(mass=6.0, z=6.0, dens=1.0 / 27.0, inverse_ln=2.0),
    )
    local_species.add_species(
        name="carbon13",
        species_data=_species_data(mass=6.5, z=6.0, dens=1.0 / 54.0, inverse_ln=2.0),
    )
    local_species.update_pressure()
    assert local_species.check_quasineutrality()
    return local_species


@pytest.mark.parametrize(
    "merge_species,keep_mass,keep_z",
    itertools.product(
        (["carbon13"], ["carbon12", "carbon13"], ["carbon13", "carbon13"]),
        (False, True),
        (False, True),
    ),
)
def test_merge_isotopes(
    simple_local_species: LocalSpecies,
    merge_species: List[str],
    keep_mass: bool,
    keep_z: bool,
):
    """Test that merging two impurity isotopes produces a valid ``LocalSpecies``."""
    pressure = simple_local_species.pressure
    simple_local_species.merge_species(
        "carbon12",
        merge_species,
        keep_base_species_mass=keep_mass,
        keep_base_species_z=keep_z,
    )
    assert "carbon12" in simple_local_species
    assert "carbon13" not in simple_local_species
    assert simple_local_species.check_quasineutrality()
    np.testing.assert_allclose(pressure, simple_local_species.pressure)
    if keep_mass:
        np.testing.assert_allclose(simple_local_species["carbon12"].mass.magnitude, 6.0)
    else:
        np.testing.assert_allclose(
            simple_local_species["carbon12"].mass.magnitude, 37.0 / 6.0
        )
    np.testing.assert_allclose(simple_local_species["carbon12"].z.magnitude, 6.0)
    np.testing.assert_allclose(
        simple_local_species["carbon12"].dens.magnitude, 1.0 / 18.0
    )


def test_merge_empty_list(simple_local_species: LocalSpecies):
    """Test that merging an empty list of species doesn't alter anything."""
    pressure = simple_local_species.pressure
    simple_local_species.merge_species("carbon12", [])
    assert "carbon12" in simple_local_species
    assert "carbon13" in simple_local_species
    assert simple_local_species.check_quasineutrality()
    np.testing.assert_allclose(pressure, simple_local_species.pressure)
    np.testing.assert_allclose(simple_local_species["carbon12"].mass.magnitude, 6.0)
    np.testing.assert_allclose(simple_local_species["carbon12"].z.magnitude, 6.0)
    np.testing.assert_allclose(
        simple_local_species["carbon12"].dens.magnitude, 1.0 / 27.0
    )


def test_merge_bad_base_species(simple_local_species: LocalSpecies):
    """Test that an incorrect ``base_species`` raises an appropriate exception."""
    with pytest.raises(ValueError) as exc:
        simple_local_species.merge_species("uranium235", ["carbon13"])
    assert "base_species" in str(exc)


def test_merge_bad_merge_species(simple_local_species: LocalSpecies):
    """Test that an incorrect ``base_species`` raises an appropriate exception."""
    with pytest.raises(ValueError) as exc:
        simple_local_species.merge_species("carbon12", ["muon"])
    assert "merge_species" in str(exc)


@pytest.mark.parametrize(
    "merge_species,keep_mass,keep_z",
    itertools.product(
        (["carbon12"],),
        (False, True),
        (False, True),
    ),
)
def test_merge_fuel_impurity(
    simple_local_species: LocalSpecies,
    merge_species: List[str],
    keep_mass: bool,
    keep_z: bool,
):
    """Test that merging two species produces a valid ``LocalSpecies``."""
    pressure = simple_local_species.pressure
    simple_local_species.merge_species(
        "deuterium",
        merge_species,
        keep_base_species_mass=keep_mass,
        keep_base_species_z=keep_z,
    )
    assert "deuterium" in simple_local_species
    assert "carbon12" not in simple_local_species
    assert simple_local_species.check_quasineutrality()
    np.testing.assert_allclose(
        simple_local_species["deuterium"].inverse_ln.magnitude, 2.75
    )
    if not keep_z:
        np.testing.assert_allclose(pressure, simple_local_species.pressure)
    if keep_z:
        np.testing.assert_allclose(simple_local_species["deuterium"].z.magnitude, 1.0)
    else:
        np.testing.assert_allclose(
            simple_local_species["deuterium"].z.magnitude,
            (2.0 / 3 + 6.0 / 27) / (2.0 / 3 + 1.0 / 27),
        )
