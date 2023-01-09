from pyrokinetics.species import Species

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def test_species_mass():
    test_mass = 4.4
    species = Species(mass=test_mass)

    assert species.get_mass() == test_mass


def test_species_charge():
    test_charge = -3.3
    species = Species(charge=test_charge)

    assert species.get_charge() == test_charge


def test_density():
    psi = np.linspace(0.0, 1.0)
    density_data = 5.0 - 5.0 * (psi**2)
    density_func = InterpolatedUnivariateSpline(psi, density_data)

    species = Species(dens=density_func)

    assert np.isclose(species.get_dens(0.5), 3.75)


def test_density_gradient():
    psi = np.linspace(0.0, 1.0)
    rho_func = InterpolatedUnivariateSpline(psi, psi**2)
    density_data = 5.0 - 5.0 * (psi**2)
    density_func = InterpolatedUnivariateSpline(psi, density_data)

    species = Species(dens=density_func, rho=rho_func)

    assert np.isclose(species.get_norm_dens_gradient(0.5), 4.0 / 3.0)


def test_temperature():
    psi = np.linspace(0.0, 1.0)
    temperature_data = 4.0 - 4.0 * (psi**2)
    temperature_func = InterpolatedUnivariateSpline(psi, temperature_data)

    species = Species(temp=temperature_func)

    assert np.isclose(species.get_temp(0.5), 3.0)


def test_temperature_gradient():
    psi = np.linspace(0.0, 1.0)
    rho_func = InterpolatedUnivariateSpline(psi, psi**2)
    temperature_data = 4.0 - 4.0 * (psi**2)
    temperature_func = InterpolatedUnivariateSpline(psi, temperature_data)

    species = Species(temp=temperature_func, rho=rho_func)

    assert np.isclose(species.get_norm_temp_gradient(0.5), 4.0 / 3.0)


def test_rotation():
    psi = np.linspace(0.0, 1.0)
    rotation_data = 3.0 - 3.0 * (psi**2)
    rotation_func = InterpolatedUnivariateSpline(psi, rotation_data)

    species = Species(rot=rotation_func)

    assert np.isclose(species.get_velocity(0.5), 2.25)


def test_rotation_gradient():
    psi = np.linspace(0.0, 1.0)
    rho_func = InterpolatedUnivariateSpline(psi, psi**2)
    rotation_data = 3.0 - 3.0 * (psi**2)
    rotation_func = InterpolatedUnivariateSpline(psi, rotation_data)

    species = Species(rot=rotation_func, rho=rho_func)

    assert np.isclose(species.get_norm_vel_gradient(0.5), 4.0 / 3.0)


def test_no_rotation():
    species = Species()
    assert np.isclose(species.get_velocity(0.5), 0.0)


def test_no_rotation_gradient():
    species = Species()
    assert np.isclose(species.get_norm_vel_gradient(0.5), 0.0)
