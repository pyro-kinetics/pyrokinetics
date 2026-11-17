import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.diagnostics.field_line import FieldLine


@pytest.fixture
def shaped_gs2(tmp_path):
    # GS2 template with elongation, triangularity and Shafranov shift (the templates are circular)
    text = (template_dir / "input.gs2").read_text()
    for old, new in (
        ("akappa = 1.0", "akappa = 1.8"),
        ("tri = 0.0", "tri = 0.4"),
        ("shift = 0.0", "shift = -0.2"),
    ):
        assert old in text
        text = text.replace(old, new)
    (tmp_path / "input.gs2").write_text(text)
    return Pyro(gk_file=tmp_path / "input.gs2")


def test_basis_angle_circular():
    # Circular Miller: B_p R is constant on the surface, so dl/dtheta is constant
    # and theta_b = theta exactly
    pyro = Pyro(gk_file=template_dir / "input.cgyro", gk_code="CGYRO")
    theta = np.linspace(-7 * np.pi, 7 * np.pi, 301)
    np.testing.assert_allclose(
        FieldLine(pyro).compute_basis_angle(theta), theta, atol=1e-10
    )


def test_basis_angle_periodic(shaped_gs2):
    field_line = FieldLine(shaped_gs2)
    theta = np.linspace(-np.pi, np.pi, 101)
    basis = field_line.compute_basis_angle(theta)
    assert np.abs(basis - theta).max() > 0.05  # shaping does move it
    assert field_line.compute_basis_angle([0.0])[0] == pytest.approx(0, abs=1e-12)
    assert basis[-1] == pytest.approx(np.pi, abs=1e-6)  # up-down symmetric
    assert np.all(np.diff(basis) > 0)
    for turns in (-3, 1, 5):
        np.testing.assert_allclose(
            field_line.compute_basis_angle(theta + 2 * np.pi * turns),
            basis + 2 * np.pi * turns,
            atol=1e-10,
        )


def test_basis_angle_matches_poloidal_arc_length(shaped_gs2):
    # Independent route through the Miller surface: dy/dtheta = (B/B_p) dl_p/dtheta
    geometry = shaped_gs2.local_geometry
    theta = np.linspace(0, 2 * np.pi, 4001)
    major_radius = geometry.get_flux_surface(theta)[0]
    b_poloidal = geometry.get_b_poloidal(theta)
    b_over_b_poloidal = np.sqrt(1 + (geometry.Fpsi / major_radius / b_poloidal) ** 2)
    integrand = np.asarray((b_over_b_poloidal * geometry.get_dLdtheta(theta)).m)
    arc = np.concatenate(
        [[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(theta))]
    )
    expected = 2 * np.pi * arc / arc[-1]
    np.testing.assert_allclose(
        FieldLine(shaped_gs2).compute_basis_angle(theta), expected, atol=1e-5
    )
