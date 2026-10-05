"""GENE's Miller geometry in Python against miller.dat files written by GENE itself (nominal
STEP deck and three one-parameter variations of it, nz0 = 512)."""

from pathlib import Path

import f90nml
import numpy as np
import pytest

from pyrokinetics.hkbm_solver.geometry import Geo
from pyrokinetics.hkbm_solver.miller import (
    COLUMNS,
    lag3interp,
    miller_from_namelist,
    pressure_terms,
    read_miller_dat,
    write_miller_dat,
)

DATA = Path(__file__).parent / "data"
# nominal STEP deck and one-parameter variations (GENE miller.dat from geometry-only GENE runs),
# reversed current and field, squareness, and a pyrokinetics GS2 -> GENE conversion (sign_Ip_CW = sign_Bt_CW = -1)
DECKS = [
    "G_alpha1",
    "G_kappa2",
    "G_delta0.6",
    "G_shat1.6",
    "STEP_ipm1_btm1",
    "STEP_zeta0.1",
    "GS2tmpl",
]


@pytest.mark.parametrize("deck", DECKS)
def test_miller_matches_gene(deck):
    nml = f90nml.read(DATA / deck / "parameters")
    h, ref = read_miller_dat(DATA / deck / "miller.dat")
    g = miller_from_namelist(nml, nz0=int(h["gridpoints"]))
    lref = h["Lref"] if h["Lref"] > 0 else 1.0
    for c in COLUMNS:
        v = g[c] * (lref if c in ("R", "Z") else 1.0)
        scale = np.abs(ref[c]).max()
        if scale == 0:
            assert np.all(v == 0)
            continue
        assert np.abs(v - ref[c]).max() / scale < 1e-10, c
    assert g["C_y"] == pytest.approx(h["Cy"], rel=1e-12)
    assert g["C_xy"] == pytest.approx(h["Cxy"], rel=1e-12)
    assert g["my_dpdx"] == pytest.approx(h["my_dpdx"], rel=1e-12)
    assert g["trpeps"] == pytest.approx(h["trpeps"], rel=1e-12)
    assert g["x0"] ** 2 == pytest.approx(h["s0"], rel=1e-12)


@pytest.mark.parametrize("deck", DECKS[:1])
def test_curvature_and_roundtrip(deck, tmp_path):
    nml = f90nml.read(DATA / deck / "parameters")
    g = miller_from_namelist(nml, nz0=512)
    write_miller_dat(g, tmp_path / "miller.dat")
    a = Geo.from_file(DATA / deck / "miller.dat")
    b = Geo.from_file(tmp_path / "miller.dat")
    c = Geo.from_miller(g)
    for x in (b, c):
        assert np.allclose(x.Ky, a.Ky, rtol=0, atol=1e-11 * np.abs(a.Ky).max())
        assert x.dpdx == pytest.approx(a.dpdx, rel=1e-14)
        assert np.allclose(x.B, a.B, rtol=1e-13)


def test_pressure_terms_step():
    """amhd = dpdx_pm = -1: GENE derives both from beta and the gradients."""
    nml = f90nml.read(DATA / "step_ky0.2" / "parameters")
    spec = [{k.lower(): v for k, v in s.items()} for s in nml["species"]]
    geo = {k.lower(): v for k, v in nml["geometry"].items()}
    amhd, dpdx = pressure_terms(geo, spec, float(nml["general"]["beta"]))
    assert amhd == pytest.approx(11.982455, rel=1e-7)
    assert dpdx == pytest.approx(0.4985523, rel=1e-7)


def test_lag3interp_cubic_exact():
    x = np.linspace(-1, 2, 40)
    xo = np.linspace(-1.05, 2.1, 77)  # includes extrapolation at both ends
    y = 1 - 2 * x + 0.5 * x**2 - 0.3 * x**3
    yo = 1 - 2 * xo + 0.5 * xo**2 - 0.3 * xo**3
    assert np.allclose(lag3interp(y, x, xo), yo, atol=1e-12)
    assert np.allclose(lag3interp(y[::-1], x[::-1], xo), yo, atol=1e-12)
