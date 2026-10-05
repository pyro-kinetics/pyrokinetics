"""hKBM solver with GENE input and output: STEP regression, deck checks, and loading the output
with pyrokinetics' GENE reader (directly and after a GS2/CGYRO round trip).  Each solve takes
~10-20 s."""

import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyrokinetics import Pyro, template_dir
from pyrokinetics.hkbm_solver import UnsupportedDeck, run
from pyrokinetics.hkbm_solver.gene_io import Deck

DATA = Path(__file__).parent / "data"

# Main-model roots of the STEP hKBM (nominal collisionality, coll = 1.35e-4), computed with the
# original solver on GENE's own miller.dat (G_alpha1): GENE sign, c_s/a.
STEP_REF = {
    0.2: (0.07914628329101277, 0.0644766528794436),
    0.3: (0.19255437691516514, 0.09727226100726354),
}


def _m(da):
    """Magnitudes of a (possibly unit-carrying) DataArray as a flat ndarray."""
    d = da.data
    return np.asarray(getattr(d, "magnitude", d)).ravel()


def _deck(tmp_path, src, subs=(), name="run"):
    d = tmp_path / name
    d.mkdir()
    txt = (DATA / src / "parameters").read_text()
    for a, b in subs:
        txt, n = re.subn(a, b, txt)
        assert n, a
    (d / "parameters").write_text(txt)
    return d


def _step(tmp_path, ky, name="step"):
    """The STEP deck of the geometry scan (explicit amhd, dpdx_pm) at nominal collisionality."""
    return _deck(
        tmp_path,
        "G_alpha1",
        [
            (r"kymin = 0.3", f"kymin = {ky}"),
            (r"coll = 0.0", "coll = 0.000135"),
            (r"diagdir = './out'", "diagdir = './'"),
        ],
        name,
    )


@pytest.mark.parametrize("ky", [0.2, 0.3])
def test_step_regression(tmp_path, ky):
    d = _step(tmp_path, ky)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r,) = run(d / "parameters")
    assert r["converged"]
    om, gam = STEP_REF[ky]
    assert r["omega"] == pytest.approx(om, abs=1e-8)
    assert r["gamma"] == pytest.approx(gam, abs=1e-8)
    info = json.loads((d / "hkbm.json").read_text())
    assert info["gamma"] == pytest.approx(r["gamma"], rel=1e-14)
    for f in ("parameters.dat", "omega.dat", "field.dat", "nrg.dat", "miller.dat"):
        assert (d / f).is_file(), f


def test_pyro_reads_output(tmp_path):
    d = _step(tmp_path, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r,) = run(d / "parameters")
        pyro = Pyro(gk_file=d / "parameters")
        pyro.load_gk_output()
    out = pyro.gk_output
    gr = float(_m(out["growth_rate"].isel(time=-1))[0])
    mf = float(_m(out["mode_frequency"].isel(time=-1))[0])
    # pyrokinetics converts to its own normalisation (here identical up to its reference mass)
    assert gr == pytest.approx(r["gamma"], rel=1e-5)
    assert mf == pytest.approx(r["omega"], rel=1e-5)
    ef = np.asarray(
        getattr(
            out["eigenfunctions"].isel(time=-1).data,
            "magnitude",
            out["eigenfunctions"].isel(time=-1).data,
        )
    )
    theta = _m(out["theta"])
    i0 = np.argmin(np.abs(theta))
    phi, apar, bpar = (ef[i, ...].ravel() for i in range(3))
    assert np.argmax(np.abs(phi)) == i0  # ballooning: phi peaks at theta = 0
    assert abs(apar[i0]) < 1e-8 * abs(phi[i0])  # twisting parity: A_par odd
    # delta B_par / phi at theta = 0 as the solver has it (pyrokinetics conjugates GENE's fields)
    res = r["result"]
    j0 = np.argmin(np.abs(res["theta"]))
    assert np.conj(bpar[i0] / phi[i0]) == pytest.approx(
        res["bpar"][j0] / res["phi"][j0], rel=1e-3
    )
    # the omega file path of the reader
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyro.load_gk_output(load_fields=False)
    assert float(_m(pyro.gk_output["growth_rate"])[-1]) == pytest.approx(
        r["gamma"], abs=1e-4
    )


@pytest.mark.parametrize("code", ["GS2", "CGYRO"])
def test_roundtrip_step_via_other_code(tmp_path, code):
    """GENE STEP deck -> pyrokinetics -> GS2/CGYRO input -> pyrokinetics -> GENE deck -> solver."""
    d0 = _step(tmp_path, 0.2, "direct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r0,) = run(d0 / "parameters")
        p = Pyro(gk_file=d0 / "parameters", gk_code="GENE")
        p.convert_gk_code(code)
        fn = tmp_path / code / f"input.{code.lower()}"
        p.write_gk_file(fn, gk_code=code)
        p2 = Pyro(gk_file=fn, gk_code=code)
        p2.convert_gk_code("GENE")
        d1 = tmp_path / f"via_{code}"
        p2.write_gk_file(d1 / "parameters", gk_code="GENE")
        (r1,) = run(d1 / "parameters")
        pyro = Pyro(gk_file=d1 / "parameters")
        pyro.load_gk_output()
    assert r1["converged"]
    assert r1["gamma"] == pytest.approx(r0["gamma"], rel=1e-6)
    assert r1["omega"] == pytest.approx(r0["omega"], rel=1e-6)
    gr = float(_m(pyro.gk_output["growth_rate"].isel(time=-1))[0])
    assert gr == pytest.approx(r0["gamma"], rel=1e-5)


def test_roundtrip_gs2_template(tmp_path):
    """Felix's use case: a GS2 input (pyrokinetics template, Cyclone-like, two species) converted
    to GENE, solved, and loaded back with pyrokinetics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Pyro(gk_file=template_dir / "input.gs2")
        p.convert_gk_code("GENE")
        d = tmp_path / "gs2"
        p.write_gk_file(d / "parameters", gk_code="GENE")
        (r,) = run(d / "parameters", scan=False)
        pyro = Pyro(gk_file=d / "parameters")
        pyro.load_gk_output()
    assert r["converged"]
    gr = float(_m(pyro.gk_output["growth_rate"].isel(time=-1))[0])
    assert gr == pytest.approx(r["gamma"], rel=1e-5)


def test_sign_conventions_invariant(tmp_path):
    """Reversing the current and the field (sign_Ip_CW = sign_Bt_CW = -1) leaves the eigenvalue unchanged."""
    d = _deck(
        tmp_path,
        "STEP_ipm1_btm1",
        [
            (r"kymin = 0.3", "kymin = 0.2"),
            (r"coll = 0.0", "coll = 0.000135"),
            (r"diagdir = './out'", "diagdir = './'"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (r,) = run(d / "parameters")
    assert r["converged"]
    # this deck has amhd = dpdx_pm = -1 (GENE resolves them to 1e-7 of the scan deck's values)
    assert r["gamma"] == pytest.approx(STEP_REF[0.2][1], rel=1e-6)
    assert r["omega"] == pytest.approx(STEP_REF[0.2][0], rel=1e-6)


def test_ky_scan_files(tmp_path):
    d = _deck(
        tmp_path,
        "G_alpha1",
        [
            (r"kymin = 0.3", "kymin = 0.2 !scanlist: 0.2, 0.25"),
            (r"coll = 0.0", "coll = 0.000135"),
        ],
    )
    deck = Deck(d / "parameters")
    assert deck.kys == [0.2, 0.25]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run(d / "parameters", outdir=tmp_path / "out")
    assert [r["suffix"] for r in res] == ["_0001", "_0002"]
    assert all(r["converged"] for r in res)
    for f in ("parameters_0002", "omega_0002", "field_0002", "nrg_0002", "scan.log"):
        assert (tmp_path / "out" / f).is_file()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyro = Pyro(gk_file=tmp_path / "out" / "parameters_0002")
        pyro.load_gk_output()
    gr = float(_m(pyro.gk_output["growth_rate"].isel(time=-1))[0])
    assert gr == pytest.approx(res[1]["gamma"], rel=1e-5)


@pytest.mark.parametrize(
    "subs,msg",
    [
        ([(r"nonlinear = .false.", "nonlinear = .true.")], "nonlinear"),
        ([(r"beta = 0.09", "beta = 0.0")], "beta"),
        ([(r"magn_geometry = 'miller'", "magn_geometry = 's_alpha'")], "miller"),
        ([(r"kx_center = 0.0", "kx_center = 0.1")], "kx_center"),
        ([(r"omn = 1.027\n    omt = 1.822", "omn = 1.5\n    omt = 1.822")], "omn"),
    ],
)
def test_unsupported_decks(tmp_path, subs, msg):
    d = _deck(tmp_path, "G_alpha1", subs)
    with pytest.raises(UnsupportedDeck, match=msg):
        Deck(d / "parameters")


def test_impurity_deck_stops(tmp_path):
    d = _deck(tmp_path, "G_alpha1", [(r"n_spec = 2", "n_spec = 3")])
    txt = (d / "parameters").read_text()
    imp = (
        "\n&species\n    charge = 6\n    dens = 0.01\n    mass = 6.0\n    name = 'carbon'\n"
        "    omn = 1.0\n    omt = 1.8\n    temp = 1.03\n/\n"
    )
    (d / "parameters").write_text(txt + imp)
    with pytest.raises(UnsupportedDeck, match="species"):
        Deck(d / "parameters")
    shutil.rmtree(d)
