"""T3D contract, units and warning propagation; not MTM transport validation."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import f90nml
import numpy as np
import pytest

from pyrokinetics.hkbm_solver import mtm
from pyrokinetics.hkbm_solver import mtm_collisional_ql as Q
from pyrokinetics.hkbm_solver import quasilinear as QL
from pyrokinetics.hkbm_solver.gene_io import Deck

DATA = Path(__file__).parent / "data/step_ky0.2/parameters"


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    path = tmp_path_factory.mktemp("collisional_ql") / "parameters"
    nml = f90nml.read(DATA)
    nml["general"]["beta"] = 0.14
    nml.write(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = Deck(path)
        result = Q.solve_deck(
            deck,
            0.2848826,
            omega0=-0.52 + 0.01j,
            npt=16,
            nturns=4,
            nE=8,
            nxi=12,
            theta_order=2,
        )
    assert result["converged"]
    return path, deck, result


def test_deck_contract_units_and_no_invented_flux(example):
    _, deck, result = example
    assert result["gamma"] == pytest.approx(0.005973737657, rel=1e-5)
    assert result["omega"] == pytest.approx(-0.5210463074, rel=1e-5)
    assert result["omega_solver"].real > 0
    rec = Q._record(deck, result, 0.2848826)
    assert rec["branch"] == "mtm" and rec["backend"] == "lorentz_ei"
    assert rec["weights"] is rec["weights_solver"] is None
    assert rec["giacomin"]["Q_e_over_Q"] is None
    assert rec["giacomin"]["Q_i_over_Q"] is None
    assert rec["giacomin"]["Gamma_over_Q"] is None
    assert rec["omitted_fields"] == ("bpar",) and "bpar" not in rec
    assert not rec["validated"] and not rec["collision_physics_match"]


def test_full_domain_fields_and_independent_weighted_widths(example):
    _, deck, result = example
    rec = Q._record(deck, result, 0.2848826)
    assert abs(rec["theta"][-1]) == pytest.approx(9 * np.pi)
    for key in ("phi", "apar", "apar_theta", "kperp2", "jacobian", "bmag"):
        assert rec[key].shape == rec["theta"].shape
    np.testing.assert_array_equal(rec["apar_theta"], rec["theta"])
    assert np.max(abs(rec["phi"])) == pytest.approx(1.0)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rs = deck.units["rho_s_over_rho_ref"]
    for field in ("phi", "apar"):
        weight = abs(rec[field]) ** 2 * rec["jacobian"]
        expect = trapz(weight * rec["kperp2"], rec["theta"]) / trapz(
            weight, rec["theta"]
        )
        assert rec["kperp2_ref"][field] == pytest.approx(expect, rel=1e-12)
        assert rec["giacomin"]["kperp2"][field] == pytest.approx(expect * rs**2)


def test_field_metrics_are_normalisation_invariant(example):
    _, _, result = example
    solver = result["solver"]
    other = dict(
        result, phi=result["phi"] * (0.3 + 0.8j), apar=result["apar"] * (0.3 + 0.8j)
    )
    kp, amp = Q._field_metrics(solver, other)
    assert kp == pytest.approx(result["kperp2"])
    assert amp == pytest.approx(result["apar_over_phi"])


def test_deck_field_conversion_nonunity_units(example):
    _, deck, result = example
    units = dict(deck.units, rho_s_over_rho_ref=2.3, T_e=1.7)
    fake = SimpleNamespace(units=units)
    r = dict(result, kperp2_ref={k: v / 2.3**2 for k, v in result["kperp2"].items()})
    rec = Q._record(fake, r, 0.2)
    p0 = result["phi"][np.argmax(abs(result["phi"]))]
    np.testing.assert_allclose(rec["apar"], result["apar"] * 2.3 / (p0 * 1.7))
    np.testing.assert_allclose(rec["kperp2"], result["solver"].kperp2 / 2.3**2)
    assert rec["ky_rho_s"] == pytest.approx(0.46)


def test_domain_warning_blocks_unqualified_layer_metric(example):
    _, deck, result = example
    rec = Q._record(deck, result, 0.2848826)
    assert rec["converged"] and rec["domain_warning"]
    assert not rec["mtm_like"]
    assert rec["transport"]["layer_fallback_required"]
    assert rec["transport"]["kr2_geometric"] is None
    assert rec["apar_shape_warning"] and not rec["resolution_converged"]


def test_seed_frequency_conversion_and_resolved_layer_metric(example, monkeypatch):
    _, deck, base = example
    calls = []

    def fake_solve(seed, **kw):
        calls.append(seed)
        return dict(
            base, omega=0.2 + 0.01j, domain_warning=False, edge_phi=0.0, edge_g=0.0
        )

    fake_solver = SimpleNamespace(**vars(base["solver"]), solve=fake_solve)
    monkeypatch.setattr(
        Q.CollisionalMTMSolver, "from_deck", lambda *a, **kw: fake_solver
    )
    fake_deck = SimpleNamespace(
        units=dict(deck.units, c_s_over_c_ref=3.0), solver_kw=deck.solver_kw
    )
    result = Q.solve_deck(fake_deck, 0.2, omega0=-0.6 + 0.03j)
    assert calls == pytest.approx([0.2 + 0.01j])
    assert result["omega"] == pytest.approx(-0.6)
    assert result["gamma"] == pytest.approx(0.03)
    assert result["gamma_solver"] == pytest.approx(0.01)
    transport = result["transport"]
    assert transport["kr2_geometric"] == pytest.approx(
        np.sqrt(result["kperp2"]["phi"] * result["kperp2"]["apar"])
    )
    assert not transport["validated"] and not transport["layer_fallback_required"]


def test_unresolved_iterate_not_returned_as_a_prediction(example):
    _, deck, result = example
    failed = dict(result, status="unresolved", converged=False, growing=False)
    rec = Q._record(deck, failed, 0.2848826)
    assert np.isnan(rec["omega"]) and np.isnan(rec["gamma"])
    assert not rec["converged"] and not rec["no_root_verdict"]
    assert "giacomin" not in rec and "phi" not in rec


def test_fields_false_keeps_transport_metadata(example):
    _, deck, result = example
    rec = Q._record(deck, result, 0.2848826, fields=False)
    assert "phi" not in rec and "theta" not in rec
    assert rec["giacomin"]["kperp2"] == result["kperp2"]
    assert rec["validity"] == result["validity"]


def test_warm_seeds_do_not_mix_backends_or_parities():
    good = dict(
        backend="lorentz_ei",
        branch="mtm",
        converged=True,
        ky=0.2,
        omega=-0.4,
        gamma=0.01,
        theta0=0.0,
    )
    other = dict(good, backend="cs", omega=-1.0)
    hybrid = dict(good, backend="hkbm", branch="hkbm", omega=0.2)
    assert Q._warm_seed([other, hybrid], 0.2) is None
    assert Q._warm_seed([other, hybrid, good], 0.22) == pytest.approx(
        (-0.4 + 0.01j) * 1.1
    )
    assert Q._warm_seed([good], 0.5) is None


def test_nonzero_theta0_explicitly_unsupported(example):
    path, _, _ = example
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (rec,) = Q.run_linear_mtm(path, ky=[0.2], theta0=[0.3])
    assert rec["status"] == "unsupported" and rec["theta0"] == 0.3
    assert not rec["converged"] and np.isnan(rec["gamma"])
    assert "theta0=0" in rec["error"]


def test_toroidal_n_input_and_input_order(example, monkeypatch):
    path, _, _ = example
    calls = []

    def fake(deck, ky, **kw):
        calls.append((ky, kw))
        return dict(status="unresolved")

    monkeypatch.setattr(Q, "solve_deck", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = Q.run_linear_mtm(path, n=[40, 20], rho_star=0.0026)
    assert [r["n"] for r in records] == [40, 20]
    assert records[0]["ky"] == pytest.approx(40 * 0.0026 / 0.5438691, rel=1e-4)
    assert calls[0][0] < calls[1][0]


def test_existing_api_routes_only_explicit_lorentz_backend(monkeypatch):
    calls = []

    def fake(source, **kw):
        calls.append(kw)
        return [dict(backend="lorentz_ei")]

    monkeypatch.setattr(Q, "run_linear_mtm", fake)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not run hKBM or collisionless MTM")

    monkeypatch.setattr(mtm, "run_linear_mtm", forbidden)
    rec = QL.run_linear(
        "unused", ky=[0.2], modes=("mtm",), mtm_kw=dict(backend="lorentz_ei", nturns=4)
    )
    assert rec == [dict(backend="lorentz_ei")]
    assert calls[0]["mtm_kw"] == dict(nturns=4)


def test_combined_adapter_filters_hkbm_warm_roots(monkeypatch):
    calls = []

    def hybrid(source, **kw):
        calls.append(kw)
        return [dict(converged=False)]

    monkeypatch.setattr(QL, "run_linear", hybrid)
    monkeypatch.setattr(Q, "run_linear_mtm", lambda *a, **kw: [])
    warm = [dict(branch="hkbm"), dict(branch="mtm", backend="lorentz_ei")]
    rec = mtm.run_linear_modes("unused", mtm_kw=dict(backend="lorentz_ei"), warm=warm)
    assert calls[0]["warm"] == [warm[0]]
    assert rec[0]["branch"] == "hkbm" and rec[0]["parity"] == "twisting"


def test_unknown_backend_fails_before_solving():
    with pytest.raises(ValueError, match="backend"):
        mtm.run_linear_modes("unused", mtm_kw=dict(backend="typo"))


@pytest.mark.parametrize(
    "kw",
    [
        dict(theta0=0.1),
        dict(omega0=-0.5 - 0.01j),
        dict(timeout=-1),
        dict(max_side_turns=-1),
    ],
)
def test_bad_inputs_fail_explicitly(example, kw):
    _, deck, _ = example
    with pytest.raises(ValueError):
        Q.solve_deck(deck, 0.2, **kw)
