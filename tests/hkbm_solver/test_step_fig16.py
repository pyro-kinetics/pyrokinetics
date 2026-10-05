"""Benchmark of the hKBM solver against GENE on the STEP (beta_e, k_y) scan.

The GENE values in data/gene_step_fig16_subset.csv are six points of the linear GENE
scan shown in Fig. 16 of D. Kennedy et al., "On the transition to large fluxes and
access to second stability in gyrokinetic simulations of electromagnetic turbulence in
STEP" (STEP-EC-HD, Psi_n = 0.49, q = 3.5, nominal collisionality; beta' consistent with
beta_e).  Columns as in the scan: beta, ky (k_y rho_s), growth_rate and mode_frequency
(c_s/a, GENE sign: omega > 0 is the ion diamagnetic direction), growth_rate_tolerance
and Ctear (tearing-parity measure).

The decks are data/step_ky0.2 (the validation deck; amhd = dpdx_pm = -1, so GENE
resolves beta' from beta and the gradients) with only kymin and beta changed.

* hKBM region: over the Fig. 16 grid at beta_e <= 0.11, k_y rho_s 0.1-0.5, the ratio of
  the solver's growth rate to GENE's has median 1.01 (10th-90th percentile 0.56-1.28)
  and omega has GENE's sign at 97 % of the points.  At the five points tested here
  (beta_e <= 0.10, k_y rho_s 0.19-0.38) it is within -23 % ... +19 %; asserted to 30 %,
  with omega in the ion direction like GENE's.
* beta_e = 0.15, k_y rho_s = 0.47: GENE's mode is a weak electron-direction mode (the
  paper's MTM side of beta_e ~ 0.11) and the solver's default seeds find no growing
  ballooning root.  In GENE's tearing-parity cells (Ctear > 0.5, k_y rho_s 0.28-0.38,
  beta_e >= 0.13) the solver still has a weak ion-direction ballooning root
  (gamma <= 0.02), so no no-root assertion is made there.

Each solve takes ~10 s (the no-root point ~50 s): marked slow.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrokinetics.hkbm_solver.gene_io import Deck

DATA = Path(__file__).parent / "data"
GENE = pd.read_csv(DATA / "gene_step_fig16_subset.csv")
HKBM = GENE[GENE.beta <= 0.105]
NOROOT = GENE[GENE.beta > 0.105]
GAMMA_RTOL = 0.30


def _solve(tmp_path, ky, beta):
    txt = (DATA / "step_ky0.2" / "parameters").read_text()
    for key, val in (("kymin", ky), ("beta", beta)):
        txt, n = re.subn(
            rf"^(\s*){key}\s*=.*$", rf"\g<1>{key} = {val!r}", txt, count=1, flags=re.M
        )
        assert n == 1
    d = tmp_path / "run"
    d.mkdir()
    (d / "parameters").write_text(txt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = Deck(d / "parameters")
        return deck.solve(ky, scan=False)


def _id(r):
    return f"b{r.beta}-ky{r.ky:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("row", list(HKBM.itertuples(index=False)), ids=_id)
def test_hkbm_region_against_gene(tmp_path, row):
    r = _solve(tmp_path, row.ky, row.beta)
    assert r["converged"] and r["gamma_ref"] > 0
    assert r["gamma_ref"] == pytest.approx(row.growth_rate, rel=GAMMA_RTOL)
    assert np.sign(r["omega_ref"]) == np.sign(row.mode_frequency) == 1


@pytest.mark.slow
@pytest.mark.parametrize("row", list(NOROOT.itertuples(index=False)), ids=_id)
def test_no_ballooning_root_beyond_hkbm(tmp_path, row):
    assert row.mode_frequency < 0  # GENE: electron-direction mode, not the hKBM
    r = _solve(tmp_path, row.ky, row.beta)
    assert not (r["converged"] and r["gamma_ref"] > 0)
