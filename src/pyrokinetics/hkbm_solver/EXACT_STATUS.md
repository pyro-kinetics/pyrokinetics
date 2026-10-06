# Full linear GK solver: validation and handover

Updated 6 October 2026. This module is under development. "Exact" distinguishes
its kinetic equations from the reduced hKBM/C&S orderings; its discretisation
still requires field, velocity and ballooning-domain convergence tests.

## Implemented

### Experimental T3D linear-mode adapter (6 October)

The collisional model is now explicitly selectable with
`quasilinear.run_linear(..., modes=("hkbm","mtm"),
mtm_kw={"backend":"lorentz_ei"})`; the existing default is unchanged.
`mtm_collisional_ql.solve_deck` provides the direct GENE-sign/deck-unit entry.
Both branches are independent; no hKBM work for MTM-only calls, no automatic
dominant-root suppression, no hidden process pool. Warm starts are backend
separated; nonzero MTM theta0 is explicitly unsupported. Failed searches have
NaN predicted frequencies and are not classified stable.

Full-domain fields and independently weighted phi/A widths now satisfy the
published T3D record contract. No fabricated MTM flux shares or Bpar field.
The geometric-mean width is exposed only with passing distribution/potential
edge checks; otherwise the record explicitly requests caller-owned fallback.
All model/A-shape/resolution warnings remain; this is NOT calibrated transport.
The first 35 MTM/operator+adapter tests passed (2607948,2.08 s), including
independent full-domain integrals, units, normalization invariance, dispatch,
warm separation and unsupported/failure records. A further nonunity-frequency
test and isolated full-suite validation are being prepared before publication.
No T3D checkout is available on Pitagora; CSD3 must pull and wire/confirm the
explicit backend in its worker. Do not claim remote deployment from this API.

### Separate collisional MTM experiment (6 October, tested experimental release)

Dan clarified the target: a **quick reduced hKBM + reduced MTM pair that very
roughly reproduces GENE beta-ky trends**, with full GK as a diagnostic rather
than the default production route. Prioritise approximate growth/branch maps,
coverage and cost, while keeping unresolved/numerically uncertain cells visible.

`mtm_collisional.CollisionalMTMSolver` now implements a coupled electron-ion
Lorentz response on theta/local pitch/energy. It starts from C&S's electron
equation, retains hot ions and no Bpar, and initially uses A=B/kperp^2. Optional
`nA>1` tests a broader even-A basis. It is not a new collision option in
`ExactSolver`, not the old `mtm.py` Krook shift, not Sugama, and not automatically
selected by `run_linear` or T3D. Equations and limitations are in PHYSICS.md.

```python
from pyrokinetics.hkbm_solver.mtm_collisional import CollisionalMTMSolver

# deck must contain the intended beta/pressure gradients and collision strength.
solver = CollisionalMTMSolver.from_deck(
    deck, 0.2848826, npt=32, nturns=16, nE=16, nxi=24, theta_order=3
)
root = solver.solve(0.52 + 0.01j, timeout=240)
```

Seed/internal frequencies use c_s/L_ref and electron-direction positive real
part; `omega_gene_deck` and `gamma_deck` are converted to deck units. `status`
distinguishes growing_root, unresolved and timeout. `validated=False` and
`collision_physics_match=False` are deliberately retained for every result.
Inspect QN/kinetic/projected residuals, `edge_phi`, `edge_g`, and
`ampere_core_residual`, not just `converged`. The prescribed/basis-projected
Ampere equations need not satisfy pointwise Ampere's law.

**104 tests pass in an isolated 817dcef2 checkout** (Pitagora 2607439, 210.80 s,
one expected warning), including 18 new MTM tests and a deliberately
short-domain root regression which must expose its large edge amplitude.
Numerical roots with physical drift sign and energy-dependent collisions have
been obtained at beta=.14, ky=.2848826, but **a converged GENE benchmark is not
yet established**. The tempting coarse pilot gamma=.0059737 versus GENE .0065721
does not survive refinement unchanged. Third-order npt64/nE16/nxi24/domain
+/-65pi gives gamma=.0161021 with the fixed A shape; doubling pitch resolution
changes it substantially. Allowing five A basis functions reduces local
Ampere mismatch but does not remove the model/resolution questions. Zero-rate
and constant-rate control searches are unresolved, not certified stable.

The two higher-pitch retries also converged: at beta=.14/ky=.2848826,
npt64/nturns32/nE16/nxi48 gives gamma=.01146579, omega_GENE=-.52751696;
nA5 at nturns16 gives .01309776/-.52402484. npt32/nturns16/nE16/nxi64
gives .00914844/-.52832485. These are not a jointly converged sequence.

**24-cell rough-pair pilot:** four betas (.09,.11,.13,.15), six ky values
(nearest .14,.237,.285,.38,.57,.95). Reuse saved old collisional hKBM roots;
solve MTM from the theoretical diamagnetic seed, not GENE eigenvalues. Profile:
npt24/nE12/nxi16, third-order theta, nA1, nturns=ceil(32*.2848826/ky), at least8.
Pitagora 2607451 took48.0 s on12 workers, median MTM search13.8 s. A pristine
817dcef2 repeat (2607476), with an asserted shared-filesystem import path,
reproduces the numerical results. All source hashes are recorded/unchanged.
An intervening attempt using a node-local /tmp PYTHONPATH fell back to the
editable worktree; it is explicitly excluded as isolation evidence.

Of21 GENE-growing sampled cells,18 have a growing pair candidate;16/18 are
within a factor of2 (median ratio1.226). Three GENE-growing higher-ky cells
remain unresolved. At beta=.13/ky=.2848826, the old twisting root wins over
MTM and overpredicts GENE growth3.71x: the branch transition is not yet right.
At beta=.15/ky=.2848826 the selected MTM gives .011693 versus GENE .007917.
No accepted MTM root fails the pilot domain criterion, but A-shape warnings
remain, and this profile is not resolution-converged. The other3 sampled
GENE-damped cells are unresolved, not demonstrated stable.

Published numerical sample: `tests/hkbm_solver/data/mtm_collisional_pair_pilot.csv`
and its `.provenance.json` sidecar. This is observational benchmark evidence,
not a test that blesses every point as physically correct. Raw records and
research drivers remain on Pitagora in `analysis/solver/mtm/`.

### Full reduced-pair first pass: 791 searches in 106 seconds (6 October)

Dan challenged the initial 1-2 hour estimate and noted GENE uses one parallel
domain. The estimate unnecessarily extrapolated an uncapped 1/ky domain with
only 12 workers on a 256-core allocation. A new 128-worker run, **2607886**,
completed all **791 searches in 106.0 s**, allocation 110 s, no errors/timeouts.
Median MTM search 14.74 s. Old collisional hKBM results were reused, not retimed.
Numerical implementation remains the source-path-verified clean 817dcef2;
all source hashes stayed unchanged. The existing 104-test evidence still applies.

Profile: npt24/nE12/nxi16, third-order, fixed A, physical drift sign, theory
seeds; nturns=min(32,max(8,ceil(32*.2848826/ky))). This caps 251 cells and uses
at most 65 total ballooning periods. It is an exploratory truncation, NOT an
exact match to GENE boundary conditions or a convergence certificate.

Local GENE template/output: nx0=64, nz0=128, nexc=1, n_pol default1. Its one
parallel period connects different kx harmonics; it is not one isolated
ballooning period with zero incoming g. See [GENE parallel boundary condition,
Eq.3.32](https://genecode.org/PAPERS_1/lapillonne.pdf). Domain check 2607880:
at beta=.14/ky=.2848826, one isolated period takes .59 s but gamma=.0007526,
versus .0114589 at nturns32 (16.05 s). Simply cutting to one period loses the
growth by a factor 15.2 in this model. nturns16 gives .0114650 in 8.18 s but
still has edge_g=.061. Low-ky results remain particularly domain sensitive.

New MTM searches: **480 growing roots, 311 unresolved**, no timeouts/errors.
Pair selects **371 old hKBM, 183 MTM, 237 unresolved**. Of **597 GENE-growing
cells** (gamma>.001), 498 have candidates and **378 are within a factor two**
(378/597 overall, 378/498 recovered); median recovered growth ratio 1.081.
There are also 56 pair candidates where GENE gamma<=.001, so this is not a
validated stability boundary. **143 MTM roots have domain flags**, including
65 selected MTMs. Field-shape/resolution limitations and old-hKBM outliers remain.

Published full table: `tests/hkbm_solver/data/mtm_collisional_pair_full.csv`
and `.provenance.json`. Complete raw SHA256:
`d2770a92a73bff2ff93cca1e9193dbea510012a7ea29d5ac5877fa67a8691b91`.
Pitagora output stem `analysis/solver/mtm/rough_pair_full_20261006` contains
JSONL, summary, PNG and PDF. The plot uses GENE colour limits, grey unresolved,
MTM-selection circles, orange MTM-domain flags and crosses for clipped values.
This completes grid coverage, NOT physical/model validation. Next improve
missing/masked branches and low-ky truncation/outliers. T3D defaults unchanged.

### Full-orbit solver and legacy hKBM collisions

`exact.ExactSolver` integrates collisionless passing and trapped particle
responses, with phi, A_parallel and delta-B_parallel, finite electron mass and
Bessel gyroaverages. phi and b use continuous linear elements; A_parallel is
constant per cell. Trapped-pitch intervals split where bounce points cross cell
edges. Both parities are available for theta0=0 on symmetric geometry.

An explicit `collision_model="legacy"` option now adds the old hKBM reduced
collisions to **twisting parity only**. Defaults remain collisionless. This is
not a full collisional GK or Sugama operator, and is not validated for MTM.

```python
from pyrokinetics.hkbm_solver.gene_io import Deck
from pyrokinetics.hkbm_solver.exact import ExactSolver

deck = Deck("parameters")
solver = ExactSolver.from_deck(deck, 0.3, npt=64, nturns=1,
                               nE=12, nlp=12, nlt=4)
root = solver.find_root(-0.19 + 0.14j, parity="twisting", timeout=120)
```

The seed and `root["omega"]` use internal units, with positive real frequency in
the electron direction. `omega_gene` reverses that real sign but retains internal
frequency units; multiply it and gamma by `deck.units["c_s_over_c_ref"]` for the
deck's units. The benchmark CLI below performs that conversion.

`theta_nodes` carries phi and b; `theta_cells` carries A_parallel. The returned
`kperp2_phi` and `kperp2_apar` integrate the element basis with the field-line
Jacobian and are in inverse rho_s squared. `edge_phi` diagnoses the end of the
chosen domain. These are not a demonstration that an MTM current layer is resolved.

Convergence requires a small frequency step and a componentwise field-equation
residual below tolerance. A failed or timed-out search is **unresolved**, not
evidence of stability. Only Im(omega)>0 is supported. Timeouts are checked between
matrix evaluations; one expensive evaluation can overrun the requested limit.
No full gyrokinetic collision operator, Landau continuation into damped modes
or GENE numerical damping is implemented. Nonzero-collision decks warn if the
default collisionless mode is used. Selecting legacy collisions warns about the
reduced operator's mismatch with GENE.

## Optional legacy reduced collisions

```python
solver = ExactSolver.from_deck(deck, 0.3, collision_model="legacy",
                               npt=32, nturns=2, nE=12, nlp=12)
root = solver.find_root(-0.19 + 0.10j, parity="twisting", timeout=120)
```

The deck's converted coll parameter is used. Direct construction also accepts
`coll`, `coll_ee`, `coll_i`, `coll_eps`. Nonzero coll with collision_model='none'
is rejected. Passing particles remain collisionless. Trapped ions receive the
old energy-dependent Krook detrapping in the orbit propagator, leaving the
diamagnetic source frequency unchanged.

For electrons, the old bounce-averaged Lorentz matrix and deflection rates are
reused on the full solver's trapped-pitch grid. It acts on
`H = <G - (1 - omega_star/omega) psi>`, with `d_l psi = i omega A_parallel` and
psi zero at the left boundary. It must NOT be applied blindly to <G>. A
self-consistent low-rank resolvent correction couples pitches within each well;
the finite-transit collisionless response is retained. The operator is therefore
a **projected legacy closure**, not a pointwise velocity-space collision operator.
Only symmetric one-well geometry and twisting parity are currently supported;
tearing and unrestricted-parity root searches with this option are rejected.

Eight new tests independently check the orbit probes against solve_ivp, the
pitch coupling against direct dense solves (including the psi subtraction),
the exact zero-collision identity and small-rate limit, metadata and input guards.
Together with non-slow exact tests, 26 tests pass. The **full isolated suite
passes: 86 tests, one expected collision-operator warning, 207.10 s** (Pitagora
job 2606419). Numerical sources were a clean 7f78cbdf checkout, excluding all
unpublished speed edits; only generated test-build version metadata was added.

Three preliminary STEP pilots at ky=.2848826, npt32/nE=nlp12/nturns2:

| beta | new collisionless gamma | new legacy-collisional gamma | old legacy gamma |
| --- | ---: | ---: | ---: |
| .09 | .1487301 | .1024053 | .0914296 |
| .13 | .0770207 | .0129221 | .0195469 |
| .15 | .0450957 | .0105435 | .0020880 |

These restore substantial collisional suppression but do not establish
resolution convergence or agreement with GENE. The beta=.13 collisionless seed
did not converge; the old collisional seed found the reported root. Not an
exhaustive spectrum search. Collisionless results and their warnings remain valid.

The full 791-cell legacy-twisting first pass completed (job 2606408, 241.7 s
scan wall time on 256 cores): 389 roots, 304 timeouts, 98 unresolved searches;
32 roots have edge_phi>=.01. The plotted comparison retains the separate
collisionless tearing pass: 103 roots, 688 unresolved, 98 endpoint flags.
Low-ky twisting outliers remain (76 above the GENE maximum growth, largest
gamma=7.39642), so neither numerical convergence nor reliable MTM identification
is claimed. Failed searches are not stable points. These full-grid runs include
hashed local speed edits; the isolated test result above does not.

## Current evidence

The tests independently compare the passing and periodic trapped kernels against
initial-value integration, including bounce points inside cells; check exponential
moments against quadrature; reject stagnation at the growing-half-plane boundary;
and verify field-weighted wave numbers by independent quadrature. A strong-damping
trapped-orbit regression first reproduced overflow, then passed after replacing
unsafe factored exponentials with bounded forward-time propagators in that regime.
Non-finite matrices or field responses now raise an explicit numerical error.

For collisionless STEP NC_fB1_ky0.3, beta_e=0.09, GENE reports
omega=0.1923, gamma=0.1366 c_s/a. The full solver at npt=96, nE=nlp=8, nlt=6,
nturns=1 gives 0.191807/0.150145. The tested velocity and domain refinements change
gamma by less than 1%, but growth remains about 10% above this GENE run. A new
GENE run with nv=64/nw=48 and nx=16 converged at omega=0.1903, gamma=0.1366
(job 2605126). The nx=16/nv=32/nw=24 control was cancelled by the scheduler before
starting, so the velocity and radial-box changes are not isolated. GENE retains
hyp_z=-1; the growth discrepancy is still unexplained, not demonstrated to be
solely numerical damping.

A six-point collisionless spectrum (ky=0.2 through 0.7) has been run at npt=32/64.
At ky=0.6 and 0.7, gamma changes from 0.04111/0.03467 to 0.05195/0.04610 when
doubling field resolution. These weak modes are **not field-converged** at npt=32.
At npt=64, increasing nE=nlp from 12 to 24 to 48 gives gamma=0.05195/0.03956/0.03826
at ky=0.6 and 0.04610/0.03754/0.03519 at ky=0.7. Joint field/velocity checks are
required before defining a production resolution; field-equation residuals alone
do not establish discretisation convergence.

All **78 hKBM solver tests pass**, including 19 exact tests (205.40 s on one CPU).
This is the earlier collisionless milestone, checked in an isolated worktree over
the prior commit, excluding the separate unpublished reduced-solver speed work.
The one warning documents the existing reduced model's collision-operator mismatch.
This does not validate collisional MTMs, long current layers or an entire beta-ky map.

## Reproducible beta-ky benchmark

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pyrokinetics.hkbm_solver.exact_benchmark --collisionless \
    --reference /path/to/STEP-EM-TRANSITION/scripts/figure16a/data.csv \
    --template /path/to/GENE/parameters \
    --output fig16_collisionless.jsonl --indices 179,251,261,276 \
    --npt 32 --nturns 2 --workers 1 --timeout 120
```

Use the site's scheduler. This is a **GENE-seeded diagnostic**, not an exhaustive
search for the dominant instability. `--parity reference` uses Ctear>0.5 to choose
tearing parity; `--parity both` tries both sectors. A nominal-collisional template
is explicitly run with collisions off, and `collision_physics_match=false` is
recorded. Frequencies returned by the CLI are in deck units and GENE's sign.

For the legacy twisting comparison, replace `--collisionless` with
`--collision-model legacy --parity twisting`. This preserves nominal coll in
the deck but still marks collision_physics_match=false for a nonzero-collision
GENE reference: the reduced model is not Sugama. Save to a NEW output path.

The CSV's rectangular domain has 36 beta x 22 ky = 792 cells, but the source
contains **791 measured rows**: beta=0.146, ky=0.009496086 is missing. The driver
preserves those rows and does not invent a missing GENE eigenvalue. A completed
future full-grid solver scan should include that cell with its reference marked
missing. The four example indices are zero-based CSV row numbers, not rectangular
indices or GENE filename suffixes.

The four-row pilot completed: three twisting searches found roots with the
collision-physics mismatch flag set. Row 276 (beta=0.14, ky=0.2848826, tearing)
was unresolved after 20 iterations (field-equation residual 0.1166). Its positive
last-iterate growth rate must not be interpreted as an MTM eigenvalue.

JSONL records contain input/source SHA256 hashes, numerical settings, root status,
residuals, edge amplitude and timing. Only an identical configuration can resume
an output file; changed code or resolution requires a new output. Missing measured
GENE tolerances remain null. Unresolved records are not automatically retried on
resume; use a new output for a changed search or refinement.

## Remaining milestones

1. Finish collisionless field/velocity/domain convergence and GENE reference checks.
2. Extend beyond the projected legacy hKBM collision closure to a validated full
   pitch-angle collision operator. A Krook shift is not that operator.
   Benchmark actual collisional MTM growth, frequency and
   current-layer structure against GENE/GS2, including collisionality scans.
   The supplied GENE deck uses Sugama collisions with conservation/FLR options;
   pitch-angle scattering alone is not identical physics. First compare like
   operators, then implement or quantify the difference from the paper's operator.
3. Establish long-theta and branch-search convergence, then run the nominal-physics
   Figure 16 comparison on all 792 grid cells. Keep missing GENE data, numerical
   failures, unresolved searches and physical stability distinct.
4. Provide validated outputs to CSD3/T3D through the agreed API and coordination log.

Pitagora's detailed resumption state is
`/pitagora_scratch/userexternal/dkennedy/hKBM/analysis/solver/HANDOVER.md`;
evidence is in `exact/stage1/` and `exact/EXACT_PROGRESS.md` beneath that directory.
Cross-machine communication is `TALKS/STEP_CIPS/COORDINATION.md`.

The C&S branch's drift-sign discrepancy remains unresolved. Its lack of growing
roots does not establish the spectrum of these full GK equations, and no error
in the paper's sign convention has been established.
