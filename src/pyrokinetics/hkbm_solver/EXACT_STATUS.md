# Full linear GK solver: validation and handover

Updated 6 October 2026. This module is under development. "Exact" distinguishes
its kinetic equations from the reduced hKBM/C&S orderings; its discretisation
still requires field, velocity and ballooning-domain convergence tests.

## Implemented

`exact.ExactSolver` integrates collisionless passing and trapped particle
responses, with phi, A_parallel and delta-B_parallel, finite electron mass and
Bessel gyroaverages. phi and b use continuous linear elements; A_parallel is
constant per cell. Trapped-pitch intervals split where bounce points cross cell
edges. Both parities are available for theta0=0 on symmetric geometry.

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
No collisions, Landau continuation into damped modes or GENE numerical damping
are implemented. Collisional decks produce an explicit warning.

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
This was checked in an isolated worktree containing only this milestone over
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
2. Implement and independently test a pitch-angle collision operator. A Krook shift
   is not that operator. Benchmark actual collisional MTM growth, frequency and
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
