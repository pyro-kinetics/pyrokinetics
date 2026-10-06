# hkbm_solver: a fast hKBM eigenvalue solver that looks like GENE

`pyrokinetics.hkbm_solver` solves the local, linear, electromagnetic eigenproblem of the hybrid
kinetic ballooning mode (hKBM) of STEP-like plasmas in seconds per k_y. It reads a **GENE
`parameters` file** and writes **GENE's output files** (`parameters`, `omega`, `field`, `nrg`,
`miller`) in GENE's format, names and normalisation, so pyrokinetics' existing GENE reader loads
the result unchanged:

```bash
hkbm-solve run_dir/                  # reads run_dir/parameters, writes run_dir/*.dat
```

```python
from pyrokinetics import Pyro
pyro = Pyro(gk_file="run_dir/parameters")
pyro.load_gk_output()
pyro.gk_output["growth_rate"], pyro.gk_output["mode_frequency"], pyro.gk_output["eigenfunctions"]
```

## Install

From a checkout of this branch:

```bash
pip install -e .            # registers the hkbm-solve command
hkbm-solve --selftest       # internal checks (about a minute)
pytest tests/hkbm_solver    # geometry against GENE, STEP regression, pyrokinetics round trip (a few minutes)
```

`python -m pyrokinetics.hkbm_solver run_dir/` works without the console script.

## Usage

```bash
hkbm-solve run_dir/                       # or run_dir/parameters
hkbm-solve run_dir/ -o out_dir/           # write elsewhere
hkbm-solve run_dir/ --omega0 0.1+0.08j    # seed (GENE sign: omega + i gamma, deck units)
hkbm-solve run_dir/ --scan                # always add a coarse search of the complex-omega plane
```

```python
from pyrokinetics.hkbm_solver import run
res = run("run_dir/parameters")           # list of dicts: ky, gamma, omega, converged, ...
```

Round trip from another code (what pyrokinetics users will normally do):

```python
from pyrokinetics import Pyro
pyro = Pyro(gk_file="input.gs2")          # or input.cgyro, ...
pyro.convert_gk_code("GENE")
pyro.write_gk_file("run_dir/parameters", gk_code="GENE")

from pyrokinetics.hkbm_solver import run
run("run_dir/parameters")

pyro = Pyro(gk_file="run_dir/parameters")
pyro.load_gk_output()                     # growth rate, frequency, eigenfunctions in pyro units
```

Output names follow GENE: a single k_y gives `parameters.dat`, `omega.dat`, `field.dat`,
`nrg.dat`, `miller.dat` (what `Pyro(gk_file=run_dir/parameters)` looks for); several k_y
(`nky0 > 1`, or GENE's scan syntax `kymin = 0.1 !scanlist: 0.1, 0.2, 0.3`) give one GENE-like run
per k_y with suffixes `_0001`, `_0002`, ... and a `scan.log`; load them one at a time with
`Pyro(gk_file="run_dir/parameters_0001")`. `hkbm.json` (`hkbm_0001.json`, ...) holds the
full-precision eigenvalue, convergence information and the unit factors.

## What is read from the deck

| namelist | used |
|---|---|
| `&box` | `kymin`, `nky0` (or a `!scan`/`!scanlist` on `kymin`), `nz0` (grid of the output fields only), `kx_center` (must be 0) |
| `&general` | `beta`, `bpar`, `coll`, `collision_op`, `nonlinear` (must be false), `hyp_*` (ignored, warned) |
| `&geometry` | `magn_geometry = 'miller'`, `q0`, `shat`, `trpeps` (or `rho`), `major_R`, `minor_r`, `kappa`, `s_kappa`, `delta`, `s_delta`, `zeta`, `s_zeta`, `drR`, `drZ`, `major_Z`, `amhd`, `dpdx_pm`, `dpdx_term`, `sign_Ip_CW`, `sign_Bt_CW` |
| `&species` | `name`, `charge`, `mass`, `temp`, `dens`, `omn`, `omt`; the hKBM GENE switches `bpar_vlasov`, `bpar_field`, `bpar_source` (`'full'`/`'none'`) |
| `&units`, `&external_contr` | echoed; flow (`ExBrate`, `pfsrate`, `Omega0_tor`) must be zero |

`amhd`/`dpdx_pm` are resolved as GENE does (`-1`: from beta and the gradients; `-2`/default: from
`amhd`; `dpdx_term = 'curv_eq_gradB'`: no pressure term). `dpdx_term = 'gradB_eq_curv'` removes
beta' from the grad-B drift of both species.

Anything the solver cannot represent stops with `UnsupportedDeck` (exit code 2 from the CLI):
nonlinear runs, flow or flow shear, k_x != 0, other than one electron species (charge -1) and one
ion species (charge +1), unequal densities or density gradients, beta = 0, non-Maxwellian or
passive species, `no_trap`, `bpar_vlasov_terms`/`_pitch` other than `'all'`, geometries other than
Miller.

## Geometry

`miller.py` is a line-by-line Python port of GENE's `miller_geometry.F90` (Miller et al., Phys.
Plasmas 5, 973 (1998)) with GENE's grids, third-order Lagrange interpolation, finite differences
and integrals, plus the `amhd`/`dpdx_pm` logic of `geometry.F90` and the curvature
`K_y = (dBdx - ga3/ga1 dBdz)/C_xy` of `set_curvature`. It reproduces the `miller.dat` written by
GENE to 1e-13 relative (tests: nominal STEP, kappa, delta, shat variations, reversed `sign_Ip_CW`
and `sign_Bt_CW`, squareness, and a pyrokinetics GS2-to-GENE conversion). The solver uses it at
nz0 = 512 whatever the deck's `nz0` (which only sets the grid of the written fields).

## Physics model (brief)

Units inside the solver: GENE's with T_ref = T_e, n_ref = n_e, m_ref = m_i (same B_ref, L_ref);
`gene_io.Deck` converts any GENE normalisation to these and back. Time dependence exp(-i omega t).

* **Ions**: gyrokinetic, integrated exactly along every orbit of the central ballooning turn
  (parallel streaming and bounce motion, passing and trapped), exact Bessel FLR, magnetic drift
  with beta' (`full_drift`), mu delta B_par in chi.
* **Trapped electrons**: bounce-averaged (omega_be >> omega >> omega_de), precession with beta'
  and the mu delta B_par term, pitch-angle scattering by the bounce-averaged Lorentz operator at
  GENE's nu_ei.
* **Passing electrons**: through psi (A_par = -(i/omega) d_l psi), no non-adiabatic part.
* **Fields**: phi, psi (A_par) and delta B_par from quasineutrality, perpendicular pressure balance
  and the vorticity (parallel current) equation; delta B_par is a field, not a closure.
* **Eigenfunction**: free, 8 even Hermite-Gaussian functions per field in the ballooning angle
  (twisting parity, width 0.35 rad); Galerkin projection, det D(omega) = 0 by secant iteration.
* The GENE delta B_par channel switches (`bpar_vlasov`, `bpar_field`, `bpar_source` per species)
  and `dpdx_term` are exact switches of the corresponding terms.

The equations are written out in `PHYSICS.md` (this folder). Against GENE for STEP (k_y rho_s
0.1-0.8, nominal collisionality) the main model gives omega within ~10 % and gamma within
-25 % ... +15 % at k_y rho_s <= 0.4 (0.064 vs GENE 0.083 at 0.2, 0.097 vs 0.097 at 0.3), the sign of
every GENE delta B_par channel test and the trend of ten geometry/beta design levers; its
high-k_y cut-off comes ~0.1-0.2 late in k_y.

## Using it as a flux model

`quasilinear.py` turns the eigenmode into quasilinear fluxes. Formulas are in `PHYSICS.md`
("Quasilinear fluxes").

**Raw linear output for a transport code's own saturation rule** (one process, no MPI; set
`OMP_NUM_THREADS=1`):

```python
from pyrokinetics.hkbm_solver.quasilinear import run_linear
modes = run_linear(pyro, n=[10, 20, 40, 70], theta0=[0.0, 0.2], rho_star=rho_ref_over_L_ref,
                   timeout=60)
# or run_linear("run_dir/parameters", ky=[0.1, 0.2, 0.3])   (k_y rho_ref, deck units)
for m in modes:
    m["gamma"], m["omega"], m["converged"], m["hkbm_like"], m["checks"]
    m["theta"], m["phi"], m["apar"], m["bpar"], m["kperp2"], m["jacobian"], m["bmag"]
    m["weights"]["Q_i"]["total"], m["weights"]["shares"], m["kperp2_avg"]
```

* `source`: a pyrokinetics `Pyro` object (any code; pyrokinetics writes a GENE deck), a GENE
  `parameters` file, or its directory. The solver has no flow, flow shear or Z_eff: those are set
  to zero or ignored, with a warning. Check `pyro.local_geometry.beta_prime`: a Pyro read from a
  GENE deck with `dpdx_pm = -1` and no reference B0 has beta_prime = 0, and the hKBM depends
  strongly on beta'. `run_linear` warns when the deck it writes has beta' = 0.
* k_y: `ky` in 1/rho_ref, or toroidal mode numbers `n` with k_y rho_ref = n rho_star/|C_y| (GENE's
  convention). rho_star = rho_ref/L_ref comes from `rho_star=`, the Pyro's reference values, or
  the deck's `rhostar`.
* `theta0`: the ballooning angle, as in GS2 and in Giacomin et al. (k_x = shat theta0 k_y in GS2's
  normalisation). The basis then loses its parity (16 functions per field), which roughly doubles
  the cost. In GENE's Miller coordinates the radial modes connect every 2 pi kappa_s k_y, with
  kappa_s = C_y q0 shat/x0 (`Geo.kx_per_theta0`; 3.59 for the STEP deck), so a GENE run with
  `kx_center` has theta0 = kx_center/(kappa_s k_y), not kx_center/(shat k_y). Against GENE
  kx_center runs (STEP, k_y rho_s 0.2 and 0.3):
  * theta0 = 0.07-0.27: gamma agrees as well as at theta0 = 0 (0.062-0.066 vs GENE 0.068-0.080
    at k_y 0.2; 0.093/0.083 vs 0.092/0.080 at k_y 0.3); omega is within 15 %;
  * theta0 = 0.5: 0.034 vs 0.029 (k_y 0.2) and 0.007 vs stable (k_y 0.3);
  * theta0 = 1: no root, against GENE's 0.002 and stable;
  * k_y 0.3, theta0 0.27: 0.040 against GENE's ~0.017. GENE had not converged there in 4 h.
  The solver's gamma(theta0) falls a little too fast at k_y 0.3 and is right at k_y 0.2.
  gamma(theta0) = gamma(-theta0) holds to 1e-3.
* Units: the deck's GENE normalisation. gamma and omega are in c_ref/L_ref with GENE's sign
  (omega > 0 is the ion diamagnetic direction). The fields are normalised to max |phi| = 1;
  kperp2 is in 1/rho_ref^2. `weights` are fluxes per <|phi|^2> in GENE's nrg definitions:
  Q_s in n_ref T_ref c_ref rho_ref^2/L_ref^2, the Jacobian-weighted average over the central
  turn, and the factor 2 for +-k_y. Each weight is split into phi (= es), apar and bpar, with em
  = apar + bpar. `shares` are Q_i, Q_e and Gamma divided by Q_i + Q_e. `weights_solver` is the
  same in the solver's units (T_e, n_e, m_i, rho_s).
* Flags: `converged` means a growing root was found. `hkbm_like` means all `checks` passed:
  converged, growing, ion direction, ballooning (at least half of |phi|^2 within |theta| <
  pi/2), not Alfvenic (|omega| < 0.8 c_s/L_ref), and k_y rho_s within `HKBM_KY_RANGE` (0.05-0.6,
  the range compared with GENE on STEP). Drop or replace modes that fail. The solver has only
  this one branch. Where another instability dominates (ITG/TEM at low beta, MTM, ETG, the
  electron-direction modes of GENE at high beta), it returns no root, a weak hKBM-like root, or
  occasionally an Alfvenic or electron-direction root. It never returns the other mode.
* Cost (one core): about 3-5 s per root at theta0 = 0 (8 Hermite functions per field; about 3
  secant iterations at 0.5 s per matrix evaluation, plus 0.3 s for the weights), 10-20 s at
  theta0 != 0, and about 5 s once to import pyrokinetics. Each search starts from the previous
  root: in k_y outward from k_y rho_s ~ 0.2, and in theta0 from the previous theta0. If that
  seed and the STEP seed fail, the default `scan="fast"` searches the ion-direction upper half
  plane with a low-resolution model (`gene_io.LOWRES`, 10x cheaper, roots within ~2 % of the
  full model), and its roots seed the full model. A k_y without a root therefore costs about
  10 s, and the mode carries `no_root_verdict`. A seed is abandoned once its iteration leaves
  |omega| < `wmax` = 1 c_s/L_ref or reaches gamma < -0.05. `timeout` is wall-clock per root
  (default 20 s at theta0 = 0, 40 s otherwise). Modes without a growing root have gamma =
  omega = NaN.

**Mapping to the quasilinear model of Giacomin et al. (J. Plasma Phys. 2025; T3D's GS2-QL).**
Each mode with a root carries `m["giacomin"]`. It is computed in rho_s, c_s/L_ref and T_e units, with
the fields chi = (e phi/T_e, A_par/(rho_s B_ref), dB_par/B_ref) of their eq. 3.2:
* `kperp2[f]`: <k_perp^2> weighted with J |chi_f|^2 for each field (their eq. 3.2), with
  k_perp^2 = g^yy k_y^2 + 2 g^xy k_x k_y + g^xx k_x^2 at the mode's theta0;
* `amplitude[f]` = max|chi_f| / max|chi_phi|;
* `Lambda_hat` = gamma sum_f amplitude[f]/kperp2[f] (their eq. 3.3);
  against GENE (STEP, k_y rho_s 0.2/0.3/0.4) the amplitudes are A_par/phi 0.76/0.32/0.23 vs
  GENE's 0.89/0.46/0.35, and dB_par/phi 0.42/0.27/0.31 vs 0.35/0.28/0.32. The A_par term of
  Lambda_hat is therefore 15-35 % low;
* `Q_i_over_Q`, `Q_e_over_Q`, `Gamma_over_Q`: the weights Q_l,s/Q_l and Gamma_l,s/Q_l of their eqs.
  3.7-3.8. Q_l is the total linear heat flux of all three fields.

The weights are ratios of fluxes of the same mode, so they do not depend on the normalisation of
the eigenfunction. GS2's gyro-Bohm units (v_th = sqrt(2) c_s, rho = sqrt(2) rho_s) scale Q and
Gamma by the same factor 2 sqrt(2), so the weights are the same in GS2 and GENE units when
T_ref = T_e. With another T_ref, use `giacomin` (always T_e) rather than `weights["shares"]`,
which is in the deck's units. Two cautions for absolute values:
* Fluxes per <|phi|^2> differ between codes by the normalisation of phi and the factor 2 for
  +-k_y. They are not used by the Giacomin rule.
* B_ref is GENE's Miller B_ref (the deck's &units Bref). If GS2's B0 differs (Bgeo), the A_par
  and dB_par amplitudes scale with B_ref/B0.

The theta0 average of their eq. 3.4 (up to theta0_max = gamma_E/(shat gamma)) needs Lambda_hat on a
theta0 grid; run_linear's `theta0` list gives it. On STEP the hKBM's gamma(theta0) falls to zero
by theta0 ~ 0.5-1 (GENE agrees), so a grid 0-1 is enough there.

**Warm starts.** `run_linear(..., warm=previous_modes)` seeds every (k_y, theta0) with the root of
a previous call (the last Newton iterate or time step on the same flux tube; the same k_y and
theta0, or the nearest k_y within a factor 1.6). A warm root costs about 3 s at theta0 = 0 and
5 s at theta0 != 0. If the warm seed fails, the usual continuation, the STEP seed and the fast
search follow.

**The solver's own saturation rule** (a comparison line):

```python
from pyrokinetics.hkbm_solver.quasilinear import fluxes
r = fluxes(pyro_or_parameters, ky=None, parallel=1, timeout=60)
r["Q_i"], r["Q_e"], r["Gamma"], r["Q_i_es"], r["Q_i_em"], r["gamma"], r["converged"]
```

`fluxes` uses the mixing-length rule <|phi|^2>(k_y) = C (gamma/<k_perp^2>)^2, integrated over
k_y rho_s (trapezoid on the default grid 0.05-0.6). The fluxes are in GENE gyro-Bohm units of the
deck. C_ML = 300 was fitted on the total heat flux (log least squares) of the stella STEP
nonlinear (q, beta_e) scan: 144 runs, 100 of them with Q >= 1 and a root.
* Rank correlation with the nonlinear flux: 0.86 over all runs.
* Scatter: a factor ~5 rms; 58 % of runs within a factor 3, 93 % within a factor 10.
* Above the nonlinear cliff (Q > 50) the rule is unbiased (factor ~3 rms).
* Below the cliff it is a factor ~4 too high, because the zonal-flow-regulated state is not in
  a quasilinear rule.
* 97 % of the quiet runs (Q < 1) get Q < 10.

The optional `rule="ml_threshold"` multiplies the flux by 0.126 below q^2 beta_e = 0.29 (C = 450).
That is the fitted position of the cliff, and it brings the scatter to a factor ~4, with 72 % of
runs within a factor 3. It is an empirical patch, valid for this STEP surface only. Use the rule
for trends and orders of magnitude, not for absolute fluxes near the threshold. The calibration is
in the analysis note QL_FLUX.md.

`parallel=N` splits the k_y list into N chains in N worker processes. Command line:
`hkbm-solve run_dir/ --fluxes [--ky 0.1 0.2 ...] [--nproc N]` prints the table and writes
`ql_fluxes.json`.

## Units of the output

Everything written is in the deck's own GENE normalisation: k_y in 1/rho_ref, gamma and omega in
c_ref/L_ref, GENE's sign (omega > 0: ion diamagnetic direction), fields phi in
(T_ref/e) rho_ref/L_ref, A_par in B_ref rho_ref^2/L_ref, B_par in B_ref rho_ref/L_ref on GENE's
z grid (straight-field-line angle) with `nx0 = 1` (the central ballooning turn; the eigenfunction
is negligible beyond it). `field` holds a short synthetic time series of the eigenmode,
F(z) exp(-i omega_c t) with omega_c = -omega + i gamma (GENE's own convention for the stored
fields: the raw GENE field grows as exp[(gamma + i omega) t]), so readers that measure gamma and
omega from the field history (pyrokinetics does) recover the eigenvalue. `nrg` holds zero
fluxes (quasilinear fluxes: `hkbm-solve --fluxes` or `quasilinear.py`, above). `omega` has GENE's 4-decimal format; use `hkbm.json` for full
precision.

## Limitations

* Local (flux tube), linear, electromagnetic, k_x = 0 ballooning mode only.
* Exactly two kinetic species: electrons and one singly charged ion species; no impurities,
  no adiabatic species, no flow or flow shear.
* Electrons are bounce-averaged (trapped) or adiabatic through psi (passing): valid for
  omega << omega_be, i.e. core, ion-scale modes at low collisionality. Electron-scale modes, MTMs
  and collisional (resistive) physics are out of scope.
* Ions along their orbits have no Landau continuation: only growing modes (gamma > 0) are found;
  "no root" means stable or not captured.
* The collision model is the solver's own (Lorentz pitch-angle scattering of trapped electrons,
  Krook detrapping of trapped ions) at GENE's nu_ei, whatever `collision_op` says.
* Root finding is local: the default seeds are the STEP hKBM branch and drift-wave-like guesses,
  with a coarse search of the complex plane when they fail (`--scan` always adds it; the most
  unstable converged root is kept). The two-field model also has shear-Alfven-like roots at
  |omega| ~ 1 c_s/a, mostly in the electron direction (on STEP: omega -0.84 and -2.13, gamma
  ~0.03 at k_y rho_s 0.2), which GENE does not show; a search can land on them. For a new regime
  give `--omega0` (for instance from one GENE run) and check `hkbm.json`.
* Validated against GENE only for the STEP hKBM (STEP-EC-HD, Psi_n = 0.49; k_y rho_s 0.05-0.8,
  the GENE delta B_par channel tests and ten geometry/beta levers).
