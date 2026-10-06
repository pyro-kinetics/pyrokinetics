# The hKBM dispersion relation solved by `hkbm_solver`

This is the model behind `solver.py` (main model = `solver.DEFAULT`). Two upgrades over the
equations below are part of the main model: exact Bessel ion FLR (J0(x), 2 J1(x)/x and the exact
Maxwellian moments Gamma0, Gamma0 - Gamma1, instead of the exponential forms quoted below), and
collisions: the trapped electrons' bounce-averaged Lorentz (pitch-angle scattering) operator
with nu_D^e(E) = nu_ei [1 + erf(x) - G(x)]/x^3 (x = sqrt(E), e-i plus e-e deflection; Helander &
Sigmar 2002, ch. 3; Hinton & Hazeltine 1976), nu_ei = 4 Z^2 (n_i/n_ref)(T_ref/T_e)^1.5
(m_ref/m_e)^0.5 coll as in GENE (`collisions_common.F90`, `compute_nuei`), H = 0 at the
trapped-passing boundary; trapped ions get Krook detrapping nu_D^i(E)/trpeps; passing particles
are collisionless.

Numbers quoted below (T_i = 1.03, gradients, beta = 0.09, p' = 0.499) are those of the STEP deck;
the solver takes them from the GENE parameters file.

GENE normalised units: lengths a, frequencies c_s/a, k_y in 1/rho_s, T in T_e, n in n_e, B in B_ref, fields in
GENE's normalisation. Time dependence exp(-i omega t); GENE's frequency is omega_GENE = -Re(omega) (> 0 = ion direction),
gamma = Im(omega).

**Symbols.**
* theta: ballooning angle on the central turn |theta| < pi (k_x = 0); dl = J B dtheta (J: GENE Jacobian, B = |B|/B_ref).
* species s = i, e: q_i = 1, q_e = -1; T_e = 1, T_i = 1.03; n_s = 1; v_Ts = sqrt(2 T_s/m_s) (m_i = 1, m_e = 2.7e-4).
* velocity variables: E = v^2/v_Ts^2, lambda = mu/E (mu = v_perp^2/(v_Ts^2 B)), v_par = sigma v_Ts sqrt(E (1 - lambda B)),
  sigma = +-1; trapped iff lambda > 1/B_max. F0 = pi^{-3/2} e^{-E}; d^3v = (pi/2) B E^{1/2} dE dlambda / sqrt(1 - lambda B)
  for each sigma.
* diamagnetic frequency: omega_*s^T(E) = -k_y (T_s/q_s) [a/L_n + a/L_Ts (E - 3/2)] (a/L_n = 1.03, a/L_Te = 1.58, a/L_Ti = 1.82).
* magnetic drift: omega_ds(theta, lambda, E) = k_y (T_s/q_s) E [lambda B K_gradB + 2 (1 - lambda B) K_kappa] / B, with
  K_kappa = K_y - p'/(2B) (curvature), K_gradB = K_kappa + p_beta p'/(2B) (grad-B drift; the p'/(2B) part is beta' in the
  drift), p' = beta sum_s n_s T_s (a/L_n + a/L_Ts) = 0.499 (GENE dpdx), K_y GENE's curvature coefficient, p_beta = 1
  (GENE dpdx_term = 'full_drift'; 0 = 'gradB_eq_curv').
* bounce average: bar[A](lambda) = oint A dl/|v_par| / oint dl/|v_par|; precession omega_de_bar(lambda, E) = bar[omega_de] = E k_y Omega(lambda).
* FLR (ions): b = k_y^2 g^yy T_i / B^2; J0 = exp(-b B lambda E / 2), I1 = 2 J1(x)/x = exp(-b B lambda E / 4)
  (exponential forms; Gamma0 -> 1/(1 + b), Delta_i = <B mu J0 I1> -> (1 + 3b/4)^{-2}); electrons J0 = I1 = 1, Delta_e = 1.
* fields: phi(theta), dB_par(theta), psi(theta) with A_par = -(i/omega) d_l psi (so E_par = -d_l(phi - psi));
  k_perp^2 = k_y^2 g^yy; beta = 0.09. Ideal-MHD reference: dB_par^MHD = -(p'/2B) k_y phi/omega; r := dB_par/dB_par^MHD
  is an OUTPUT.
* switches (GENE bpar_vlasov / bpar_field / bpar_source per species, continuous in [0, 1], all 1 for the physical
  system): v_s (mu dB_par in species s's gyrokinetic equation), f_s (species s's dB_par response in the field
  equations), s_s (species s's delta p_perp in pressure balance).

**Species responses** (h_s = non-adiabatic part, GENE's h; the perturbed distribution is h_s - (q_s/T_s) F0 chi_s,
chi_s = J0 phi + v_s (T_s/q_s) mu I1 dB_par; mu = lambda E):

    ions (integrated along every orbit):
        [ omega - omega_di + i v_par d_l ] h_i = (F0/T_i) (omega - omega_*i^T) ( J0 phi + v_i T_i mu I1 dB_par )
        passing: h_i = 0 where the orbit enters the central turn; trapped: h_i periodic over the bounce orbit
    trapped electrons (omega_be, omega_te >> omega >> omega_de; Zocco et al. 2026 eq. 3.8 with beta' and dB_par):
        h_e = -F0 { (1 - omega_*e^T/omega) psi + (omega - omega_*e^T)/(omega - omega_de_bar) bar[ phi - v_e mu dB_par - (1 - omega_de/omega) psi ] }
    passing electrons (Zocco 3.7):
        h_e = -F0 (1 - omega_*e^T/omega) psi

**Field equations** (QN: quasineutrality; PB: perpendicular pressure balance; TCH: vorticity / parallel-current closure):

    QN :  sum_s q_s [ int d3v J0 h_s - (q_s/T_s) phi + (f_s - v_s) Delta_s dB_par / B ] = 0
    PB :  (2/beta) dB_par + sum_s s_s [ T_s int d3v mu I1 h_s + (f_s - v_s) (2 T_s Delta_s / B^2) dB_par ] = 0
    TCH:  (2/beta) B d_l[ (k_perp^2/B) d_l psi ]
            = omega sum_s (q_s^2/T_s) int d3v F0 (omega - omega_*s^T) J0 chi_s + omega sum_s q_s int d3v J0 omega_ds h_s
              - omega^2 [ sum_s (q_s^2/T_s) phi - sum_s q_s (f_s - v_s) Delta_s dB_par / B ]

With all switches = 1 the (f_s - v_s) terms vanish and these are the standard h-form gyrokinetic field equations
(QN: sum_s q_s int J0 h_s = sum_s (q_s^2/T_s) phi; PB: 2 dB_par/beta = -sum_s T_s int mu I1 h_s). The switch terms make
each GENE channel test exact: e.g. v_i = 0 alone (GENE bpar_vlasov = F for ions) leaves the ions' instantaneous
magnetisation response Delta_i dB_par/B in QN and their 2 T_i Delta_i dB_par/B^2 self term in PB.

**Dispersion relation.** phi, dB_par, psi = sum_{n<8} c_n H_{2n}(theta/sigma) exp(-theta^2/2 sigma^2) (sigma = 0.35 rad,
twisting parity); QN, PB, TCH projected on the same functions with weight J dtheta (bending term by parts):

    D(omega) c = 0,   det D(omega) = 0      (24 x 24 complex matrix; root by secant on det D, continuation in k_y and switches)

Entries are energy integrals: trapped electrons with the plasma dispersion function (Z-function moments, Landau
continuation, Zocco App. B), ions by exact exponential integration along the orbits on a generalised Gauss-Laguerre
energy grid (Im omega > 0 only). ~10 s per root.

## Quasilinear fluxes (`quasilinear.py`)

**Definitions (GENE's nrg).** For one linear mode (k_x = 0 turn, k_y > 0; GENE adds the -k_y
mirror, factor 2), species s, with h_s the non-adiabatic part and the gyroaveraged potential
chi_s = J0 phi - v_par J0 A_par + (T_s/q_s) mu I1 dB_par (v_par in c_s units, mu = v_perp^2/(v_Ts^2 B)):

    Gamma_s = -2 n_s < Re[ conj(int d3v h_s ...) i k_y (field) ] >,   Q_s = the same with T_s E,  E = v^2/v_Ts^2
    phi channel   (Gamma_es, Q_es):  -2 n_s     < Re[ conj(int d3v J0 E^k h_s) i k_y phi ] > T_s^k
    A_par channel (in Gamma_em, Q_em): +2 n_s   < Re[ conj(int d3v v_par J0 E^k h_s) i k_y A_par ] > T_s^k
    dB_par channel (in Gamma_em, Q_em): -2 n_s (T_s/q_s) < Re[ conj(int d3v mu I1 E^k h_s) i k_y dB_par ] > T_s^k

k = 0 particles, k = 1 heat; <f> = int f J dtheta / int J dtheta over the central turn. GENE
(diag.F90 exec_diag_nrg) takes these moments of its gyrocentre f1 = h - (q/T) F0 chi plus the FLR
corrections of flr_corr_ff.F90 (get_mom00_phi = -(q/T)(1 - Gamma0) n, get_mom00_bpar =
Delta01 n/B, get_momI00_phi = (q/T) Delta01 n, get_momI00_bpar = 2 Delta01 n/B, ...). These
corrections turn the f1 moments into the h moments above, plus real multiples of the same
field (-(q/T) n phi in the density, ...). Such multiples carry no flux, so GENE's phi/em split is
the h-based split above. GENE's em column is A_par + dB_par. The solver reports the two separately.

**Moments from the model.**
* Ions: h_i is integrated along every orbit (`ion_stream.StreamingIons.flux_moments`, the same
  quadrature as the dispersion relation). The two directions sigma = +-1 are kept separately for
  the v_par moment.
* Electrons, even part: h_e = -F0 (1 - omega_*e^T/omega) psi (all electrons) plus the trapped
  response, which is bounce-averaged (energy quadrature with collisions, Z-function moments
  without). Its moments are the theta <- lambda maps of the matrix (G0, Gl, Gd) applied to the
  energy moments K_j of the eigenmode's trapped source, one order higher in E for the heat flux.
  Maxwellian moments: <E^k> = Gamma(k + 3/2)/Gamma(3/2), <(v_perp/v_T)^2 E^k> = (2/3) <E^(k+1)>,
  <lambda B> = 2/3 at fixed E.
* Electrons, parallel moments (A_par channel): the bounce-averaged model has no odd part. The
  moments of the drift-kinetic equation (omega - omega_d) h + i v_par d_l h = (q F0/T)(omega -
  omega_*^T) chi give, with Gamma_k = int d3v v_par E^k h_e,

        i B d_l(Gamma_k/B) = R_k = -S_k phi + T_k dB_par/B - omega int E^k h_e + int E^k omega_de h_e,
        S_k = (omega - a_e) <E^k> - b_e <E^(k+1)>,   T_k = (omega - a_e) N_k - b_e N_(k+1),

  (omega_*e^T = a_e + b_e E). This is integrated along theta with the integration constant set
  by Gamma(-pi)/B(-pi) = -Gamma(pi)/B(pi), which is Gamma(0) = 0 for a twisting-parity mode.
  Summed over species with charges, the k = 0 equation is the vorticity (TCH) equation the
  solver imposes. Since that equation holds only projected on the basis, the A_par particle
  fluxes of ions and electrons agree only approximately (1-50 % of this small channel on STEP,
  growing with k_y). The phi and dB_par particle fluxes are ambipolar to round-off: QN and PB are
  imposed on test functions that span phi and dB_par.
* A_par = -(i/omega) d_l psi, dl = J B dtheta.

**Saturation (mixing length).** <|phi|^2>(k_y) = C (gamma/<k_perp^2>)^2, with <k_perp^2> = int
k_y^2 g^yy |phi|^2 J / int |phi|^2 J (solver units: rho_s, c_s/L_ref). The fluxes are the trapezoid
integral over k_y rho_s of weight x <|phi|^2>. One constant C is fitted to nonlinear fluxes (see the
analysis note QL_FLUX.md); modes without a growing root contribute nothing.

**theta0 != 0.** A mode with k_x = shat k_y theta0 sees k_y K_y -> k_y (K_y + kappa K_x) and
k_y^2 g^yy -> k_y^2 (g^yy + 2 kappa g^xy + kappa^2 g^xx), with kappa = k_x/k_y = -theta0 d(g^xy/g^xx)/dtheta
(mean secular slope over the turn, = C_y q0 shat/r0), and GENE's K_x = -(ga2/ga1) dBdz/C_xy. This puts
the k_perp minimum near theta0. Because of STEP's reversed local shear at the outboard midplane,
the minimum sits on the other side of theta = 0 for small theta0. The twisting parity is lost,
so the basis gets the odd Hermite functions too (16 per field). gamma(theta0) = gamma(-theta0)
holds to 1e-3 for up-down symmetric geometry. GENE's radial modes connect every 2 pi kappa k_y,
so a GENE run with kx_center has theta0 = kx_center/(kappa k_y). Against such GENE runs (STEP,
k_y rho_s 0.2/0.3, theta0 0.07-1) gamma agrees within ~20 % up to theta0 ~ 0.3 (README).

## Microtearing branch (`mtm.py`)

`mtm.py` solves the collisionless gyrokinetic microtearing dispersion relation of Chandran &
Schekochihin (J. Plasma Phys. 2024, arXiv:2211.02103). It is used as the tearing-parity branch in
`run_linear(..., modes=("hkbm", "mtm"))`.

**Model.**

- Ampere's law at theta ~ 1 fixes A_par = C B/k_perp^2. A_par enters the rest of the problem only
  through its line integral psi_inf.
- The dispersion relation (C&S 2.39) is

      omega - omega_*e (1 + eta_e/2) + i sqrt(pi) [v_Te/L + omega^2 B_max/(2 v_Te) int J Gamma dPhi] = 0.

- dPhi comes from quasineutrality at |theta| >> 1 (C&S 2.46). This is an integral equation along the
  extended ballooning angle, out to |theta| ~ 1/(k_perp rho_e). The ions are Boltzmann
  (tau = T_i/T_e), and W_p is the passing-electron kernel. The equation is solved by GMRES. The W_p
  product uses a blocked propagator recursion.
- The trapped-electron kernel (C&S A8) is an option and is off in production. Its bounce-averaged
  precession resonance is not resolved near the real axis.
- Krook collisions on the passing electrons (omega -> omega + i nu_D^e(E)) are also an option. They
  can only damp: a Krook does not change the layer-integrated current, so it cannot give the
  classical collisional drive.

**Geometry.** This is GENE's Miller/MXH geometry. Turn j of the extended angle is the central turn
of a mode with ballooning angle theta0 - 2 pi j, the same shift as `Geo.shifted`. The magnetic drift
is GENE's full_drift form with the electron charge.

**Root finding.** A low-resolution secant search gives the no-root verdict in 0.1-2.5 s; the cost
grows with the number of turns, that is, at low k_y. A production-resolution secant search follows
(0.2-3 s per root).

**Tests against the paper** (Patel et al. 2022 Table 2 surface, `tests/hkbm_solver/test_mtm.py`).

These agree with the paper:

- B range, the averaged beta_e, the omega_0 line of Fig. 3, and the amplitudes of Fig. 5;
- the exact limits: eta_e = 0, the dPhi term off, and the cold-ion closed form.

The growth rates of C&S Fig. 3 are reproduced quantitatively (omega_r within 7 %, gamma within
5-40 %, the collapse near k rho_e ~ beta_e included), and the sign structure of Fig. 5 with them,
**only with the electron magnetic drift reversed** relative to GENE's convention (`drift_sign=-1`,
a diagnostic). Removing any of the following does not reproduce them:

- the pressure term from the drift;
- beta' altogether;
- the radial (K_x) part of the drift.

A possible cause on the paper's side is an electron drift built from ion-normalised geometry
coefficients without the charge sign: for example, GS2-style gbdrift/cvdrift used for electrons as
for ions. This is a hypothesis, not a confirmed fact.

The reversed-drift tests therefore check that the implementation solves the paper's equations; the
physical sign is the default. Two checks support it:

- omega_De(theta = 0)/omega_*e > 0, which is bad curvature at the outboard midplane, holds on a
  circular deck and on the Patel surface. The formula is the same as the solver's validated
  electron precession.
- On a circular deck the extended drift follows the s-alpha form cos theta + s theta sin theta.

On the Patel surface the transit-averaged passing-electron drift is opposite to omega_*e (shaping:
kappa 3, delta 0.45, R/a 1.79), with or without beta'.

**Collisionless result with GENE's (physical) drift sign.**

- On the Patel surface, on STEP (Fig. 16 of Kennedy et al. 2023) and on a CBC-like circular deck
  (k_y rho_s 0.1-0.5) there is **no growing collisionless MTM**.
- omega_r stays within a few per cent of omega_*e(1 + eta_e/2). GENE's STEP MTMs sit at this
  frequency to <1 %, so it agrees.
- GENE's weak STEP MTMs (gamma ~ 0.005-0.01 c_s/a) are therefore not given by this branch. They
  need a collisional (Lorentz) passing-electron model, which is not implemented.
- On STEP the reversed sign gives omega 9 % off GENE's and gamma 10x too high.

**Flags.** The MTM records carry `checks`:

- converged and growing;
- electron_direction;
- ordering: k_perp rho_e/beta_e < 0.3;
- phi_decays.

`no_root_verdict` is True when the low-resolution search finds no growing root.
