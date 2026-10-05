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
