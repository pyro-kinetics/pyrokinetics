"""
Fast eigenvalue solver for the hybrid kinetic ballooning mode (hKBM): a reduced gyrokinetic
dispersion relation (Zocco, Rodriguez & Edmiston, J. Plasma Phys. 92, E39 (2026), extended)
with a free eigenfunction in the ballooning angle, delta B_par as a field from perpendicular
pressure balance, ions integrated along the field line (ion_stream.py) and GENE's delta B_par
channel switches as continuous multipliers.  The equations are written out in the package
README ("Physics model").

Main model (solve() defaults, DEFAULT): ions='stream', bpar='field', psi=True,
basis=('hermite', 8, 0.35), flr='bessel' (exact ion J0, 2J1/x), coll=COLL_NOMINAL (electron
pitch-angle scattering by the bounce-averaged Lorentz operator, nu_ei from GENE's coll;
trapped-ion Krook detrapping).  DEFAULT_V1 = collisionless with exponential FLR.
    from pyrokinetics.hkbm_solver.solver import solve
    r = solve(0.2)                # STEP deck -> r['omega_gene'], r['gamma'], r['phi'], r['bpar'], r['psi'], ...
    r = solve(0.3, vl_i=0.5)      # ions feel half of their mu dB_par term
For a GENE parameters file use pyrokinetics.hkbm_solver.run() (gene_io.py), which sets params, geometry,
coll and the switches from the deck.

Units/signs: GENE normalised units with T_ref = T_e, n_ref = n_e, m_ref = m_i, B_ref, L_ref
(frequencies in c_s/L_ref, k_y in 1/rho_s); exp(-i omega t) inside the code ("model sign",
omega_r < 0 = ion direction); results also in GENE's sign: omega_gene = -Re(omega), gamma = Im(omega).

Options (Solver(ky, ...)):
  ions   'stream' (along-the-orbit kinetic ions, Im omega > 0 only), 'zocco' (local drift-kinetic, Zocco 3.6),
         'kpar' (plane-wave streaming surrogate), 'adiabatic' (h_i = (F0/T_i) chi), 'none'
  bpar   'field' (perpendicular pressure balance), 'closure' (dB_par = r dB_par^MHD, r input), 'off'
  psi    two-field (QN + PB + TCH vorticity equation, psi free) or one-field (passing electrons adiabatic)
  basis  ('hermite', N, sigma) even Hermite-Gaussians | dict of arrays on the theta grid
  switches (multipliers, default 1): vl_e, vl_i (GENE bpar_vlasov), fd_e, fd_i (bpar_field; split as fdq_s = QN
         magnetisation term C13 and fdp_s = PB self term C33), src_e, src_i (bpar_source), bp_e, bp_i (beta' in the
         grad-B drift, 1 = full_drift, 0 = gradB_eq_curv)
  stream_kw dict for ion_stream.StreamingIons (quadrature sizes, ion_psi=True: ions respond to A_par too)
  flr 'exp' (Solver default) | 'lin' | 'bessel' (exact; streaming/adiabatic ions; DEFAULT); old_pol, zocco_flr:
         regression only
  coll   GENE's coll (0 = collisionless, Solver default; COLL_NOMINAL = 1.35e-4 in DEFAULT): nu_ei = 4 coll/sqrt(m_e)
         (nu_ei_gene); coll_model 'lorentz' (bounce-averaged Lorentz operator on the trapped electrons, H = 0 at the
         trapped-passing boundary) | 'krook' (nu_D/eps, eps = trpeps); coll_ee (e-e deflection, default True);
         coll_i (trapped-ion Krook detrapping nu_D^i/eps, default True); nE_e (electron energy nodes, 96)
  miller GENE miller.dat path or geometry.Geo object (default: the STEP deck); geom: a geometry.Geometry;
         params: dict(Ti, omn, omte, omti, beta, me) in the units above (default PARAMS = STEP)
Root finding: secant on det of the row-scaled Galerkin matrix D(omega); continuation() follows a root in any parameter
(adaptive steps; gamma_floor stops paths of models without Landau continuation).
"""

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import erf, eval_hermite, factorial, roots_legendre

from . import kernels as Z  # Mint, Mint_kpar (energy integrals)
from .geometry import Geo, Geometry

SQPI = np.sqrt(np.pi)
_GEO = {}
ME_DEFAULT = 0.00027230851074233294  # m_e/m_ref of the STEP deck
COLL_NOMINAL = 0.000135  # GENE coll of the STEP deck
# STEP-EC-HD Psi_n = 0.49 deck (Kennedy et al., Nucl. Fusion 63, 126061 (2023)) in the solver's units
# (T_ref = T_e, n_ref = n_e, m_ref = m_i, L_ref = a): the default plasma parameters
PARAMS = dict(
    Ti=1.03,
    omn=1.027,
    omte=1.578,
    omti=1.822,
    beta=0.09,
    q0=3.5,
    R0=1.962,
    me=ME_DEFAULT,
)
# its Miller geometry (GENE &geometry; amhd and dpdx_pm as GENE resolves them from beta and the gradients, written
# with 8 digits as in GENE's parameters.dat: amhd = -1, dpdx_pm = -1 give the same values to 1e-7 relative)
STEP_GEOMETRY = dict(
    magn_geometry="miller",
    q0=3.5,
    shat=1.2,
    amhd=11.982455,
    dpdx_pm=0.4985523,
    dpdx_term="full_drift",
    delta=0.283,
    drr=-0.399,
    drz=0.0,
    kappa=2.56,
    major_r=1.962,
    major_z=0.0,
    minor_r=1.0,
    s_delta=0.292,
    s_kappa=0.015,
    trpeps=0.32446813859776175,
)
STEP_SPECIES = [
    dict(
        name="electron",
        charge=-1,
        dens=1.0,
        mass=ME_DEFAULT,
        omn=1.027,
        omt=1.578,
        temp=1.0,
    ),
    dict(name="ion1", charge=1, dens=1.0, mass=1.0, omn=1.027, omt=1.822, temp=1.03),
]


def step_geo(nz0=512):
    """GENE's Miller geometry of the STEP deck at nz0 points (Python port of GENE's get_miller)."""
    from .miller import miller_from_namelist

    nml = dict(geometry=STEP_GEOMETRY, general=dict(beta=0.09), species=STEP_SPECIES)
    return Geo.from_miller(miller_from_namelist(nml, nz0=nz0))


def geometry(nth=1024, miller=None):
    """Central-turn geometry; miller: None (STEP deck, GENE's Miller geometry computed in Python at
    nz0 = 512), a GENE miller.dat path, or a Geo object; cached for None and paths."""
    if isinstance(miller, Geo):
        return Geometry(miller, nth=nth)
    key = (nth, miller)
    if key not in _GEO:
        _GEO[key] = Geometry(
            step_geo() if miller is None else Geo.from_file(miller), nth=nth
        )
    return _GEO[key]


# ---------------------------------------------------------------------- collisions (pitch-angle scattering)
def nu_ei_gene(coll, me=ME_DEFAULT, Te=1.0, ni=1.0, Z=1.0):
    """GENE's thermal electron-ion collision frequency in c_ref/L_ref (= c_s/a here), from GENE's coll
    (src/collisions_common.F90 compute_nuei; GENE doc gene.tex, 'coll': nu_c = pi lnLambda e^4 n_ref L_ref/(2^1.5 T_ref^2)):
        nu_ei = 4 Z^2 (n_i/n_ref) (T_ref/T_e)^1.5 (m_ref/m_e)^0.5 coll,
    = the Hinton & Hazeltine (1976, eqs. 4.12, 4.36) electron-ion rate nu_ei(v) at v = v_Te = sqrt(2 T_e/m_e), whose
    v-dependence nu_ei (v_Te/v)^3 is the e-i pitch-angle scattering (deflection) frequency.  STEP: coll = 1.35e-4
    -> 0.032724 (GENE writes 'nu_ei = 0.032724' to parameters.dat)."""
    return 4 * Z**2 * ni / Te**1.5 / np.sqrt(me) * coll


def _erf_minus_G(x):
    """erf(x) - G(x), G(x) = [erf(x) - x erf'(x)]/(2 x^2) the Chandrasekhar function (small-x limit 4x/(3 sqrt pi))."""
    x = np.asarray(x, float)
    xs = np.where(x < 1e-3, 1.0, x)
    G = (erf(xs) - xs * 2 / SQPI * np.exp(-(xs**2))) / (2 * xs**2)
    return np.where(x < 1e-3, 4 * x / (3 * SQPI), erf(xs) - G)


def nuD_e(E, nu_ei, ee=True, zeff=1.0):
    """Electron deflection frequency nu_D^e(E) = nu_ei [Z_eff + (erf(x) - G(x))]/x^3, x = v/v_Te = sqrt(E)
    (Helander & Sigmar, Collisional Transport in Magnetized Plasmas, CUP 2002, ch. 3: nu_D^ab = nu_ab
    [erf(x_b) - G(x_b)]/x_a^3; e-i with infinitely heavy ions -> nu_ei/x^3).  ee=False: e-i only.
    """
    x = np.sqrt(E)
    return nu_ei * (zeff + (_erf_minus_G(x) if ee else 0.0)) / x**3


def nuD_i(E, coll, Ti, mi=1.0, Z=1.0, ni=1.0):
    """Ion-ion deflection frequency, GENE normalisation (nu_ei formula with ion quantities):
    4 Z^4 n_i coll/(T_i^1.5 m_i^0.5) [erf(x) - G(x)]/x^3, x = v/v_Ti = sqrt(E)."""
    x = np.sqrt(E)
    return 4 * Z**4 * ni * coll / (Ti**1.5 * np.sqrt(mi)) * _erf_minus_G(x) / x**3


def hermite_basis(theta, N, sigma, parity="even"):
    """Hermite-Gaussian functions H_n(theta/sigma) exp(-theta^2/2 sigma^2), n even (twisting parity) or all."""
    ns = [2 * k for k in range(N)] if parity == "even" else list(range(N))
    x = theta / sigma
    out = []
    for n in ns:
        f = eval_hermite(n, x) * np.exp(-(x**2) / 2) / np.sqrt(2.0**n * factorial(n))
        out.append(f + 0j)
    return np.array(out)


class Solver:
    """Reduced hKBM eigenproblem at one k_y.

    ions: 'zocco' (local drift kinetic, 3.6), 'kpar' (plane-wave streaming), 'adiabatic' (h_i = (F0/T_i) chi:
          polarisation + instantaneous dB_par response only), 'none'.
    bpar: 'closure' (dB_par = r dB_par^MHD, r complex allowed), 'field' (perpendicular pressure balance), 'off'.
    psi:  False (hybrid: passing electrons adiabatic) or True (two-field QN + TCH with psi free).
    switches (continuous multipliers, default 1): vl_e, vl_i (mu dB_par in the species' GK equation),
          fd_e, fd_i (species' dB_par response in the field equations, GENE bpar_field: C13, C33),
          src_e, src_i (species' delta p_perp in pressure balance, GENE bpar_source), bp_e, bp_i (fraction of beta'
          in the grad-B drift: 1 = GENE full_drift, 0 = gradB_eq_curv).
    basis: ('hermite', N, sigma) | ('gene', run) | dict(phi=[...], bpar=[...], psi=[...]) arrays on geo.theta.
    old_pol: use the one-field model's original polarisation (1 - b)/T_i + 1 (regression only).
    zocco_flr: J0 I1 -> 1 in the ion dB_par coupling (as the original one-field model) instead of 1 - (3/4) b B lam E (regression only).
    """

    def __init__(
        self,
        ky,
        ions="adiabatic",
        bpar="field",
        r=0.7,
        psi=False,
        basis=("hermite", 6, 0.2),
        parity="even",
        nth=1024,
        nlam=120,
        nt_e=200,
        nt_i=160,
        kpar=None,
        old_pol=False,
        zocco_flr=False,
        flr="exp",
        params=PARAMS,
        theta_max=None,
        stream_kw=None,
        coll=0.0,
        coll_model="lorentz",
        coll_ee=True,
        coll_i=True,
        coll_eps=None,
        nE_e=96,
        miller=None,
        geom=None,
        r_eff=None,
        **sw,
    ):
        self.ky, self.ions, self.bpar, self.r, self.use_psi = ky, ions, bpar, r, psi
        (
            self.coll,
            self.coll_model,
            self.coll_ee,
            self.coll_i,
            self.coll_eps,
            self.nE_e,
        ) = (float(coll), coll_model, coll_ee, coll_i, coll_eps, nE_e)
        assert coll_model in ("lorentz", "krook"), coll_model
        self.p = p = params
        self.sw = dict(
            vl_e=1.0,
            vl_i=1.0,
            fd_e=1.0,
            fd_i=1.0,
            src_e=1.0,
            src_i=1.0,
            bp_e=1.0,
            bp_i=1.0,
            fdq_e=None,
            fdq_i=None,
            fdp_e=None,
            fdp_i=None,
            a_e=1.0,
            a_i=1.0,
            d_e=1.0,
            d_i=1.0,
        )
        for k, v in sw.items():
            if k not in self.sw:
                raise KeyError(k)
            self.sw[k] = float(v)
        # precession vs direct drive test: the mu dB_par term of species s split
        # into a precession shift (a_s: omega_ds -> omega_ds - a_s r_eff beta'-part of omega_ds, acting on the phi
        # response) and the direct-drive remainder (d_s); a_s = d_s = 1 is the main model, a_s = d_s = 0 is vl_s = 0.
        # The vl_s-carrying algebraic terms (dB_par source of h, (f - v) terms, TCH source moment) take vl_s d_s.
        self.r_eff = r_eff
        self.dec = {
            s_: (self.sw["a_" + s_] != 1.0 or self.sw["d_" + s_] != 1.0)
            for s_ in ("e", "i")
        }
        for s_ in ("e", "i"):
            if self.dec[s_]:
                assert (
                    r_eff is not None and bpar == "field"
                ), "a_s/d_s need r_eff and bpar=field"
                assert (
                    s_ == "e" or ions == "stream"
                ), "ion a_i/d_i implemented for ions=stream"
                self.sw["vl_" + s_] *= self.sw["d_" + s_]
        for s_ in (
            "e",
            "i",
        ):  # fd split: fdq = QN magnetisation term (C13), fdp = PB self term (C33); default fd
            for t_ in ("fdq_", "fdp_"):
                if self.sw[t_ + s_] is None:
                    self.sw[t_ + s_] = self.sw["fd_" + s_]
        self.old_pol, self.zocco_flr, self.kpar, self.flr = (
            old_pol,
            zocco_flr,
            kpar,
            flr,
        )
        assert flr in ("exp", "lin", "bessel") and not (
            flr == "bessel" and ions in ("zocco", "kpar")
        ), "flr='bessel' is implemented for ions='stream' (and 'adiabatic'); local Zocco/kpar ions use 'exp'/'lin'"
        self.geo = geo = geom if geom is not None else geometry(nth, miller)
        self.wgt = geo.J * geo.dth
        Ti = p["Ti"]
        self.b = (
            ky**2 * geo.gyy * Ti / geo.B**2
        )  # ion b = k_perp^2 rho_i^2 / 2 (GENE units)
        self.ae = ky * (p["omn"] - 1.5 * p["omte"])
        self.be = ky * p["omte"]  # omega_*e^T = ae + be E
        self.ai = -ky * Ti * (p["omn"] - 1.5 * p["omti"])
        self.bi = -ky * Ti * p["omti"]
        self.Kc = geo.Kc
        self.Kg_e = geo.Kc + self.sw["bp_e"] * geo.dpdx / (2 * geo.B)
        self.Kg_i = geo.Kc + self.sw["bp_i"] * geo.dpdx / (2 * geo.B)
        self.bmhd = -geo.dpdx / (2 * geo.B)  # dB_par^MHD = bmhd * ky phi / omega
        # ---- basis
        self.basis_name = basis
        self._make_basis(basis, parity)
        self.nlam, self.nt_e, self.nt_i = nlam, nt_e, nt_i
        self._setup_trapped()
        self._setup_collisions()
        self._setup_ions()
        self.last_vec = None
        if ions == "stream":
            from . import ion_stream

            if (
                bpar == "closure"
            ):  # ion dB_par source = (r ky/omega) * bmhd * phi: carry bmhd * phi basis
                self.Bb = self.bmhd[None, :] * self.Bphi
            skw = dict(stream_kw or {})
            if self.flr == "bessel":
                skw.setdefault("bessel", True)
            self.stream = ion_stream.StreamingIons(self, **skw)

    # ------------------------------------------------------------------ basis
    def _make_basis(self, basis, parity):
        th = self.geo.theta
        if isinstance(basis, dict):
            Bphi = np.atleast_2d(basis["phi"])
            Bb = np.atleast_2d(basis.get("bpar", basis["phi"]))
            Bpsi = np.atleast_2d(basis.get("psi", basis["phi"]))
        elif basis[0] == "hermite":
            H = hermite_basis(th, basis[1], basis[2], parity)
            Bphi = Bb = Bpsi = H
        else:
            raise ValueError(basis)
        self.Bphi, self.Bb, self.Bpsi = Bphi, Bb, Bpsi
        self.N = Bphi.shape[0]

    # ------------------------------------------------------------------ trapped electrons: geometry-only setup
    def _setup_trapped(self):
        geo = self.geo
        g = geo.g
        lmin, lmax = 1 / geo.Bmax, 1 / geo.Bmin
        s = np.linspace(0, 1, self.nlam + 2)[1:-1]
        lam = np.append(lmin + (lmax - lmin) * s, lmax * (1 - 1e-7))
        self.lam = lam
        # bounce quadrature points / weights for each lambda (z = zc + hw sin phi)
        nphi = 600
        zs, ws = [], []
        for l in lam:  # noqa: E741
            z1, z2 = geo.bounce_points(l)
            zc, hw = 0.5 * (z1 + z2), 0.5 * (z2 - z1)
            ph = (np.arange(nphi) + 0.5) / nphi * np.pi - np.pi / 2
            z = zc + hw * np.sin(ph)
            vp2 = np.clip(1 - l * g.sB(z), 1e-300, None)
            w = g.sJ(z) * g.sB(z) / np.sqrt(vp2) * hw * np.cos(ph)
            zs.append(z)
            ws.append(w / w.sum())
        zs, ws = np.array(zs), np.array(ws)  # (nlam, nphi)
        Bz = g.sB(zs)
        Kz = g.sK(zs)
        Kcz = Kz - geo.dpdx / (2 * Bz)
        Kge = Kcz + self.sw["bp_e"] * geo.dpdx / (2 * Bz)
        lB = lam[:, None] * Bz
        Dz = (
            lB * Kge + 2 * (1 - lB) * Kcz
        ) / Bz  # electron drift coefficient: omega_de = -E ky D
        self.Omega = -np.sum(ws * Dz, axis=1)  # omega_de_bar = E ky Omega
        # beta' part of the (bounce-averaged) drift, independent of bp_e: Omega(shift s) = Omega + s Omega_bp
        self.Omega_bp = np.sum(ws * lam[:, None] * geo.dpdx / (2 * Bz), axis=1)
        self._bz, self._bw, self._bD, self._bP = zs, ws, Dz, -self.bmhd_at(Bz)

        def bavg(F, extra=None):
            """bounce averages of each row of F (on geo.theta) -> (nrows, nlam)."""
            out = []
            for f in F:
                fr = CubicSpline(
                    np.append(geo.theta, np.pi),
                    np.append(f.real, f.real[0]),
                    bc_type="periodic",
                )
                fi = CubicSpline(
                    np.append(geo.theta, np.pi),
                    np.append(f.imag, f.imag[0]),
                    bc_type="periodic",
                )
                v = fr(zs) + 1j * fi(zs)
                if extra is not None:
                    v = v * extra
                out.append(np.sum(ws * v, axis=1))
            return np.array(out)

        self.bar_phi = bavg(self.Bphi)  # bar[b_n]
        self.bar_Pphi = bavg(
            self.Bphi, extra=geo.dpdx / (2 * Bz)
        )  # bar[(dpdx/2B) b_n] (closure)
        self.bar_b = bavg(self.Bb)  # bar[dB_par basis]
        self.bar_psi = bavg(self.Bpsi)
        self.bar_Dpsi = bavg(self.Bpsi, extra=Dz)
        # theta <- lambda maps: int_{1/Bmax}^{1/B(theta)} B dlam/sqrt(1 - lam B) [w] K(lam) = 2 int_0^tmax dt [w] K
        nt = self.nt_e
        tmax = np.sqrt(np.clip(1 - geo.B / geo.Bmax, 0, None))
        t = (np.arange(nt) + 0.5) / nt
        tt = t[None, :] * tmax[:, None]
        lamB = 1 - tt**2
        lamq = np.clip(lamB / geo.B[:, None], lam[0], lam[-1])
        j = np.clip(np.searchsorted(lam, lamq) - 1, 0, lam.size - 2)
        fr = (lamq - lam[j]) / (lam[j + 1] - lam[j])
        base = (2 * tmax / nt)[:, None] * np.ones_like(fr)
        ue = (
            -self.ky
            * (lamB * self.Kg_e[:, None] + 2 * (1 - lamB) * geo.Kc[:, None])
            / geo.B[:, None]
        )
        nthe = geo.theta.size

        def build(wt):
            G = np.zeros((nthe, lam.size))
            rows = np.repeat(np.arange(nthe), nt)
            np.add.at(G, (rows, j.ravel()), (wt * (1 - fr)).ravel())
            np.add.at(G, (rows, (j + 1).ravel()), (wt * fr).ravel())
            return G

        self.G0 = build(base)  # density
        self.Gl = build(base * lamq)  # mu = lam E moment (pressure)
        self.Gd = build(base * ue)  # local drift u_e moment (TCH)
        # beta' part of the local drift weight: u_e(shift s) = u_e + s ky lam p'/(2B)  (Gd(s) = Gd + s Gbp)
        self.Gbp = build(base * self.ky * lamB * geo.dpdx / (2 * geo.B[:, None] ** 2))

    def bmhd_at(self, B):
        return -self.geo.dpdx / (2 * B)

    # ------------------------------------------------------------------ collisions: setup
    def _setup_collisions(self):
        """Pitch-angle scattering (coll > 0).  Electrons: energy grid in x = sqrt(E) (Gauss-Legendre on [0, 6]) and,
        for coll_model 'lorentz', the bounce-averaged Lorentz operator on the trapped lambda grid,
            bar[C(H)] = nu_D^e(E) (2/tau(lam)) d/dlam [lam I(lam) dH/dlam],
            tau = int J B dz/xi, I = int J xi dz between the bounce points, xi = sqrt(1 - lam B),
        with H = 0 at the trapped-passing boundary lam = 1/B_max (the passing electrons' non-adiabatic part is zero in
        this model) and zero flux at lam = 1/B_min; conservative finite volumes on self.lam.  'krook': bar[C(H)] =
        -nu_D^e(E)/eps H, eps = coll_eps or GENE's trpeps.  Ions (coll_i): Krook detrapping nu_D^i(E)/eps of the trapped
        ions in ion_stream (ions integrated along the orbit); passing ions collisionless.
        """
        self.nu_ei = (
            nu_ei_gene(self.coll, me=self.p.get("me", ME_DEFAULT))
            if self.coll > 0
            else 0.0
        )
        self.ion_nu = None
        self.kappa0 = None
        if self.coll <= 0:
            return
        geo = self.geo
        g = geo.g
        eps = self.coll_eps if self.coll_eps is not None else float(g.h["trpeps"])
        self.eps_coll = eps
        if self.coll_i:
            Ti, coll = self.p["Ti"], self.coll
            self.ion_nu = lambda E: nuD_i(E, coll, Ti) / eps
        x, wx = roots_legendre(self.nE_e)
        xmax = 6.0
        x = 0.5 * xmax * (x + 1)
        wx = 0.5 * xmax * wx
        self.eE = x**2
        self.eW = (
            2 * x**2 * np.exp(-(x**2)) * wx
        )  # int dE E^1/2 e^-E f(E) = sum eW f(eE)
        self.e_nuD = nuD_e(self.eE, self.nu_ei, ee=self.coll_ee)
        if self.coll_model == "krook":
            self.e_nuK = self.e_nuD / eps
            return
        lam = self.lam
        lc = 1 / geo.Bmax

        def tauI(l, nphi=600):  # noqa: E741
            z1, z2 = geo.bounce_points(l)
            zc, hw = 0.5 * (z1 + z2), 0.5 * (z2 - z1)
            ph = (np.arange(nphi) + 0.5) / nphi * np.pi - np.pi / 2
            z = zc + hw * np.sin(ph)
            xi = np.sqrt(np.clip(1 - l * g.sB(z), 1e-300, None))
            dz = hw * np.cos(ph) * np.pi / nphi
            return np.sum(g.sJ(z) * g.sB(z) / xi * dz), np.sum(g.sJ(z) * xi * dz)

        lprev = np.concatenate([[lc], lam[:-1]])
        lh = 0.5 * (lprev + lam)  # half nodes j - 1/2 (j = 0: between lam_c and lam_0)
        tau = np.array([tauI(l)[0] for l in lam])  # noqa: E741
        Ih = np.array([tauI(l)[1] for l in lh])  # noqa: E741
        am = lh * Ih / (lam - lprev)  # a_{j-1/2}
        ap = np.append(am[1:], 0.0)  # a_{j+1/2}; zero flux at lam_max
        lnext = np.append(lam[1:], lam[-1])
        c = 2 / (tau * 0.5 * (lnext - lprev))
        self.L_lo, self.L_up, self.L_dg = c * am, c * ap, c * (am + ap)
        self.e_tau, self.e_I = tau, Ih
        Lm = (
            np.diag(self.L_dg) - np.diag(self.L_lo[1:], -1) - np.diag(self.L_up[:-1], 1)
        )
        self.kappa0 = float(
            np.min(np.linalg.eigvals(Lm).real)
        )  # slowest detrapping rate / nu_D (cf. 1/eps)

    @staticmethod
    def _tridiag(lo, dg, up, rhs):
        """Batched Thomas algorithm: lo, dg, up (nE, n) (row j: lo[j] x_{j-1} + dg[j] x_j + up[j] x_{j+1}),
        rhs (nE, nb, n) -> x (nE, nb, n).  Diagonally dominant for Im omega > 0."""
        n = dg.shape[-1]
        cp = np.empty_like(dg)
        dp = np.empty_like(rhs)
        cp[:, 0] = up[:, 0] / dg[:, 0]
        dp[..., 0] = rhs[..., 0] / dg[:, 0, None]
        for j in range(1, n):
            den = dg[:, j] - lo[:, j] * cp[:, j - 1]
            cp[:, j] = up[:, j] / den
            dp[..., j] = (rhs[..., j] - lo[:, j, None] * dp[..., j - 1]) / den[:, None]
        x = np.empty_like(rhs)
        x[..., -1] = dp[..., -1]
        for j in range(n - 2, -1, -1):
            x[..., j] = dp[..., j] - cp[:, j, None] * x[..., j + 1]
        return x

    def _eH(self, omega, c0, c1, Omega=None):
        """Collisional trapped-electron response H(E, n, lam) (nE, nb, nlam) to the bounce-averaged source
        (omega - omega_*e^T(E)) (c0 + E c1):  (omega - E ky Omega(lam)) H - i bar[C(H)] = source.
        Omega: precession override (precession vs direct drive test), default self.Omega.
        """
        E = self.eE[:, None, None]
        S = (omega - self.ae - self.be * E) * (c0[None] + E * c1[None])
        w = self.ky * (self.Omega if Omega is None else Omega)
        if self.coll_model == "krook":
            den = omega + 1j * self.e_nuK[:, None] - self.eE[:, None] * w[None, :]
            return S / den[:, None, :]
        nu = self.e_nuD[:, None]
        dg = omega - self.eE[:, None] * w[None, :] + 1j * nu * self.L_dg[None, :]
        return self._tridiag(
            -1j * nu * self.L_lo[None, :], dg, -1j * nu * self.L_up[None, :], S
        )

    # ------------------------------------------------------------------ ions: geometry-only setup
    def _setup_ions(self):
        geo, p = self.geo, self.p
        Ti = p["Ti"]
        nt = self.nt_i
        t = (np.arange(nt) + 0.5) / nt
        self.ti_lamB = lamB = 1 - t**2
        self.ti_w = (
            self.ky
            * Ti
            * (
                lamB[None, :] * self.Kg_i[:, None]
                + 2 * (1 - lamB[None, :]) * geo.Kc[:, None]
            )
            / geo.B[:, None]
        )
        self.ti_lam = lamB[None, :] / geo.B[:, None]
        # only theta where some basis function lives (speed): |b| > 1e-10 max
        amp = (
            np.max(np.abs(self.Bphi), axis=0)
            + np.max(np.abs(self.Bb), axis=0)
            + np.max(np.abs(self.Bpsi), axis=0)
        )
        self.ion_mask = amp > 1e-9 * amp.max()
        if self.ions == "kpar" and self.kpar is None:
            ph = self.Bphi[0]
            dphi = np.gradient(ph, geo.theta) / (geo.J * geo.B)
            self.kpar = float(
                np.sqrt(
                    np.sum(geo.J * geo.B * np.abs(dphi) ** 2)
                    / np.sum(geo.J * geo.B * np.abs(ph) ** 2)
                )
            )

    # ------------------------------------------------------------------ species kernels at given omega
    def ion_coeffs(self, omega):
        """Local ion moments per theta: H_i = A_pp phi + A_pb dB_par, P_i = int mu I1 h_i = A_bp phi + A_bb dB_par
        (dB_par entries already include the v_i multiplier)."""
        geo, p = self.geo, self.p
        Ti = p["Ti"]
        vl = self.sw["vl_i"]
        nth = geo.theta.size
        b = self.b
        if self.ions in ("none", "stream"):
            z = np.zeros(nth, complex)
            return z, z, z, z
        G0, D01, D2 = self.flr_factors()
        if self.ions == "adiabatic":
            # h_i = (F0/T_i) chi: the omega -> infinity ion response (polarisation + instantaneous dB_par response)
            App = G0 / Ti + 0j
            Apb = vl * (np.ones_like(b) if self.zocco_flr else D01) / geo.B + 0j
            Abp = D01 / (Ti * geo.B) + 0j
            Abb = vl * D2 / geo.B**2 + 0j
            return App, Apb, Abp, Abb
        m = self.ion_mask
        w = self.ti_w[m]
        lam = self.ti_lam[m]
        lB = self.ti_lamB[None, :]
        bb = b[m][:, None]
        oa = omega - self.ai
        cache = {}

        def Mk(k, c):
            """(1 + c)^-(k + 3/2) M_k(omega, w/(1 + c)) [streaming a/sqrt(1 + c)]: the energy integral with an extra
            exp(-c E) (FLR factor exp(-coef b B lam E))."""
            key = (k, id(c) if not np.isscalar(c) else c)
            if key in cache:
                return cache[key]
            f = 1 + c
            wp = np.broadcast_to(w / f, w.shape)
            if self.ions == "zocco":
                v = Z.Mint(k, omega, wp.ravel()).reshape(w.shape)
            else:
                vT = np.sqrt(2 * Ti)
                a = np.broadcast_to(
                    (self.kpar * vT * np.sqrt(1 - self.ti_lamB))[None, :] / np.sqrt(f),
                    w.shape,
                )
                v = Z.Mint_kpar(k, omega, wp.ravel(), a.ravel()).reshape(w.shape)
            cache[key] = v * f ** (-(k + 1.5))
            return cache[key]

        def I(terms):  # noqa: E743
            """(2/sqrt(pi)) int_0^1 dt sum (coef E^k [FLR]) int dE E^(1/2) e^-E (omega - ai - bi E)/(omega - w E)
            terms: list of (coef(theta,t), k, c(theta,t)) with FLR factor exp(-c E) (c = 0: none).
            """
            acc = 0
            for coef, k, c in terms:
                acc = acc + coef * (oa * Mk(k, c) - self.bi * Mk(k + 1, c))
            return (2 / SQPI) * np.sum(acc, axis=1) / self.nt_i

        if self.flr == "lin":
            z0 = 0.0
            cJ0I1 = 0.0 if self.zocco_flr else -0.75
            App_m = I([(1.0, 0, z0), (-bb * lB, 1, z0)]) / Ti
            Apb_m = vl * I([(lam, 1, z0), (cJ0I1 * lam * bb * lB, 2, z0)])
            Abp_m = I([(lam, 1, z0), (-0.75 * lam * bb * lB, 2, z0)]) / Ti
            Abb_m = vl * I([(lam**2, 2, z0), (-0.5 * lam**2 * bb * lB, 3, z0)])
        else:
            c1, c34, c12 = bb * lB, 0.75 * bb * lB, 0.5 * bb * lB
            App_m = I([(1.0, 0, c1)]) / Ti  # J0^2 ~ exp(-b B lam E)
            Apb_m = vl * I([(lam, 1, c34)])  # mu J0 I1 ~ mu exp(-(3/4) b B lam E)
            Abp_m = I([(lam, 1, c34)]) / Ti
            Abb_m = vl * I([(lam**2, 2, c12)])  # mu^2 I1^2 ~ mu^2 exp(-(1/2) b B lam E)
        out = []
        for X in (App_m, Apb_m, Abp_m, Abb_m):
            full = np.zeros(nth, complex)
            full[m] = X
            out.append(full)
        return out

    def flr_factors(self):
        """F0 moments (times B powers) of the Bessel factors: Gamma0 ~ <J0^2>, Delta01 = B <mu J0 I1>,
        D2 = B^2 <mu^2 I1^2> (= 2 Delta01 exactly); 'lin' small-b, 'exp' exponential (Pade-like) forms, 'bessel' exact
        (J0(x), I1 = 2 J1(x)/x, x^2 = 2 b s, <f> = int_0^inf f(s) e^-s ds by Gauss-Laguerre).
        """
        b = self.b
        if self.flr == "lin":
            return 1 - b, 1 - 1.5 * b, 2 * (1 - 1.5 * b)
        if self.flr == "bessel":
            if not hasattr(self, "_flrB"):
                from scipy.special import j0, j1, roots_laguerre

                sq, ws = roots_laguerre(64)
                x = np.sqrt(2 * b[:, None] * sq[None, :])
                xs = np.where(x > 1e-8, x, 1.0)
                J0 = j0(x)
                I1 = np.where(x > 1e-8, 2 * j1(xs) / xs, 1.0)
                self._flrB = (
                    np.sum(ws * J0**2, axis=1),
                    np.sum(ws * sq * J0 * I1, axis=1),
                    np.sum(ws * sq**2 * I1**2, axis=1),
                )
            return self._flrB
        return 1 / (1 + b), 1 / (1 + 0.75 * b) ** 2, 2 / (1 + 0.5 * b) ** 3

    def electron_kernels(self, omega, Omega=None):
        """Trapped-electron energy kernels per lambda: for a source C = c0 + E c1,
        K^(j) = int dE E^(1/2+j) e^-E (omega - ae - be E)/(omega - E ky Omega) C = sum over moments; returns M_k(lam).
        """
        w = self.ky * (self.Omega if Omega is None else Omega)
        return [Z.Mint(n, omega, w) for n in range(4)]

    @staticmethod
    def _K(M, j, c0, c1, oa, be):
        return oa * c0 * M[j] + (oa * c1 - be * c0) * M[j + 1] - be * c1 * M[j + 2]

    # ------------------------------------------------------------------ the Galerkin matrix
    def matrix(self, omega):
        """Return the (row-scaled) Galerkin matrix D(omega) and the block layout."""
        omega = complex(omega)
        geo, p, sw = self.geo, self.p, self.sw
        Ti = p["Ti"]
        ky = self.ky
        wgt = self.wgt
        Bphi, Bb, Bpsi = self.Bphi, self.Bb, self.Bpsi
        fields = (
            ["phi"]
            + (["bpar"] if self.bpar == "field" else [])
            + (["psi"] if self.use_psi else [])
        )
        basis = dict(phi=Bphi, bpar=Bb, psi=Bpsi)
        # closure: dB_par = r bmhd ky phi / omega  -> dB_par is phi times C(theta)
        Ccl = (
            self.r * self.bmhd * ky / omega
            if self.bpar == "closure"
            else 0.0 * self.bmhd
        )
        # ---------- local responses (theta-diagonal), per source field -> per equation
        App, Apb, Abp, Abb = self.ion_coeffs(omega)
        G0i, d01i, d2i = self.flr_factors()
        exQN = (sw["fdq_i"] - sw["vl_i"]) * d01i / geo.B - (
            sw["fdq_e"] - sw["vl_e"]
        ) / geo.B  # sum q n (f - v) Delta01/B
        exPB = (
            sw["src_i"] * (sw["fdp_i"] - sw["vl_i"]) * Ti * d2i / geo.B**2
            + sw["src_e"] * (sw["fdp_e"] - sw["vl_e"]) * 2 / geo.B**2
        )
        pol = G0i / Ti + 1 if self.old_pol else 1 / Ti + 1
        oe = omega - self.ae
        # local density-like operators: each is dict source -> theta array (the 'phi' source in closure mode
        # already includes its dB_par)
        # QN residual (ion - electron - pol):  H_i - H_e - (1/T_i + 1) phi + exQN dB_par
        loc = {eq: {} for eq in ("qn", "pb", "tch")}
        loc["qn"]["phi"] = App - pol + (Apb + exQN) * Ccl
        loc["qn"]["bpar"] = Apb + exQN
        loc["qn"]["psi"] = (
            1 - ky * p["omn"] / omega
        ) + 0 * App  # - H_e(passing psi part) = +(1 - w*e/omega) psi
        # PB residual: 2 dB_par/beta + s_i T_i P_i + s_e P_e + exPB dB_par
        loc["pb"]["phi"] = sw["src_i"] * Ti * Abp + 0j
        loc["pb"]["bpar"] = 2 / p["beta"] + sw["src_i"] * Ti * Abb + exPB
        loc["pb"]["psi"] = (
            -sw["src_e"] * (1 - (self.ae + 2.5 * self.be) / omega) / geo.B + 0j
        )
        # TCH residual: bend - omega^2 H_i - omega (omega - w*e) phi + v_e omega (omega - ae - 2.5 be) dB/B
        #               + omega Dm_e + omega^2 pol phi - omega^2 exQN dB_par
        om2 = omega**2
        Bterm = sw["vl_e"] * omega * (oe - 2.5 * self.be) / geo.B - om2 * exQN
        loc["tch"]["phi"] = (
            -om2 * App
            - omega * (omega - ky * p["omn"])
            + om2 * (1 / Ti + 1)
            + (-om2 * Apb + Bterm) * Ccl
        )
        loc["tch"]["bpar"] = -om2 * Apb + Bterm
        # passing+trapped psi part of omega Dm_e: Dm_e(psi, all electrons) = -T1/omega
        T1c = -ky * (self.Kg_e + geo.Kc) / geo.B * (oe - 2.5 * self.be)
        loc["tch"]["psi"] = -T1c + 0j
        # ---------- trapped electrons (nonlocal): sources c0, c1 per basis function
        be = self.be
        Hcache = {}
        Mcache = {}

        def Kmom(j, f, s=0.0):
            """K_j(n, lam) = int dE E^(1/2 + j) e^-E H_n(lam, E): analytic (Z function) without collisions,
            energy quadrature with them.  s: precession shift Omega -> Omega + s Omega_bp (precession vs direct drive test).
            """
            c0, c1 = src(f)
            Om = self.Omega + s * self.Omega_bp if s != 0 else None
            if self.coll <= 0:
                if s not in Mcache:
                    Mcache[s] = self.electron_kernels(omega, Om)
                return self._K(Mcache[s], j, c0, c1, oe, be)
            if (f, s) not in Hcache:
                Hcache[(f, s)] = self._eH(omega, c0, c1, Om)
            return np.einsum("e,enl->nl", self.eW * self.eE**j, Hcache[(f, s)])

        # precession vs direct drive test, electrons: phi response = K(a r) + d [K(0) - K(r)]  (list of (shift, coefficient))
        if self.dec["e"]:
            ae_, de_, rr = sw["a_e"], sw["d_e"], self.r_eff
            eparts = [(ae_ * rr, 1.0)] + ([(0.0, de_), (rr, -de_)] if de_ != 0 else [])
        else:
            eparts = [(0.0, 1.0)]

        def Kphi(j, f, G=None):
            """sum over the electron parts of [G(s)] K_j(s); G: None (no theta weight) or the TCH drift map"""
            parts = eparts if f == "phi" else [(0.0, 1.0)]
            acc = 0
            for s_, c_ in parts:
                K = Kmom(j, f, s_)
                if G is None:
                    acc = acc + c_ * K
                else:
                    acc = acc + c_ * ((G + s_ * self.Gbp) if s_ != 0 else G) @ K.T
            return acc

        def src(field):
            if field == "phi":
                c0 = self.bar_phi
                c1 = (
                    sw["vl_e"]
                    * self.r
                    * self.lam[None, :]
                    * (ky / omega)
                    * self.bar_Pphi
                    if self.bpar == "closure"
                    else 0 * c0
                )
            elif field == "bpar":
                c0 = 0 * self.bar_b
                c1 = -sw["vl_e"] * self.lam[None, :] * self.bar_b
            else:
                c0 = -self.bar_psi
                c1 = -(ky / omega) * self.bar_Dpsi
            return c0, c1

        # projections of the theta<-lambda maps on the test functions
        test = dict(qn=Bphi, pb=Bb, tch=Bpsi)
        eqs = (
            ["qn"]
            + (["pb"] if "bpar" in fields else [])
            + (["tch"] if "psi" in fields else [])
        )
        blocks = {}
        SM = self.stream.moments(omega) if self.ions == "stream" else None
        tfield = dict(qn="phi", pb="bpar", tch="psi")
        for eq in eqs:
            T = test[eq]
            for f in fields:
                S = basis[f]
                # local part
                A = (np.conj(T) * (wgt * loc[eq][f])[None, :]) @ S.T
                # trapped-electron part
                if eq == "qn":  # - H_e,tr = + (1/sqrt pi) G0 K0
                    K = Kphi(0, f)
                    A = A + (np.conj(T) * wgt[None, :]) @ self.G0 @ K.T / SQPI
                elif eq == "pb":  # + s_e P_e,tr = - s_e (1/sqrt pi) Gl K1
                    K = Kphi(1, f)
                    A = (
                        A
                        - sw["src_e"]
                        * (np.conj(T) * wgt[None, :])
                        @ self.Gl
                        @ K.T
                        / SQPI
                    )
                else:  # + omega Dm_e,tr = - omega (1/sqrt pi) Gd K1 (each part with its own drift)
                    A = (
                        A
                        - omega
                        * (np.conj(T) * wgt[None, :])
                        @ Kphi(1, f, G=self.Gd)
                        / SQPI
                    )
                if eq == "tch" and f == "psi":
                    A = A + self.bend_matrix()
                if SM is not None and (
                    f in ("phi", "bpar") or (f == "psi" and ("H", "phi", "psi") in SM)
                ):
                    tf = tfield[eq]
                    if self.bpar == "closure" and f == "phi":
                        cfac = self.r * ky / omega
                        sm = {
                            k: SM[(k, tf, "phi")] + cfac * SM[(k, tf, "bpar")]
                            for k in ("H", "L", "D")
                        }
                        if eq == "qn":
                            A = A + sm["H"]
                        else:
                            A = A - omega * (sm["L"] + sm["D"])
                    elif eq == "qn":
                        A = A + SM[("H", tf, f)]
                    elif eq == "pb":
                        A = A + sw["src_i"] * Ti * SM[("P", tf, f)]
                    else:
                        A = A - omega * (SM[("L", tf, f)] + SM[("D", tf, f)])
                blocks[(eq, f)] = A
        # ---------- assemble with row scaling
        sizes = [basis[f].shape[0] for f in fields]
        off = np.concatenate([[0], np.cumsum(sizes)])
        D = np.zeros((off[-1], off[-1]), complex)
        for a, eq in enumerate(eqs):
            for c, f in enumerate(fields):
                D[off[a] : off[a + 1], off[c] : off[c + 1]] = blocks[(eq, f)]
        self._blocks = blocks
        return D, fields

    def bpar_response(self, omega, a=1.0, d=1.0):
        """dB_par from perpendicular pressure balance alone, for a GIVEN phi = a Bphi[0] (and psi = d Bpsi[0]) and a
        given omega (e.g. GENE's): solves the PB rows for the dB_par coefficients.  Returns dB_par(theta) and
        r(theta) = dB_par/dB_par^MHD."""
        assert self.bpar == "field"
        D, fields = self.matrix(omega)
        bl = self._blocks
        rhs = -bl[("pb", "phi")] @ (a * np.ones(self.Bphi.shape[0]))
        if self.use_psi:
            rhs = rhs - bl[("pb", "psi")] @ (d * np.ones(self.Bpsi.shape[0]))
        c = np.linalg.solve(bl[("pb", "bpar")], rhs)
        bp = c @ self.Bb
        phi = a * self.Bphi[0]
        mhd = self.bmhd * self.ky / omega * phi
        return bp, bp / np.where(np.abs(mhd) > 0, mhd, 1)

    def bend_matrix(self):
        if not hasattr(self, "_bend"):
            geo = self.geo
            JB = geo.J * geo.B
            dpsi = np.array([np.gradient(f, geo.theta) / JB for f in self.Bpsi])
            kp2 = self.ky**2 * geo.gyy
            self._bend = (
                -(2 / self.p["beta"])
                * (np.conj(dpsi) * (JB * geo.dth * kp2 / geo.B)[None, :])
                @ dpsi.T
            )
        return self._bend

    def scales(self):
        """Row scaling (omega independent) so that all blocks are O(1)."""
        if not hasattr(self, "_scales"):
            nrm = np.real(np.sum(self.wgt * np.abs(self.Bphi) ** 2, axis=1))
            s = [1 / nrm]
            if self.bpar == "field":
                s.append(
                    np.real(np.sum(self.wgt * np.abs(self.Bb) ** 2, axis=1)) ** -1
                    * self.p["beta"]
                    / 2
                )
            if self.use_psi:
                s.append(1 / np.abs(np.diag(self.bend_matrix())))
            self._scales = np.concatenate(s)
        return self._scales

    def fval(self, omega):
        """Smallest-modulus eigenvalue of the scaled D(omega) (tracked by eigenvector overlap) and eigenvector."""
        D, fields = self.matrix(omega)
        D = self.scales()[:, None] * D
        if D.shape[0] == 1:
            return D[0, 0], np.ones(1, complex)
        ev, V = np.linalg.eig(D)
        if self.last_vec is not None:
            ov = np.abs(V.conj().T @ self.last_vec) / (
                np.linalg.norm(V, axis=0) * np.linalg.norm(self.last_vec)
            )
            score = np.abs(ev) / (np.abs(ev).max() + 1e-300) - 0.5 * ov
            i = (
                np.argmin(score)
                if np.abs(ev).min() > 1e-3 * np.abs(ev).max()
                else np.argmin(np.abs(ev))
            )
        else:
            i = np.argmin(np.abs(ev))
        return ev[i], V[:, i]

    def det(self, omega):
        D, _ = self.matrix(omega)
        D = self.scales()[:, None] * D
        return np.linalg.det(D)

    # ------------------------------------------------------------------ root finding
    def solve(self, omega0, tol=1e-9, maxit=60, method="det", verbose=False):
        """Secant iteration on f(omega) = smallest eigenvalue of D(omega) (method 'eig') or det D ('det')."""
        t0 = time.time()
        f = (lambda w: self.fval(w)[0]) if method == "eig" else self.det
        w0 = complex(omega0)
        w1 = w0 * (1 + 1e-3) + 1e-4j
        f0, f1 = f(w0), f(w1)
        ok = False
        for it in range(maxit):
            if f1 == f0:
                break
            w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
            if not np.isfinite(w2):
                break
            if abs(w2 - w1) > 0.2:
                w2 = w1 + 0.2 * (w2 - w1) / abs(w2 - w1)  # damp big jumps
            w0, f0, w1 = w1, f1, w2
            if method == "eig":
                f1, v = self.fval(w1)
                self.last_vec = v
            else:
                f1 = f(w1)
            if verbose:
                print("   it %d omega %s |f| %.2e" % (it, w1, abs(f1)))
            if abs(w1 - w0) < tol:
                ok = True
                break
        res = self.result(w1)
        res.update(
            converged=bool(ok and abs(f1) < 1e-5),
            iterations=it + 1,
            seconds=time.time() - t0,
            fres=abs(f1),
        )
        return res

    def nullvec(self, omega):
        D, fields = self.matrix(omega)
        D = self.scales()[:, None] * D
        u, s, vh = np.linalg.svd(D)
        return vh[-1].conj(), fields, s

    def result(self, omega):
        """Eigenfunctions at omega (null vector of D), r(theta) = dB_par/dB_par^MHD, widths, diagnostics."""
        omega = complex(omega)
        v, fields, s = self.nullvec(omega)
        sizes = [
            dict(phi=self.Bphi, bpar=self.Bb, psi=self.Bpsi)[f].shape[0] for f in fields
        ]
        off = np.concatenate([[0], np.cumsum(sizes)])
        coef = {f: v[off[i] : off[i + 1]] for i, f in enumerate(fields)}
        geo = self.geo
        phi = coef["phi"] @ self.Bphi
        i0 = np.argmin(np.abs(geo.theta))
        nrm = (
            phi[i0]
            if abs(phi[i0]) > 1e-12 * np.abs(phi).max()
            else phi[np.argmax(np.abs(phi))]
        )
        phi = phi / nrm
        if "bpar" in coef:
            bpar = coef["bpar"] @ self.Bb / nrm
        elif self.bpar == "closure":
            bpar = self.r * self.bmhd * self.ky / omega * phi
        else:
            bpar = 0 * phi
        psi = coef["psi"] @ self.Bpsi / nrm if "psi" in coef else 0 * phi
        mhd = self.bmhd * self.ky / omega * phi
        r_th = np.where(
            np.abs(mhd) > 1e-6 * np.abs(mhd).max(),
            bpar / np.where(mhd == 0, 1, mhd),
            np.nan,
        )
        w = np.abs(phi) ** 2
        frac05 = float(np.sum(w[np.abs(geo.theta) < 0.5]) / np.sum(w))
        frac_pi2 = float(np.sum(w[np.abs(geo.theta) < np.pi / 2]) / np.sum(w))
        sel = np.abs(geo.theta) < 0.5
        rw = np.sum((r_th * w)[sel]) / np.sum(w[sel]) if self.bpar != "off" else 0
        return dict(
            omega=omega,
            omega_gene=-omega.real,
            gamma=omega.imag,
            theta=geo.theta,
            phi=phi,
            bpar=bpar,
            psi=psi,
            r0=complex(r_th[i0]) if self.bpar != "off" else 0j,
            r_avg=complex(rw),
            frac05=frac05,
            frac_pi2=frac_pi2,
            sv_ratio=float(s[-1] / s[0]),
            fields=fields,
        )

    def scan(self, wr=(-0.6, 0.3, 46), wi=(-0.1, 0.4, 26)):
        xr = np.linspace(*wr)
        xi = np.linspace(*wi)
        A = np.empty((xi.size, xr.size))
        for j, y in enumerate(xi):
            for i, x in enumerate(xr):
                D, _ = self.matrix(x + 1j * y)
                s = np.linalg.svd(self.scales()[:, None] * D, compute_uv=False)
                A[j, i] = s[-1] / s[0]
        return xr, xi, A


# ---------------------------------------------------------------------- continuation helpers
def make(ky, **kw):
    return Solver(ky, **kw)


def continuation(
    build,
    values,
    omega0,
    verbose=True,
    label="",
    maxjump=0.06,
    minstep=1e-3,
    gamma_floor=None,
):
    """Follow a root along a parameter: build(value) -> Solver.  Seeds by linear extrapolation of the last two
    roots; if the new root jumps by more than maxjump (or does not converge) the parameter step is halved
    (adaptive), so the branch is not lost to a neighbouring root (e.g. the shear-Alfven roots at |omega| ~ 1-3).
    gamma_floor: stop when a converged root has gamma < gamma_floor (models without Landau continuation, e.g.
    ions='stream' or 'kpar'); the last result then carries 'crossing' = parameter value where gamma = gamma_floor
    (linear interpolation) and 'stopped' = True.
    Returns the list of results at the requested values (intermediate steps are not returned).
    """
    out = []
    hist = []  # (value, omega) of accepted points
    vcur = None
    for v in values:
        if vcur is None:
            S = build(v)
            r = S.solve(omega0)
            r["param"] = v
            out.append(r)
            vcur = v
            if r["converged"]:
                hist.append((v, r["omega"]))
            if verbose:
                _pr(label, v, r)
            continue
        target = v
        step = target - vcur
        while True:
            vtry = vcur + step
            if len(hist) >= 2 and hist[-1][0] != hist[-2][0]:
                (v1, w1), (v2, w2) = hist[-2], hist[-1]
                seed = w2 + (w2 - w1) * (vtry - v2) / (v2 - v1)
            else:
                seed = hist[-1][1] if hist else omega0
            S = build(vtry)
            r = S.solve(seed)
            ref = hist[-1][1] if hist else omega0
            if (
                gamma_floor is not None
                and r["converged"]
                and r["omega"].imag < gamma_floor
                and abs(r["omega"] - seed) < maxjump
            ):
                vp, wp = hist[-1]
                f = (wp.imag - gamma_floor) / (wp.imag - r["omega"].imag)
                r["crossing"] = float(vp + f * (vtry - vp))
                r["stopped"] = True
                r["param"] = vtry
                out.append(r)
                if verbose:
                    print(
                        "%s stopped at %g: gamma %.4f < floor; crossing at %.4g"
                        % (label, vtry, r["omega"].imag, r["crossing"]),
                        flush=True,
                    )
                return out
            good = (
                r["converged"]
                and abs(r["omega"] - seed) < maxjump
                and abs(r["omega"] - ref) < 3 * maxjump
            )
            if good or abs(step) <= minstep * max(1, abs(target)):
                if good:
                    hist.append((vtry, r["omega"]))
                vcur = vtry
                if abs(vcur - target) < 1e-12:
                    r["param"] = v
                    out.append(r)
                    if verbose:
                        _pr(label, v, r)
                    break
                step = target - vcur
            else:
                step = step / 2
    return out


def _pr(label, v, r):
    print(
        "%s %8.4g  omega_GENE %+.4f gamma %+.4f  r0 %s  frac05 %.3f  conv %s  %.1fs"
        % (
            label,
            v,
            r["omega_gene"],
            r["gamma"],
            np.round(r["r0"], 3),
            r["frac05"],
            r["converged"],
            r["seconds"],
        ),
        flush=True,
    )


DEFAULT_V1 = dict(
    ions="stream", bpar="field", psi=True, basis=("hermite", 8, 0.35)
)  # collisionless, exponential FLR (first main model)
# main model: + exact Bessel ion FLR + electron pitch-angle scattering (bounce-averaged Lorentz
# operator, nu_ei from GENE coll) + trapped-ion Krook detrapping, at the STEP deck's coll
DEFAULT = dict(DEFAULT_V1, flr="bessel", coll=COLL_NOMINAL, coll_model="lorentz")
# GENE STEP hKBM roots (omega, gamma) per k_y rho_s, used as seeds
GENE_SEEDS = {
    0.05: (0.0331, 0.0052),
    0.1: (0.0676, 0.0243),
    0.2: (0.0977, 0.0827),
    0.3: (0.1933, 0.0947),
    0.4: (0.258, 0.100),
}


def solve(ky, omega0=None, **kw):
    """One root of the hKBM dispersion relation.

    ky: k_y rho_s.  omega0: seed in GENE sign convention (omega_GENE + i gamma; default: GENE's fB1 value at the
    nearest tabulated k_y).  kw: any Solver option, default DEFAULT (streaming ions, dB_par from pressure balance,
    two-field QN + PB + TCH, Hermite N = 8, sigma 0.35); switches vl_e, vl_i, fd_e, fd_i, src_e, src_i, bp_e, bp_i
    (continuous multipliers in [0, 1], GENE bpar_vlasov / bpar_field / bpar_source / dpdx_term).
    Returns dict: omega_gene, gamma (GENE sign), omega (model sign, exp(-i omega t)), theta, phi, bpar, psi
    (normalised phi(0) = 1), r0 = dB_par/dB_par^MHD at theta = 0, r_avg (|phi|^2-weighted over |theta| < 0.5),
    frac05 (|phi|^2 fraction within |theta| < 0.5 rad), converged, seconds."""
    opts = dict(DEFAULT)
    opts.update(kw)
    if opts.get("ions") in ("zocco", "kpar") and "flr" not in kw:
        opts["flr"] = "exp"
    S = Solver(ky, **opts)
    if omega0 is None:
        k = min(GENE_SEEDS, key=lambda x: abs(np.log(x / ky)))
        g = GENE_SEEDS[k]
        omega0 = complex(g[0] * ky / k, g[1])
    w0 = complex(-complex(omega0).real, complex(omega0).imag)
    return S.solve(w0)


def selftest():
    """Regression and internal tests (about 1 minute)."""
    ok = True
    from . import ion_stream as IS

    om = -0.1 + 0.08j
    A = Solver(0.2, ions="zocco", bpar="field", psi=True, basis=("hermite", 4, 0.25))
    B = Solver(
        0.2,
        ions="stream",
        bpar="field",
        psi=True,
        basis=("hermite", 4, 0.25),
        stream_kw=dict(vscale=1e-8),
    )
    DA, _ = A.matrix(om)
    DB, _ = B.matrix(om)
    e = np.abs(DA - DB).max() / np.abs(DA).max()
    ok &= e < 1e-4
    print("streaming ions, v_Ti -> 0 vs local Zocco ions: rel diff %.1e" % e)
    rng = np.random.default_rng(1)
    n = 40
    dt = rng.uniform(0.1, 3, (2, 3, n))
    Sv = 1.3 - 0.4j
    gt = IS.StreamingIons._integrate(
        np.full((1, 2, 3, n), -1j * Sv), 1j * om * dt, dt, True
    )
    e = np.abs(gt - 2 * Sv / om).max()
    ok &= e < 1e-8
    print("trapped-ion periodic orbit, zero drift: |g - 2S/omega| = %.1e" % e)
    # collisions: the energy-quadrature path at vanishing coll reproduces the analytic (Z-function) kernels
    A = Solver(0.2, ions="zocco", bpar="field", psi=True, basis=("hermite", 4, 0.35))
    for model in ("lorentz", "krook"):
        B = Solver(
            0.2,
            ions="zocco",
            bpar="field",
            psi=True,
            basis=("hermite", 4, 0.35),
            coll=1e-14,
            coll_model=model,
        )
        e = (
            np.abs(A.matrix(om)[0] - B.matrix(om)[0]).max()
            / np.abs(A.matrix(om)[0]).max()
        )
        ok &= e < 1e-6
        print(
            "collisions (%s), coll -> 0 vs analytic kernels: rel diff %.1e" % (model, e)
        )
    C = Solver(
        0.2,
        ions="zocco",
        bpar="field",
        psi=True,
        basis=("hermite", 2, 0.35),
        coll=COLL_NOMINAL,
    )
    ok &= abs(C.nu_ei - 0.032724) < 1e-6
    print(
        "nu_ei(coll = 1.35e-4) = %.6f (GENE 0.032724); Lorentz slowest detrapping kappa0 = %.3f nu_D (1/trpeps = %.3f)"
        % (C.nu_ei, C.kappa0, 1 / C.eps_coll)
    )
    from scipy.special import i0e, i1e

    Bs = Solver(
        0.6,
        ions="adiabatic",
        bpar="field",
        psi=True,
        basis=("hermite", 2, 0.35),
        flr="bessel",
    )
    G0, D01, _ = Bs.flr_factors()
    e = max(np.abs(G0 - i0e(Bs.b)).max(), np.abs(D01 - i0e(Bs.b) + i1e(Bs.b)).max())
    ok &= e < 1e-10
    print("exact FLR moments vs Gamma0, Gamma0 - Gamma1: %.1e" % e)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok
