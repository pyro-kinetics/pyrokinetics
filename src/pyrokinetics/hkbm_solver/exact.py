"""
Exact linear electromagnetic gyrokinetic eigenvalue solver in ballooning space (no asymptotic
ordering): every species kinetic (electrons with their real mass), fields phi, A_par and delta B_par
from quasineutrality, parallel Ampere's law and perpendicular pressure balance, exact Bessel
gyroaverages, GENE's full_drift magnetic drift (with the K_x / secular part on the extended line),
both parities, theta0 free.  Collisionless (Stage 1).

Method (frequency domain).  For a trial omega (Im omega > 0) the gyrokinetic equation of each
species, written for G = T h/(q F0) (h the non-adiabatic part, exp(-i omega t)),

    i v_par d_l G + (omega - omega_d) G = (omega - omega_*^T) <chi>,
    <chi> = J0 (phi - v_par A_par) + (T/Z) mu B I1 b,    I1 = 2 J1(a)/a,  b = dB_par/B,

is integrated exactly along every orbit for every velocity node: passing particles along the whole
extended field line (G = 0 entering the domain, the outgoing/decaying condition for Im omega > 0),
trapped particles over a bounce period in each well (periodic).  The fields are piecewise constant on
cells (npt per 2 pi); per cell (or the part of a cell inside a well) the orbit integral is done with the
exponential integrator, exact for a constant source and with the exact time of flight, drift phase,
time-averaged J0, v_par J0 and mu B I1 of the segment (quadrature with the bounce/peak singularities
mapped out).  The response is semi-separable in (theta, theta'), so the dense field matrix M(omega)
is assembled with blocked matrix products (O(N^2 n_v), stable for Im omega > 0).  Field equations
(Galerkin, cell indicator test functions, volume element J dtheta):

    QN:    sum_s (Z_s^2 n_s/T_s) [phi V_k - <<J0 G_s>>_k] = 0
    Amp:   k_perp^2 A_par - (beta/2) sum_s (Z_s^2 n_s/T_s) <<v_par J0 G_s>> = 0
    Perp:  B^2 b + (beta/2) sum_s Z_s n_s <<mu B I1 G_s>> = 0

(<<.>>_k = int_cell J dtheta int d^3v F0/n .).  With theta0 = 0 on an up-down symmetric surface the
problem splits into the twisting (phi even) and tearing (phi odd) parities: half-domain unknowns.
Eigenvalues: zeros of the bordered dispersion function D(omega) = 1/[M^-1]_rr (r the normalising
unknown: phi near theta0 for twisting, A_par near theta0 for tearing), secant iteration.

Units (as solver.py/mtm.py): T_ref = T_e, n_ref = n_e, m_ref = m_i, lengths L_ref, k in 1/rho_s,
frequencies in c_s/L_ref, model sign (omega_r > 0 electron direction; GENE's omega = -Re omega).
phi in T_e/e, A_par in T_e/(e c_s) (v_par A_par in T_e/e), b = dB_par/B.  dl = J B dtheta.
    omega_d = (T/Z) k_y [lam B K_y,g + 2 (1 - lam B) K_c] E/B,  K_c = K_y - dpdx/(2B),
    omega_*^T = -(T/Z) k_y [omn + omt (E - 3/2)],  E = v^2/v_t^2,  lam = mu/E,  v_t = sqrt(2T/m),
    a^2 = 2 k_perp^2 T m lam E/(Z^2 B),  k_perp^2 = k_y^2 g^yy(theta; theta0).

    from pyrokinetics.hkbm_solver import exact
    S = exact.ExactSolver.from_deck(Deck("run/parameters"), ky_ref=0.3, nturns=8)
    r = S.find_root(omega0, parity="twisting")
"""

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.special import j0 as besselj0
from scipy.special import j1 as besselj1
from scipy.special import roots_genlaguerre, roots_legendre

PREF = 0.5 / np.sqrt(np.pi)  # pi^-3/2 x (pi/2): d^3v F0/n per sigma in (E, lam), weight sqrt(E) e^-E


def _phi1(x):
    """(e^x - 1)/x."""
    small = np.abs(x) < 1e-2
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 + x / 2 + x**2 / 6 + x**3 / 24, np.expm1(xs) / xs)


def _phi2(x):
    """(e^x - 1 - x)/x^2."""
    small = np.abs(x) < 1e-1
    xs = np.where(small, 1.0, x)
    ser = 0.5 + x / 6 + x**2 / 24 + x**3 / 120 + x**4 / 720 + x**5 / 5040
    return np.where(small, ser, (np.expm1(xs) - xs) / xs**2)


def _I1(a):
    """2 J1(a)/a."""
    aa = np.where(a > 1e-8, a, 1.0)
    return np.where(a > 1e-8, 2 * besselj1(aa) / aa, 1.0 - a**2 / 8)


class Species:
    """One kinetic species in solver units (charge Z, mass m/m_i, T/T_e, n/n_e, gradients)."""

    def __init__(self, name, Z, m, T, n, omn, omt):
        self.name, self.Z, self.m, self.T, self.n = name, float(Z), float(m), float(T), float(n)
        self.omn, self.omt = float(omn), float(omt)
        self.vt = np.sqrt(2 * self.T / self.m)

    def __repr__(self):
        return (f"Species({self.name}, Z={self.Z}, m={self.m:.6g}, T={self.T}, n={self.n}, "
                f"omn={self.omn}, omt={self.omt})")


def species_from_deck(deck):
    """Kinetic species of a gene_io.Deck in solver units (reference T_e, n_e, m of the first ion)."""
    from .gene_io import _species_list

    sp = _species_list(deck.nml)
    e = [s for s in sp if float(s["charge"]) < 0][0]
    ion = [s for s in sp if float(s["charge"]) > 0][0]
    Te, ne, mi = float(e["temp"]), float(e["dens"]), float(ion["mass"])
    out = []
    for s in sp:
        out.append(Species(str(s.get("name", "s")).strip(), float(s["charge"]), float(s["mass"]) / mi,
                           float(s["temp"]) / Te, float(s["dens"]) / ne, float(s["omn"]),
                           float(s["omt"])))
    return out


class ExtendedGeometry:
    """Geometry on the extended ballooning line theta in [-pi - 2 pi nt, pi + 2 pi nt] for ballooning
    angle theta0 (turn j = Geo.shifted(theta0 - 2 pi j)), as cubic splines in theta."""

    def __init__(self, geo, theta0=0.0, nturns=8, bp=1.0):
        self.geo, self.theta0, self.nt, self.bp = geo, float(theta0), int(nturns), float(bp)
        g = geo
        nz = g.z.size
        js = np.arange(-self.nt, self.nt + 1)
        # d(k_x/k_y)/d(theta0) from the secular part of g^xy/g^xx: r(z) = kpt z + periodic, so
        # kpt = (r(pi) - r(-pi))/(2 pi) with r(pi) extrapolated from the end of GENE's z grid (no
        # z = pi node); this makes the turns join continuously (Geo.kx_per_theta0 uses the end nodes)
        r = g.gxy / g.gxx
        rpi = np.polyval(np.polyfit(g.z[-6:], r[-6:], 4), np.pi)
        kpt = (rpi - r[0]) / (2 * np.pi)
        self.kpt = kpt
        # up-down symmetric surface: K_x is odd and periodic, so K_x(-pi) = 0; GENE's Miller leaves a
        # residue there (~1e-4) that the turn shift kappa_j K_x would amplify into K_y jumps between
        # turns that grow with j and break the theta -> -theta symmetry: set it to zero.
        Kx = g.Kx.copy()
        k = np.arange(1, nz)
        sc = lambda a: np.max(np.abs(a)) + 1e-30  # noqa: E731
        self.updown = bool(
            np.max(np.abs(g.B[k] - g.B[nz - k])) < 1e-9 * sc(g.B)
            and np.max(np.abs(g.Ky[k] - g.Ky[nz - k])) < 1e-9 * sc(g.Ky)
            and np.max(np.abs(Kx[k] + Kx[nz - k])) < 1e-9 * sc(Kx)
        )
        if self.updown:
            Kx[0] = 0.0
        th, B, J, Ky, gyy = [], [], [], [], []
        for j in js:
            kap = -kpt * (self.theta0 - 2 * np.pi * j)
            th.append(g.z + 2 * np.pi * j)
            B.append(g.B)
            J.append(g.J)
            Ky.append(g.Ky + kap * Kx)
            gyy.append(g.gyy + 2 * kap * g.gxy + kap**2 * g.gxx)
        # closing point theta = pi + 2 pi nt: first point of the next turn
        kap = -kpt * (self.theta0 - 2 * np.pi * (self.nt + 1))
        th.append([np.pi + 2 * np.pi * self.nt])
        B.append(g.B[:1])
        J.append(g.J[:1])
        Ky.append(g.Ky[:1] + kap * Kx[:1])
        gyy.append(g.gyy[:1] + 2 * kap * g.gxy[:1] + kap**2 * g.gxx[:1])
        th = np.concatenate(th)
        self.sB = g.sB  # periodic
        self.sJ = g.sJ
        self.sKy = CubicSpline(th, np.concatenate(Ky))
        self.sgyy = CubicSpline(th, np.concatenate(gyy))
        self.dpdx = g.dpdx
        self.Bmax, self.Bmin = g.Bmax, g.Bmin
        zz = np.linspace(-np.pi, np.pi, 40001)
        Bz = g.sB(zz)
        self.zmax = float(zz[np.argmax(Bz)])
        self.d2Bmax = float(g.sB(self.zmax, 2))
        self.lo, self.hi = -np.pi - 2 * np.pi * self.nt, np.pi + 2 * np.pi * self.nt

    def at(self, th):
        """dict of B, J, Ky, Kc, Kg (lam B coefficient, K_c + bp dpdx/2B), gyy at theta (array)."""
        B = self.sB(th)
        Ky = self.sKy(th)
        Kc = Ky - self.dpdx / (2 * B)
        return dict(B=B, J=self.sJ(th), Ky=Ky, Kc=Kc, Kg=Kc + self.bp * self.dpdx / (2 * B),
                    gyy=self.sgyy(th))


class ExactSolver:
    """Exact collisionless linear EM gyrokinetics at one (k_y, theta0).

    geo        geometry.Geo;  species: list of Species;  ky: k_y rho_s;  beta: beta_e at B_ref
    nturns     ballooning turns each side of the central one (domain |theta| <= pi (2 nturns + 1))
    npt        field cells per 2 pi (even)
    nE, nlp, nlt  energy (Gauss-Laguerre) and passing pitch nodes; trapped pitch nodes per interval
               between the bounce-point-on-cell-edge pitches 1/B(edge)
    nq         quadrature nodes per segment (geometry averages)
    fields     subset of ("phi", "apar", "bpar")
    bp         1: dpdx in the grad-B drift (full_drift); 0: gradB_eq_curv
    drift_sign multiplies omega_d (diagnostic)
    """

    def __init__(self, geo, species, ky, beta, theta0=0.0, nturns=8, npt=32, nE=16, nlp=16,
                 nlt=2, nq=6, fields=("phi", "apar", "bpar"), bp=1.0, drift_sign=1.0,
                 trapped=True, block=None):
        if npt % 2:
            raise ValueError("npt must be even")
        self.geo, self.species, self.ky, self.beta = geo, list(species), float(ky), float(beta)
        self.theta0, self.nturns, self.npt = float(theta0), int(nturns), int(npt)
        self.fields = tuple(f for f in ("phi", "apar", "bpar") if f in fields)
        self.drift_sign, self.trapped = float(drift_sign), bool(trapped)
        self.eg = ExtendedGeometry(geo, theta0, nturns, bp)
        self.N = (2 * self.nturns + 1) * self.npt
        self.dth = 2 * np.pi / self.npt
        self.edges = self.eg.lo + self.dth * np.arange(self.N + 1)
        self.theta = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.block = block if block is not None else self.npt
        t0 = time.time()
        self._cells(nq)
        self._velocity(nE, nlp, nlt)
        self._passing(nq)
        if self.trapped:
            self._trapped(nq)
        self.t_setup = time.time() - t0
        self.symmetric = self._is_symmetric()

    # ------------------------------------------------------------------ setup
    def _cells(self, nq):
        x, w = roots_legendre(2 * nq)
        a, b = self.edges[:-1], self.edges[1:]
        th = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * x[None, :]
        ww = 0.5 * (b - a)[:, None] * w[None, :]
        G = self.eg.at(th)
        J = G["J"]
        self.V = (J * ww).sum(1)  # int_cell J dtheta
        self.K2 = (J * self.ky**2 * G["gyy"] * ww).sum(1)  # int J k_perp^2
        self.VB2 = (J * G["B"] ** 2 * ww).sum(1)  # int J B^2
        gc = self.eg.at(self.theta)
        self.kperp2 = self.ky**2 * gc["gyy"]
        self.Bc, self.Jc = gc["B"], gc["J"]

    def _velocity(self, nE, nlp, nlt):
        E, wE = roots_genlaguerre(nE, 0.5)
        self.E, self.wE = E, wE
        t, wt = roots_legendre(nlp)
        t, wt = 0.5 * (t + 1), 0.5 * wt
        Bmax, Bmin = self.eg.Bmax, self.eg.Bmin
        self.lp = (1 - t**2) / Bmax
        self.wlp = wt * 2 * t / Bmax
        self.tp = t
        # trapped pitches: the segment integrals (time of flight per cell) have square-root
        # singularities in lam where a bounce point crosses a cell edge, lam = 1/B(edge).  The lam
        # range is split at these values (and at local maxima of B); in each interval
        # lam = mid + half sin(phi), Gauss-Legendre in phi (nlt nodes), which absorbs the square
        # roots at both ends (the GS2 idea of bounce points on the grid, in integral form).
        ze = -np.pi + self.dth * np.arange(self.npt + 1)
        zz = np.linspace(-np.pi, np.pi, 20001)
        Bz = self.geo.sB(zz)
        imax = np.nonzero((Bz[1:-1] > Bz[:-2]) & (Bz[1:-1] >= Bz[2:]))[0] + 1
        bks = np.concatenate([1 / self.geo.sB(ze), 1 / Bz[imax], [1 / Bmax, 1 / Bmin]])
        bks = bks[(bks >= 1 / Bmax - 1e-14) & (bks <= 1 / Bmin + 1e-14)]
        bks = np.unique(np.round(bks, 13))
        bks = bks[np.concatenate([[True], np.diff(bks) > 1e-10 * bks[-1]])]
        x, wx = roots_legendre(nlt)
        ph = 0.5 * np.pi * x
        lt, wl = [], []
        for la, lb in zip(bks[:-1], bks[1:]):
            mid, half = 0.5 * (la + lb), 0.5 * (lb - la)
            lt.append(mid + half * np.sin(ph))
            wl.append(half * np.cos(ph) * 0.5 * np.pi * wx)
        self.lt = np.concatenate(lt)
        self.wlt = np.concatenate(wl)

    def _seg_moments(self, th, wq, lam):
        """Orbit integrals over segments.  th, wq: (..., nq) nodes and weights of dtheta (with the
        mapping Jacobians, singularities removed); lam broadcastable.  Returns dict of
        tau = int JB/sqrt(1 - lam B) dtheta, ell = int JB dtheta, dK = int JB w_hat/sqrt dtheta
        (w_hat = [lam B K_g + 2 (1 - lam B) K_c]/B) and the node arrays needed for the E-dependent
        gyroaverages (weights JB/sqrt wq, k_perp^2 lam/B, lam B)."""
        G = self.eg.at(th)
        B = G["B"]
        lB = lam * B
        sq = np.sqrt(np.clip(1 - lB, 1e-300, None))
        JB = G["J"] * B
        wt = JB / sq * wq
        what = (lB * G["Kg"] + 2 * (1 - lB) * G["Kc"]) / B
        return dict(
            tau=wt.sum(-1),
            ell=(JB * wq).sum(-1),
            dK=(wt * what).sum(-1),
            wt=wt,
            wl=JB * wq,
            k2lB=self.ky**2 * G["gyy"] * lam / B,  # a^2 = 2 T m E k2lB / Z^2
            lB=lB,
        )

    def _gyro(self, seg, sp):
        """E-dependent segment averages for species sp: a1 = <J0>_t, a2 = int J0 dl/tau (x sigma v_t
        sqrt E gives <v_par J0>_t), a3 = <lam B I1>_t (x E gives <mu B I1>_t).  Shapes (nE, ...)."""
        E = self.E.reshape((-1,) + (1,) * seg["wt"].ndim)
        a = np.sqrt(2 * sp.T * sp.m * E * seg["k2lB"][None] / sp.Z**2)
        J0 = besselj0(a)
        tau = seg["tau"][None]
        tz = np.where(tau > 0, tau, 1.0)
        a1 = (seg["wt"][None] * J0).sum(-1) / tz
        a2 = (seg["wl"][None] * J0).sum(-1) / tz
        a3 = (seg["wt"][None] * seg["lB"][None] * _I1(a)).sum(-1) / tz
        return a1, a2, a3

    def _passing(self, nq):
        """Passing orbits: one segment per cell.  Cells next to the B maximum (where 1 - lam B is
        smallest) use theta = p + eps sinh(u) so the near-singular transit is resolved."""
        eg = self.eg
        x, w = roots_legendre(nq)
        a, b = self.edges[:-1], self.edges[1:]
        nl = self.lp.size
        # peak positions zmax + 2 pi j nearest to each cell
        p = eg.zmax + 2 * np.pi * np.round((0.5 * (a + b) - eg.zmax) / (2 * np.pi))
        near = (p > a - self.dth) & (p < b + self.dth)
        th = np.empty((nl, self.N, nq))
        wq = np.empty((nl, self.N, nq))
        th[:] = 0.5 * (a + b)[None, :, None] + 0.5 * (b - a)[None, :, None] * x
        wq[:] = 0.5 * (b - a)[None, :, None] * w
        c = np.sqrt(self.lp * abs(eg.d2Bmax) / 2)  # (nl,)
        eps = np.maximum(self.tp, 1e-12) / np.maximum(c, 1e-12)
        idx = np.nonzero(near)[0]
        for il in range(nl):
            ua = np.arcsinh((a[idx] - p[idx]) / eps[il])
            ub = np.arcsinh((b[idx] - p[idx]) / eps[il])
            u = 0.5 * (ua + ub)[:, None] + 0.5 * (ub - ua)[:, None] * x
            th[il, idx] = p[idx, None] + eps[il] * np.sinh(u)
            wq[il, idx] = 0.5 * (ub - ua)[:, None] * w * eps[il] * np.cosh(u)
        seg = self._seg_moments(th, wq, self.lp[:, None, None])
        self.pas = dict(tau=seg["tau"], ell=seg["ell"], dK=seg["dK"],
                        gyro=[self._gyro(seg, sp) for sp in self.species])

    def _wells(self, lam):
        """Wells of lam (trapped) in one period starting at the B maximum: list of (z1, z2)."""
        g = self.geo
        z0 = self.eg.zmax
        zs = z0 + np.linspace(0, 2 * np.pi, 4001)
        f = 1 - lam * g.sB(zs)
        s = f > 0
        out = []
        i = 0
        while i < zs.size:
            if s[i]:
                j = i
                while j + 1 < zs.size and s[j + 1]:
                    j += 1
                z1 = brentq(lambda z: 1 - lam * g.sB(z), zs[i - 1], zs[i]) if i > 0 else zs[0]
                z2 = (brentq(lambda z: 1 - lam * g.sB(z), zs[j], zs[j + 1])
                      if j + 1 < zs.size else zs[-1])
                out.append((z1, z2))
                i = j + 1
            else:
                i += 1
        return out

    def _trapped(self, nq):
        """Trapped orbits: for each trapped pitch and well, the cells crossed between the bounce
        points, segment integrals with theta = c + hw sin(phi) (bounce singularities removed), for
        every turn of the domain (wells cut by the domain ends are dropped)."""
        x, w = roots_legendre(nq)
        lo, dth = self.eg.lo, self.dth
        self.trp = []
        for il, lam in enumerate(self.lt):
            for z1, z2 in self._wells(lam):
                c0, hw = 0.5 * (z1 + z2), 0.5 * (z2 - z1)
                # turns j with the well inside the domain
                js = [j for j in range(-self.nturns - 1, self.nturns + 2)
                      if z1 + 2 * np.pi * j >= self.eg.lo and z2 + 2 * np.pi * j <= self.eg.hi]
                if not js:
                    continue
                js = np.array(js)
                # cells crossed (relative to turn 0 placement), segment phi ranges
                k1 = int(np.floor((z1 - lo) / dth))
                k2 = int(np.floor((z2 - lo) / dth))
                cells = np.arange(k1, k2 + 1)
                ea = np.maximum(lo + dth * cells, z1)
                eb = np.minimum(lo + dth * (cells + 1), z2)
                pa = np.arcsin(np.clip((ea - c0) / hw, -1, 1))
                pb = np.arcsin(np.clip((eb - c0) / hw, -1, 1))
                ph = 0.5 * (pa + pb)[:, None] + 0.5 * (pb - pa)[:, None] * x
                wph = 0.5 * (pb - pa)[:, None] * w * hw * np.cos(ph)
                thz = c0 + hw * np.sin(ph)  # (M, nq), turn 0
                th = thz[None] + 2 * np.pi * js[:, None, None]  # (nW, M, nq)
                wq = np.broadcast_to(wph, th.shape)
                # 1 - lam B ~ cos^2 phi near the ends: the JB/sqrt weight times hw cos(phi) is smooth
                seg = self._seg_moments(th, wq, lam)
                cellw = cells[None, :] + self.npt * js[:, None]
                self.trp.append(dict(il=il, lam=lam, cells=cellw, tau=seg["tau"], ell=seg["ell"],
                                     dK=seg["dK"], gyro=[self._gyro(seg, sp) for sp in self.species]))

    def _is_symmetric(self):
        """theta0 = 0 and an up-down symmetric surface (B, J, K_y, g^yy even about theta = 0)."""
        if self.theta0 != 0.0:
            return False
        t = np.linspace(0.1, self.eg.hi - 0.1, 997)
        G1, G2 = self.eg.at(t), self.eg.at(-t)
        for k in ("B", "J", "Ky", "gyy"):
            if np.max(np.abs(G1[k] - G2[k])) > 1e-6 * (np.max(np.abs(G1[k])) + 1e-30):
                return False
        return True

    # ------------------------------------------------------------------ per omega
    def _rowcol(self, sp, a1, a2, a3, tau, wv, sigma):
        """Row (moment) and column (source) factors, each a list over (QN, Amp, Perp) / (phi, apar,
        bpar) of arrays shaped like a1, restricted to the active fields."""
        cs = sp.Z**2 * sp.n / sp.T
        sv = sigma * sp.vt * np.sqrt(self.E).reshape((-1,) + (1,) * (a1.ndim - 1))
        base = PREF * wv * tau
        rows, cols = [], []
        if "phi" in self.fields:
            rows.append(-cs * base * a1)
            cols.append(a1)
        if "apar" in self.fields:
            rows.append(-(self.beta / 2) * cs * base * sv * a2)
            cols.append(-sv * a2)
        if "bpar" in self.fields:
            E = self.E.reshape((-1,) + (1,) * (a1.ndim - 1))
            rows.append((self.beta / 2) * sp.Z * sp.n * base * E * a3)
            cols.append((sp.T / sp.Z) * E * a3)
        return rows, cols

    def _omega_factors(self, sp, omega, tau, dK):
        """x = i (omega dt - psi) per segment, dt, and c = -i (omega - omega_*^T(E)); arrays (nE, ...)."""
        sh = (-1,) + (1,) * tau.ndim
        E = self.E.reshape(sh)
        vs = sp.vt * np.sqrt(E)
        dt = tau[None] / vs
        psi = self.drift_sign * (sp.T / sp.Z) * self.ky * E * dK[None] / vs
        x = 1j * (omega * dt - psi)
        ws = -(sp.T / sp.Z) * self.ky * (sp.omn + sp.omt * (self.E - 1.5))
        c = -1j * (omega - ws)
        return x, dt, c

    def assemble(self, omega, rows_cells=None):
        """Dense field matrix M(omega): shape (nf * nr, nf * N), rows ordered (equation, cell in
        rows_cells), columns (field, cell)."""
        nf, N = len(self.fields), self.N
        rc = np.arange(N) if rows_cells is None else np.asarray(rows_cells)
        nr = rc.size
        rpos = -np.ones(N, int)
        rpos[rc] = np.arange(nr)
        M = np.zeros((nf, nr, nf, N), complex)
        # local terms
        ii = np.arange(nr)
        diag = []
        if "phi" in self.fields:
            diag.append(sum(sp.Z**2 * sp.n / sp.T for sp in self.species) * self.V)
        if "apar" in self.fields:
            diag.append(self.K2)
        if "bpar" in self.fields:
            diag.append(self.VB2)
        for f in range(nf):
            M[f, ii, f, rc] += diag[f][rc]
        for isp, sp in enumerate(self.species):
            self._add_passing(M, sp, isp, omega, rc, rpos)
            if self.trapped:
                self._add_trapped(M, sp, isp, omega, rpos)
        return M.reshape(nf * nr, nf * N)

    def _add_passing(self, M, sp, isp, omega, rc, rpos):
        P = self.pas
        a1, a2, a3 = P["gyro"][isp]  # (nE, nl, N)
        x, dt, c = self._omega_factors(sp, omega, P["tau"], P["dK"])  # x, dt: (nE, nl, N)
        wv = (self.wE[:, None] * self.wlp[None, :])[:, :, None]
        nE, nl, N = x.shape
        c3 = c[:, None, None]
        for sigma in (1, -1):
            sl = slice(None) if sigma > 0 else slice(None, None, -1)
            rows, cols = self._rowcol(sp, a1, a2, a3, P["tau"][None], wv, sigma)
            # path order arrays, velocity flattened: (nv, N)
            xs = x[..., sl].reshape(nE * nl, N)
            dts = dt[..., sl].reshape(nE * nl, N)
            R = [r[..., sl].reshape(nE * nl, N) for r in rows]
            C = [(cc * c3)[..., sl].reshape(nE * nl, N) for cc in cols]
            cells = np.arange(N)[sl]
            self._path_kernel(M, xs, dts, R, C, cells, rpos)

    def _path_kernel(self, M, x, dt, R, C, cells, rpos):
        """Add the open-path (passing) response.  x, dt: (nv, Np) in path order; R, C: lists of
        (nv, Np) row/column factors; cells: (Np,) cell of each path position; rpos: cell -> row."""
        nv, Np = x.shape
        p1 = _phi1(x)
        p2 = _phi2(x)
        L = np.concatenate([np.zeros((nv, 1), complex), np.cumsum(x, axis=1)], axis=1)  # L_m
        nf = len(R)
        rowpos = rpos[cells]
        want = np.nonzero(rowpos >= 0)[0]
        if want.size == 0:
            return
        # column factors without the reference exponential: dt phi1 C, (nv, Np)
        cb = [dt * p1 * Cf for Cf in C]
        rb = [p1 * Rf for Rf in R]
        # diagonal
        for fr in range(nf):
            for fc in range(nf):
                d = np.sum(R[fr][:, want] * dt[:, want] * p2[:, want] * C[fc][:, want], axis=0)
                M[fr, rowpos[want], fc, cells[want]] += d
        # blocks of path positions containing wanted rows
        bs = self.block
        ReL = L.real
        s = int(want[0])
        last = int(want[-1])
        while s <= last:
            # grow the block while the exponent spread stays bounded
            e = min(s + bs, Np)
            while e - s > 1 and np.max(ReL[:, s] - ReL[:, e]) > 200:
                e = s + max(1, (e - s) // 2)
            rws = np.arange(s, e)
            rws = rws[rowpos[rws] >= 0]
            if rws.size and rws[-1] > 0:
                Lref = L[:, s]
                er = np.exp(L[:, rws] - Lref[:, None])  # |.| <= 1
                ncol = rws[-1]  # columns m' < m <= rws[-1]
                ec = np.exp(Lref[:, None] - L[:, 1:ncol + 1])  # m' = 0..ncol-1, L_{m'+1}
                A = np.concatenate([(rb[f][:, rws] * er).T for f in range(nf)], axis=0)
                Bm = np.concatenate([cb[f][:, :ncol] * ec for f in range(nf)], axis=1)
                K = (A @ Bm).reshape(nf, rws.size, nf, ncol)
                # strictly lower: m' < m
                mask = np.arange(ncol)[None, :] < rws[:, None]
                K *= mask[None, :, None, :]
                # scatter (cells of columns may repeat only for trapped paths; passing: unique)
                rr, cc = rowpos[rws][:, None], cells[None, :ncol]
                for fc in range(nf):
                    M[:, rr, fc, cc] += K[:, :, fc, :]
            s = e

    def _add_trapped(self, M, sp, isp, omega, rpos):
        """Periodic bounce response in each well (dense per well), both directions."""
        for T in self.trp:
            cellsw = T["cells"]  # (nW, Mc)
            rp = rpos[cellsw]
            keep = np.any(rp >= 0, axis=1)
            if not keep.any():
                continue
            cellsw, rp = cellsw[keep], rp[keep]
            tau, dK = T["tau"][keep], T["dK"][keep]
            a1, a2, a3 = [g[:, keep] for g in T["gyro"][isp]]  # (nE, nW, Mc)
            x, dt, c = self._omega_factors(sp, omega, tau, dK)  # (nE, nW, Mc)
            nE, nW, Mc = x.shape
            wv = (self.wE * self.wlt[T["il"]])[:, None, None]
            # path: forward (sigma = +1) cells 0..Mc-1, backward (sigma = -1) Mc-1..0
            X = np.concatenate([x, x[..., ::-1]], -1)
            DT = np.concatenate([dt, dt[..., ::-1]], -1)
            rf, cf = self._rowcol(sp, a1, a2, a3, tau[None], wv, 1)
            rbk, cbk = self._rowcol(sp, a1, a2, a3, tau[None], wv, -1)
            nf = len(rf)
            Rr = [np.concatenate([rf[f], rbk[f][..., ::-1]], -1) for f in range(nf)]
            Cc = [np.concatenate([cf[f], cbk[f][..., ::-1]], -1) * c[:, None, None] for f in range(nf)]
            n2 = 2 * Mc
            p1, p2 = _phi1(X), _phi2(X)
            L = np.concatenate([np.zeros(X.shape[:-1] + (1,), complex), np.cumsum(X, -1)], -1)
            Ptot = np.exp(L[..., -1])  # (nE, nW)
            # g_in(m) = sum_m' Rin(m, m') c_m',  c = dt phi1 s
            Lm = L[..., :n2]  # L_m
            Lq = L[..., 1:]  # L_{m'+1}
            D = Lm[..., :, None] - Lq[..., None, :]  # (nE, nW, n2, n2)
            low = np.arange(n2)[:, None] > np.arange(n2)[None, :]
            inv = 1 / (1 - Ptot)[..., None, None]
            Rin = np.where(low, np.exp(D) * inv, np.exp(D + L[..., -1][..., None, None]) * inv)
            # gbar(m) = phi1_m g_in(m) + dt_m phi2_m s_m
            Kv = p1[..., :, None] * Rin * (DT * p1)[..., None, :]
            idx = np.arange(n2)
            Kv[..., idx, idx] += DT * p2
            # fold path positions onto cells: position m -> cell index within well
            # fold path positions onto the well's cells: forward position a and backward position
            # 2 Mc - 1 - a are the same cell.  Columns: Y[g, e, w, m, b] = sum over the two
            # positions q of cell b of Kv[e, w, m, q] C[g, e, w, q]
            C4 = np.stack(Cc)  # (nf, nE, nW, n2)
            Y = Kv[None] * C4[:, :, :, None, :]  # (nf, nE, nW, n2, n2)
            Y = Y[..., :Mc] + Y[..., ::-1][..., :Mc]
            Yf, Yb = Y[..., :Mc, :], Y[..., ::-1, :][..., :Mc, :]
            Rs = np.stack(Rr)
            Rf, Rb = Rs[..., :Mc], Rs[..., ::-1][..., :Mc]
            # K[f, g, w, a, b] = sum_e (Rf[f, e, w, a] Yf[g, e, w, a, b] + Rb[...] Yb[...])
            K = np.einsum("fewa,gewab->fgwab", Rf, Yf, optimize=True)
            K += np.einsum("fewa,gewab->fgwab", Rb, Yb, optimize=True)
            nWk = cellsw.shape[0]
            for iw in range(nWk):
                ok = rp[iw] >= 0
                if not ok.any():
                    continue
                rr = rp[iw][ok][:, None]
                cc = cellsw[iw][None, :]
                for fr in range(nf):
                    for fc in range(nf):
                        M[fr, rr, fc, cc] += K[fr, fc, iw][ok]

    # ------------------------------------------------------------------ eigenproblem
    def _parity_setup(self, parity):
        """Half-domain reduction for parity in ('twisting', 'tearing') on symmetric problems, or
        None (full).  Returns (rows_cells, columns map function, normalising index)."""
        N, nf = self.N, len(self.fields)
        if parity is None:
            k0 = int(np.argmin(np.abs(self.theta - self.theta0)))
            fr = "phi" if "apar" not in self.fields else "phi"
            return np.arange(N), None, self.fields.index(fr) * N + k0
        if not self.symmetric:
            raise ValueError("parity reduction needs theta0 = 0 and an up-down symmetric surface")
        half = np.arange(N // 2, N)
        sg = {"twisting": dict(phi=1, apar=-1, bpar=1), "tearing": dict(phi=-1, apar=1, bpar=-1)}[parity]
        signs = np.array([sg[f] for f in self.fields])
        norm = "phi" if parity == "twisting" or "apar" not in self.fields else "apar"
        return half, signs, self.fields.index(norm) * (N // 2) + 0

    def matrix(self, omega, parity=None):
        rows, signs, r = self._parity_setup(parity)
        Mf = self.assemble(omega, rows)
        if signs is None:
            return Mf, r
        N, nf, h = self.N, len(self.fields), self.N // 2
        Mf = Mf.reshape(-1, nf, N)
        right = Mf[:, :, h:]
        left = Mf[:, :, :h][:, :, ::-1]  # cell N-1-k for k = h..N-1
        Mr = right + signs[None, :, None] * left
        return Mr.reshape(-1, nf * h), r

    def D(self, omega, parity=None):
        """Bordered dispersion function 1/[M^-1]_rr and the solution of M x = e_r."""
        M, r = self.matrix(complex(omega), parity)
        lu = lu_factor(M, check_finite=False)
        e = np.zeros(M.shape[0], complex)
        e[r] = 1.0
        xv = lu_solve(lu, e, check_finite=False)
        self._last = (complex(omega), parity, xv)
        return 1.0 / xv[r]

    def find_root(self, omega0, parity=None, tol=1e-7, maxit=40, verbose=False, dw=None):
        """Secant iteration on D(omega) from omega0 (model sign).  Returns a result dict."""
        t0 = time.time()
        x0 = complex(omega0)
        dw = dw if dw is not None else 0.02 * abs(x0) + 1e-3
        x1 = x0 + dw * (1 + 1j) / np.sqrt(2)
        f0, f1 = self.D(x0, parity), self.D(x1, parity)
        conv = False
        it = 0
        for it in range(1, maxit + 1):
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if x2.imag <= 1e-6:
                x2 = complex(x2.real, max(1e-6, 0.5 * x1.imag))
            x0, f0 = x1, f1
            x1, f1 = x2, self.D(x2, parity)
            if verbose:
                print(f"   it {it}: omega {x1:.7g}  |D| {abs(f1):.3g}  ({time.time() - t0:.1f} s)",
                      flush=True)
            if abs(x1 - x0) < tol * max(abs(x1), 1e-3):
                conv = True
                break
        return self.result(x1, parity, conv, it, time.time() - t0)

    def eigenvector(self, parity=None):
        """Fields on the full theta grid for the last D() solve: dict phi, apar, bpar (cells)."""
        omega, par, xv = self._last
        N, nf = self.N, len(self.fields)
        out = {}
        if par is None:
            for i, f in enumerate(self.fields):
                out[f] = xv[i * N:(i + 1) * N].copy()
        else:
            h = N // 2
            sg = {"twisting": dict(phi=1, apar=-1, bpar=1),
                  "tearing": dict(phi=-1, apar=1, bpar=-1)}[par]
            for i, f in enumerate(self.fields):
                u = xv[i * h:(i + 1) * h]
                out[f] = np.concatenate([sg[f] * u[::-1], u])
        return out

    def result(self, omega, parity, conv, it, sec):
        f = self.eigenvector(parity)
        r = dict(omega=complex(omega), omega_gene=-complex(omega).real, gamma=complex(omega).imag,
                 converged=bool(conv), iters=it, seconds=sec, parity=parity, theta=self.theta.copy(),
                 fields=f, kperp2=self.kperp2.copy(), N=self.N, nturns=self.nturns, npt=self.npt)
        # parity diagnostics and Giacomin-like numbers (kperp^2 averages weighted by |field|^2 J)
        if "phi" in f:
            ph = f["phi"]
            w = self.V * np.abs(ph) ** 2
            r["kperp2_phi"] = float(np.sum(w * self.kperp2) / np.sum(w))
            r["edge_phi"] = float(np.abs(ph[[0, -1]]).max() / np.abs(ph).max())
        if "apar" in f:
            A = f["apar"]
            r["C_tear"] = float(abs(np.sum(A * self.Jc * self.Bc)) / np.sum(np.abs(A) * self.Jc * self.Bc))
        return r

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_deck(cls, deck, ky_ref, **kw):
        """From a gene_io.Deck at k_y rho_ref (deck units); beta, species, dpdx_term from the deck."""
        sp = species_from_deck(deck)
        rs = deck.units["rho_s_over_rho_ref"]
        bp = deck.solver_kw.get("bp_e", 1.0)
        fields = kw.pop("fields", ("phi", "apar") + (("bpar",) if deck.solver_kw.get("bpar") != "off" else ()))
        S = cls(deck.geo, sp, ky_ref * rs, deck.params["beta"], bp=bp, fields=fields, **kw)
        S.deck_units = deck.units
        return S
