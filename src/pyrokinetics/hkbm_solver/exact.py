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
trapped particles over a bounce period in each well (periodic).  phi and b use continuous linear
elements on cell edges; A_par is constant in each cell (npt per 2 pi).  Per orbit segment the
exponential integrator uses a source linear in transit time, with quadrature for the time of flight,
drift phase and gyroaverages (bounce/peak singularities mapped out).  Segments adjacent to a bounce
point are subdivided to resolve theta's nonlinear dependence on transit time.  The response is
semi-separable in (theta, theta'), so the dense field matrix M(omega)
is assembled with blocked matrix products (O(N^2 n_v), stable for Im omega > 0).  Field equations
(Galerkin, matching node/cell test functions, volume element J dtheta):

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
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.special import j0 as besselj0
from scipy.special import j1 as besselj1
from scipy.special import roots_genlaguerre, roots_legendre

PREF = 0.5 / np.sqrt(
    np.pi
)  # pi^-3/2 x (pi/2): d^3v F0/n per sigma in (E, lam), weight sqrt(E) e^-E


def _I1(a):
    """2 J1(a)/a."""
    aa = np.where(a > 1e-8, a, 1.0)
    return np.where(a > 1e-8, 2 * besselj1(aa) / aa, 1.0 - a**2 / 8)


class Species:
    """One kinetic species in solver units (charge Z, mass m/m_i, T/T_e, n/n_e, gradients)."""

    def __init__(self, name, Z, m, T, n, omn, omt):
        self.name, self.Z, self.m, self.T, self.n = (
            name,
            float(Z),
            float(m),
            float(T),
            float(n),
        )
        self.omn, self.omt = float(omn), float(omt)
        self.vt = np.sqrt(2 * self.T / self.m)

    def __repr__(self):
        return (
            f"Species({self.name}, Z={self.Z}, m={self.m:.6g}, T={self.T}, n={self.n}, "
            f"omn={self.omn}, omt={self.omt})"
        )


def species_from_deck(deck):
    """Kinetic species of a gene_io.Deck in solver units (reference T_e, n_e, m of the first ion)."""
    from .gene_io import _species_list

    sp = _species_list(deck.nml)
    e = [s for s in sp if float(s["charge"]) < 0][0]
    ion = [s for s in sp if float(s["charge"]) > 0][0]
    Te, ne, mi = float(e["temp"]), float(e["dens"]), float(ion["mass"])
    out = []
    for s in sp:
        out.append(
            Species(
                str(s.get("name", "s")).strip(),
                float(s["charge"]),
                float(s["mass"]) / mi,
                float(s["temp"]) / Te,
                float(s["dens"]) / ne,
                float(s["omn"]),
                float(s["omt"]),
            )
        )
    return out


class ExtendedGeometry:
    """Geometry on the extended ballooning line theta in [-pi - 2 pi nt, pi + 2 pi nt] for ballooning
    angle theta0 (turn j = Geo.shifted(theta0 - 2 pi j)), as cubic splines in theta."""

    def __init__(self, geo, theta0=0.0, nturns=8, bp=1.0):
        self.geo, self.theta0, self.nt, self.bp = (
            geo,
            float(theta0),
            int(nturns),
            float(bp),
        )
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
        return dict(
            B=B,
            J=self.sJ(th),
            Ky=Ky,
            Kc=Kc,
            Kg=Kc + self.bp * self.dpdx / (2 * B),
            gyy=self.sgyy(th),
        )


def _phis(x):
    """phi_k(x) = sum_n x^n/(n + k)!, k = 1..4 (phi_1 = (e^x - 1)/x, ...): series for |x| < 0.5,
    upward recurrence phi_{k+1} = (phi_k - 1/k!)/x otherwise."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 0.5
    xs = np.where(small, 1.0, x)
    p1 = np.expm1(xs) / xs
    p2 = (p1 - 1.0) / xs
    p3 = (p2 - 0.5) / xs
    p4 = (p3 - 1.0 / 6) / xs
    out = []
    fact = [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
    ]
    series_x = np.where(small, x, 0.0)
    for k, pk in zip((1, 2, 3, 4), (p1, p2, p3, p4)):
        ser = sum(series_x**n / fact[n + k] for n in range(12))
        out.append(np.where(small, ser, pk))
    return out


class ExactSolver:
    """Exact collisionless linear EM gyrokinetics at one (k_y, theta0).

    geo        geometry.Geo;  species: list of Species;  ky: k_y rho_s;  beta: beta_e at B_ref
    nturns     ballooning turns each side of the central one (domain |theta| <= pi (2 nturns + 1))
    npt        cells per 2 pi (even).  phi and b are continuous piecewise linear (values on the cell
               edges, "nodes"), A_par is constant per cell: E_par = -d_l phi + i omega A_par is then
               piecewise constant and can vanish exactly (staggered fields)
    nE, nlp, nlt  energy (Gauss-Laguerre) and passing pitch nodes; trapped pitch nodes per interval
               between the bounce-point-on-cell-edge pitches 1/B(edge)
    nq         quadrature nodes per segment (geometry averages)
    fields     subset of ("phi", "apar", "bpar")
    bp         1: dpdx in the grad-B drift (full_drift); 0: gradB_eq_curv
    drift_sign multiplies omega_d (diagnostic)
    """

    def __init__(
        self,
        geo,
        species,
        ky,
        beta,
        theta0=0.0,
        nturns=8,
        npt=32,
        nE=12,
        nlp=12,
        nlt=4,
        nq=6,
        fields=("phi", "apar", "bpar"),
        bp=1.0,
        drift_sign=1.0,
        trapped=True,
        block=None,
        species_on=None,
        nbs=4,
    ):
        if npt % 2:
            raise ValueError("npt must be even")
        self.geo, self.species, self.ky, self.beta = (
            geo,
            list(species),
            float(ky),
            float(beta),
        )
        self.theta0, self.nturns, self.npt = float(theta0), int(nturns), int(npt)
        self.fields = tuple(f for f in ("phi", "apar", "bpar") if f in fields)
        self.drift_sign, self.trapped = float(drift_sign), bool(trapped)
        self.nbs = int(nbs)
        self.species_on = (
            list(species_on) if species_on is not None else [True] * len(self.species)
        )
        self.eg = ExtendedGeometry(geo, theta0, nturns, bp)
        self.N = (2 * self.nturns + 1) * self.npt
        self.dth = 2 * np.pi / self.npt
        self.edges = self.eg.lo + self.dth * np.arange(self.N + 1)
        self.theta = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.block = block if block is not None else self.npt
        # unknown layout: phi (N+1 nodes), apar (N cells), bpar (N+1 nodes)
        self.ftype = {"phi": "node", "apar": "cell", "bpar": "node"}
        self.size = {
            f: (self.N + 1 if self.ftype[f] == "node" else self.N) for f in self.fields
        }
        off, o = {}, 0
        for f in self.fields:
            off[f] = o
            o += self.size[f]
        self.off, self.nunk = off, o
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
        """Galerkin matrices of the local terms: node mass matrices with weights J and J B^2
        (tridiagonal: per cell the 2 x 2 block int w_a w_b), int_cell J k_perp^2 per cell.
        """
        x, w = roots_legendre(2 * nq)
        a, b = self.edges[:-1], self.edges[1:]
        u = 0.5 * (x + 1)
        th = a[:, None] + (b - a)[:, None] * u[None, :]
        ww = (b - a)[:, None] * 0.5 * w[None, :]
        G = self.eg.at(th)
        J = G["J"]
        hat = np.stack([1 - u, u])  # (2, nq)
        self.mJ = np.einsum("cq,iq,jq->cij", J * ww, hat, hat)  # (N, 2, 2)
        self.mJB2 = np.einsum("cq,iq,jq->cij", J * G["B"] ** 2 * ww, hat, hat)
        self.mJkperp2 = np.einsum(
            "cq,iq,jq->cij", J * self.ky**2 * G["gyy"] * ww, hat, hat
        )
        self.K2 = (J * self.ky**2 * G["gyy"] * ww).sum(1)
        self.V = (J * ww).sum(1)
        self.VB = (J * G["B"] * ww).sum(1)
        gc = self.eg.at(self.theta)
        self.kperp2 = self.ky**2 * gc["gyy"]
        self.Bc, self.Jc = gc["B"], gc["J"]
        gn = self.eg.at(self.edges)
        self.kperp2_n = self.ky**2 * gn["gyy"]
        self.Jn = gn["J"]

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
        tau = int JB/sqrt(1 - lam B) dtheta, dK = int JB w_hat/sqrt dtheta
        (w_hat = [lam B K_g + 2 (1 - lam B) K_c]/B) and the node arrays needed for the E-dependent
        gyroaverages (weights JB/sqrt wq, JB wq, k_perp^2 lam/B, lam B)."""
        G = self.eg.at(th)
        B = G["B"]
        lB = lam * B
        sq = np.sqrt(np.clip(1 - lB, 1e-300, None))
        JB = G["J"] * B
        wt = JB / sq * wq
        what = (lB * G["Kg"] + 2 * (1 - lB) * G["Kc"]) / B
        return dict(
            tau=wt.sum(-1),
            dK=(wt * what).sum(-1),
            wt=wt,
            wl=JB * wq,
            k2lB=self.ky**2 * G["gyy"] * lam / B,  # a^2 = 2 T m E k2lB / Z^2
            lB=lB,
        )

    def _gyro(self, seg, sp):
        """E-dependent segment averages for species sp: a1 = <J0>_t, a2 = int J0 dl/tau (x sigma v_t
        sqrt E gives <v_par J0>_t), a3 = <lam B I1>_t (x E gives <mu B I1>_t).  Shapes (nE, ...).
        """
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
        smallest) use theta = p + eps sinh(u) so the near-singular transit is resolved.
        """
        eg = self.eg
        x, w = roots_legendre(nq)
        a, b = self.edges[:-1], self.edges[1:]
        nl = self.lp.size
        p = eg.zmax + 2 * np.pi * np.round((0.5 * (a + b) - eg.zmax) / (2 * np.pi))
        near = (p > a - self.dth) & (p < b + self.dth)
        th = np.empty((nl, self.N, nq))
        wq = np.empty((nl, self.N, nq))
        th[:] = 0.5 * (a + b)[None, :, None] + 0.5 * (b - a)[None, :, None] * x
        wq[:] = 0.5 * (b - a)[None, :, None] * w
        c = np.sqrt(self.lp * abs(eg.d2Bmax) / 2)
        eps = np.maximum(self.tp, 1e-12) / np.maximum(c, 1e-12)
        idx = np.nonzero(near)[0]
        for il in range(nl):
            ua = np.arcsinh((a[idx] - p[idx]) / eps[il])
            ub = np.arcsinh((b[idx] - p[idx]) / eps[il])
            u = 0.5 * (ua + ub)[:, None] + 0.5 * (ub - ua)[:, None] * x
            th[il, idx] = p[idx, None] + eps[il] * np.sinh(u)
            wq[il, idx] = 0.5 * (ub - ua)[:, None] * w * eps[il] * np.cosh(u)
        seg = self._seg_moments(th, wq, self.lp[:, None, None])
        self.pas = dict(
            tau=seg["tau"],
            dK=seg["dK"],
            gyro=[self._gyro(seg, sp) for sp in self.species],
        )

    def _wells(self, lam):
        """Wells of lam (trapped) in one period starting at the B maximum: list of (z1, z2)."""
        g = self.geo
        z0 = self.eg.zmax
        zs = z0 + np.linspace(0, 2 * np.pi, 4001)
        s = 1 - lam * g.sB(zs) > 0
        out = []
        i = 0
        while i < zs.size:
            if s[i]:
                j = i
                while j + 1 < zs.size and s[j + 1]:
                    j += 1
                z1 = (
                    brentq(lambda z: 1 - lam * g.sB(z), zs[i - 1], zs[i])
                    if i > 0
                    else zs[0]
                )
                z2 = (
                    brentq(lambda z: 1 - lam * g.sB(z), zs[j], zs[j + 1])
                    if j + 1 < zs.size
                    else zs[-1]
                )
                out.append((z1, z2))
                i = j + 1
            else:
                i += 1
        return out

    def _trapped(self, nq):
        """Trapped orbits: for each trapped pitch and well, the segments between the bounce points
        (points P_0 = z1, the cell edges, P_Mc = z2), segment integrals with theta = c + hw sin(phi)
        (bounce singularities removed), for every turn of the domain (wells cut by the domain ends
        are dropped).  Point j lies in local cell min(j, Mc - 1) at fraction alpha_j."""
        x, w = roots_legendre(nq)
        lo, dth = self.eg.lo, self.dth
        self.trp = []
        for il, lam in enumerate(self.lt):
            for z1, z2 in self._wells(lam):
                c0, hw = 0.5 * (z1 + z2), 0.5 * (z2 - z1)
                js = [
                    j
                    for j in range(-self.nturns - 1, self.nturns + 2)
                    if z1 + 2 * np.pi * j >= self.eg.lo
                    and z2 + 2 * np.pi * j <= self.eg.hi
                ]
                if not js:
                    continue
                js = np.array(js)
                k1 = int(np.floor((z1 - lo) / dth))
                k2 = int(np.floor((z2 - lo) / dth))
                if lo + dth * k2 >= z2 and k2 > k1:  # z2 exactly on an edge
                    k2 -= 1
                cells = np.arange(k1, k2 + 1)
                Mc = cells.size
                # points: bounce point, cell edges, bounce point; the end segments (next to the
                # bounce points, where theta(t) is far from linear) are split into nbs pieces
                # uniform in the sin-map angle
                pz = np.concatenate([[z1], lo + dth * cells[1:], [z2]])
                phz = np.arcsin(np.clip((pz - c0) / hw, -1, 1))
                phz[0], phz[-1] = -0.5 * np.pi, 0.5 * np.pi
                nbs = self.nbs
                if Mc == 1:
                    php = np.linspace(phz[0], phz[1], 2 * nbs + 1)
                else:
                    last = np.linspace(phz[-2], phz[-1], nbs + 1)
                    php = np.concatenate(
                        [
                            np.linspace(phz[0], phz[1], nbs + 1),
                            phz[2:-2],
                            last[1:] if Mc == 2 else last,
                        ]
                    )
                pts = c0 + hw * np.sin(php)
                pts[0], pts[-1] = z1, z2
                ns = pts.size - 1
                segc = np.clip(
                    np.floor((0.5 * (pts[:-1] + pts[1:]) - lo) / dth).astype(int) - k1,
                    0,
                    Mc - 1,
                )  # local cell of each segment
                pcl = np.clip(np.floor((pts - lo) / dth).astype(int) - k1, 0, Mc - 1)
                alpha = (pts - (lo + dth * (k1 + pcl))) / dth
                pa, pb = php[:-1], php[1:]
                ph = 0.5 * (pa + pb)[:, None] + 0.5 * (pb - pa)[:, None] * x
                wph = 0.5 * (pb - pa)[:, None] * w * hw * np.cos(ph)
                thz = c0 + hw * np.sin(ph)
                th = thz[None] + 2 * np.pi * js[:, None, None]
                wq = np.broadcast_to(wph, th.shape)
                seg = self._seg_moments(th, wq, lam)
                self.trp.append(
                    dict(
                        il=il,
                        lam=lam,
                        k1=k1 + self.npt * js,
                        Mc=Mc,
                        ns=ns,
                        segc=segc,
                        pcl=pcl,
                        alpha=alpha,
                        tau=seg["tau"],
                        dK=seg["dK"],
                        gyro=[self._gyro(seg, sp) for sp in self.species],
                    )
                )

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
        """Row (moment) and column (source) factors per segment, lists over the active fields
        (QN/phi, Ampere/A_par, perpendicular balance/b), arrays shaped like a1."""
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

    def assemble(self, omega, rowsel=None):
        """Dense field matrix M(omega), rows = equations at the unknowns in rowsel (global unknown
        indices; default all), columns = all unknowns."""
        rs = np.arange(self.nunk) if rowsel is None else np.asarray(rowsel)
        rpos = -np.ones(self.nunk, int)
        rpos[rs] = np.arange(rs.size)
        M = np.zeros((rs.size, self.nunk), complex)
        # adiabatic (Boltzmann) parts of all species; a species switched off has h = 0
        cs_tot = sum(sp.Z**2 * sp.n / sp.T for sp in self.species)
        N = self.N
        k = np.arange(N)
        for f in self.fields:
            o = self.off[f]
            if self.ftype[f] == "cell":
                gi = o + k
                ok = rpos[gi] >= 0
                M[rpos[gi[ok]], gi[ok]] += self.K2[ok]
            else:
                mm = cs_tot * self.mJ if f == "phi" else self.mJB2
                for i in (0, 1):
                    for j in (0, 1):
                        gi, gj = o + k + i, o + k + j
                        ok = rpos[gi] >= 0
                        M[rpos[gi[ok]], gj[ok]] += mm[ok, i, j]
        for isp, sp in enumerate(self.species):
            if not self.species_on[isp]:
                continue
            self._add_passing(M, sp, isp, omega, rpos)
            if self.trapped:
                self._add_trapped(M, sp, isp, omega, rpos)
        return M

    def _end_map(self, f, nodes_a, nodes_b, cells):
        """Global unknown indices of field f at segment ends a, b (node fields) or of the segment's
        cell (cell fields)."""
        o = self.off[f]
        if self.ftype[f] == "cell":
            return o + cells, o + cells
        return o + nodes_a, o + nodes_b

    def _add_passing(self, M, sp, isp, omega, rpos):
        P = self.pas
        a1, a2, a3 = P["gyro"][isp]  # (nE, nl, N)
        x, dt, c = self._omega_factors(sp, omega, P["tau"], P["dK"])
        wv = (self.wE[:, None] * self.wlp[None, :])[:, :, None]
        nE, nl, N = x.shape
        c3 = c[:, None, None]
        for sigma in (1, -1):
            sl = slice(None) if sigma > 0 else slice(None, None, -1)
            rows, cols = self._rowcol(sp, a1, a2, a3, P["tau"][None], wv, sigma)
            xs = x[..., sl].reshape(nE * nl, N)
            dts = dt[..., sl].reshape(nE * nl, N)
            R = [r[..., sl].reshape(nE * nl, N) for r in rows]
            C = [(cc * c3)[..., sl].reshape(nE * nl, N) for cc in cols]
            cells = np.arange(N)[sl]
            na = (
                cells if sigma > 0 else cells + 1
            )  # node at the start of each path segment
            nb = cells + 1 if sigma > 0 else cells
            self._path_kernel(M, xs, dts, R, C, cells, na, nb, rpos)

    def _path_kernel(self, M, x, dt, R, C, cells, na, nb, rpos):
        """Open-path (passing) response with linear-in-time sources between the segment ends.
        x, dt: (nv, Np) in path order; R, C: per field (nv, Np) row/column factors of each segment;
        cells, na, nb: (Np,) cell and end nodes of each path segment.  In end space (segment m, end
        e in a/b): G_a = phi2 g_in + dt [(phi3 - phi4) s_a + phi4 s_b],
                   G_b = (phi1 - phi2) g_in + dt [(phi2 - 2 phi3 + phi4) s_a + (phi3 - phi4) s_b],
        g_in(m) = sum_{m' < m} e^{L_m - L_{m'+1}} dt_m' [(phi1 - phi2) s_a + phi2 s_b]_m'.
        """
        nv, Np = x.shape
        p1, p2, p3, p4 = _phis(x)
        L = np.concatenate([np.zeros((nv, 1), complex), np.cumsum(x, axis=1)], axis=1)
        nf = len(R)
        fl = self.fields
        emap = [
            self._end_map(f, na, nb, cells) for f in fl
        ]  # per field: (idx_a, idx_b)
        rowok = [(rpos[ea] >= 0) | (rpos[eb] >= 0) for ea, eb in emap]
        want = np.nonzero(np.any(np.array(rowok), axis=0))[0]
        if want.size == 0:
            return
        rend = [p2, p1 - p2]  # row end factors (times g_in)
        cend = [dt * (p1 - p2), dt * p2]  # column end factors
        diag = [[dt * (p3 - p4), dt * p4], [dt * (p2 - 2 * p3 + p4), dt * (p3 - p4)]]

        def scatter(K, rws, ncol):
            # K: (nf, 2, nr, nf, 2, ncol) end-space block -> M
            for fr in range(nf):
                for er in range(2):
                    gi = emap[fr][er][rws]
                    pr = rpos[gi]
                    ok = pr >= 0
                    if not ok.any():
                        continue
                    for fc in range(nf):
                        for ec in range(2):
                            gj = emap[fc][ec][:ncol]
                            M[pr[ok][:, None], gj[None, :]] += K[fr, er][ok][:, fc, ec]

        # diagonal (m' = m)
        for fr in range(nf):
            for er in range(2):
                gi = emap[fr][er][want]
                pr = rpos[gi]
                ok = pr >= 0
                for fc in range(nf):
                    for ec in range(2):
                        gj = emap[fc][ec][want]
                        d = np.sum(
                            R[fr][:, want] * diag[er][ec][:, want] * C[fc][:, want],
                            axis=0,
                        )
                        np.add.at(M, (pr[ok], gj[ok]), d[ok])
        bs = self.block
        ReL = L.real
        s = int(want[0])
        last = int(want[-1])
        while s <= last:
            e = min(s + bs, Np)
            while e - s > 1 and np.max(ReL[:, s] - ReL[:, e]) > 200:
                e = s + max(1, (e - s) // 2)
            rws = np.arange(s, e)
            rws = rws[np.isin(rws, want)]
            if rws.size and rws[-1] > 0:
                Lref = L[:, s]
                er_ = np.exp(L[:, rws] - Lref[:, None])
                ncol = int(rws[-1])
                ec_ = np.exp(Lref[:, None] - L[:, 1 : ncol + 1])
                A = np.concatenate(
                    [
                        (R[f][:, rws] * er_ * rend[e2][:, rws]).T
                        for f in range(nf)
                        for e2 in range(2)
                    ],
                    axis=0,
                )
                Bm = np.concatenate(
                    [
                        C[f][:, :ncol] * cend[e2][:, :ncol] * ec_
                        for f in range(nf)
                        for e2 in range(2)
                    ],
                    axis=1,
                )
                K = (A @ Bm).reshape(nf, 2, rws.size, nf, 2, ncol)
                mask = np.arange(ncol)[None, :] < rws[:, None]
                K *= mask[None, None, :, None, None, :]
                scatter(K, rws, ncol)
            s = e

    def _add_trapped(self, M, sp, isp, omega, rpos):
        """Periodic bounce response in each well (dense per well), both directions, linear-in-time
        sources between the points P_j (bounce points and cell edges)."""
        nf = len(self.fields)
        for T in self.trp:
            k1 = T["k1"]  # (nW,) first cell of each well
            Mc, alpha = T["Mc"], T["alpha"]
            # local unknowns per field: nodes k1 .. k1 + Mc (Mc + 1) or cells k1 .. k1 + Mc - 1
            gl = []
            for f in self.fields:
                n = Mc + 1 if self.ftype[f] == "node" else Mc
                gl.append(self.off[f] + k1[:, None] + np.arange(n)[None, :])  # (nW, n)
            rok = [rpos[g] >= 0 for g in gl]
            keep = np.any(np.concatenate(rok, axis=1), axis=1)
            if not keep.any():
                continue
            tau, dK = T["tau"][keep], T["dK"][keep]
            a1, a2, a3 = [g[:, keep] for g in T["gyro"][isp]]  # (nE, nW, Mc)
            x, dt, c = self._omega_factors(sp, omega, tau, dK)
            wv = (self.wE * self.wlt[T["il"]])[:, None, None]
            n2 = 2 * T["ns"]
            X = np.concatenate([x, x[..., ::-1]], -1)
            DT = np.concatenate([dt, dt[..., ::-1]], -1)
            rf, cf = self._rowcol(sp, a1, a2, a3, tau[None], wv, 1)
            rbk, cbk = self._rowcol(sp, a1, a2, a3, tau[None], wv, -1)
            Rr = [np.concatenate([rf[f], rbk[f][..., ::-1]], -1) for f in range(nf)]
            Cc = [
                np.concatenate([cf[f], cbk[f][..., ::-1]], -1) * c[:, None, None]
                for f in range(nf)
            ]
            p1, p2, p3, p4 = _phis(X)
            L = np.concatenate(
                [np.zeros(X.shape[:-1] + (1,), complex), np.cumsum(X, -1)], -1
            )
            P = np.exp(L[..., -1])  # (nE, nW): propagator over one bounce
            rend = [p2, p1 - p2]
            cend = [DT * (p1 - p2), DT * p2]
            dg = [[DT * (p3 - p4), DT * p4], [DT * (p2 - 2 * p3 + p4), DT * (p3 - p4)]]
            nWk = X.shape[1]
            # periodic response g_in(m) = sum_m' e^{L_m - L_m'+1} c_m' [1/(1 - P) for m' < m,
            # P/(1 - P) for m' >= m]: two rank-nE products per well (rows (f, end, m), E summed)
            q = -1 / np.expm1(L[..., -1])  # avoids cancellation when P is near 1
            sh = (nWk, nf, 2, n2, nf, 2, n2)
            low = np.arange(n2)[:, None] > np.arange(n2)[None, :]
            if np.max(-L[..., -1].real) > 600:
                # For slow particles / strongly growing trial frequencies, separate
                # row/column exponentials overflow before the triangular selection.
                # Evaluate only the physical, forward-time propagation around the
                # bounce. Its real exponent is non-positive for Im(omega) > 0.
                phase = L[..., :-1, None] - L[..., None, 1:]
                phase += np.where(low, 0, L[..., -1, None, None])
                propagator = np.exp(phase) * q[..., None, None]
                K = np.empty(sh, complex)
                for fr in range(nf):
                    for er in range(2):
                        for fc in range(nf):
                            for ec in range(2):
                                K[:, fr, er, :, fc, ec, :] = np.einsum(
                                    "ewm,ewmn,ewn->wmn",
                                    Rr[fr] * rend[er],
                                    propagator,
                                    Cc[fc] * cend[ec],
                                )
            else:
                Lc = L - L[..., n2 // 2, None]
                eR, eC = np.exp(Lc[..., :n2]), np.exp(-Lc[..., 1:])
                A = np.stack(
                    [Rr[f] * rend[e2] * eR for f in range(nf) for e2 in range(2)]
                )
                A = A.reshape(nf * 2, nE_ := A.shape[1], nWk, n2).transpose(2, 0, 3, 1)
                A = A.reshape(nWk, nf * 2 * n2, nE_)
                Bm = np.stack(
                    [Cc[f] * cend[e2] * eC for f in range(nf) for e2 in range(2)]
                )
                Bm = Bm.transpose(2, 1, 0, 3).reshape(nWk, nE_, nf * 2 * n2)
                Klow = (A * q.T[:, None, :]) @ Bm
                Kup = (A * (P * q).T[:, None, :]) @ Bm
                K = np.where(
                    low[None, None, None, :, None, None, :],
                    Klow.reshape(sh),
                    Kup.reshape(sh),
                )
            idx = np.arange(n2)
            for fr in range(nf):
                for er in range(2):
                    for fc in range(nf):
                        for ec in range(2):
                            d = np.sum(Rr[fr] * dg[er][ec] * Cc[fc], axis=0)  # (nW, n2)
                            K[:, fr, er, idx, fc, ec, idx] += d
            # interpolation of path ends to local unknowns.  Path position m < ns: forward segment
            # m from point m to m + 1; m >= ns: backward segment ns - 1 - (m - ns) from point
            # ns - (m - ns) to ns - 1 - (m - ns).
            m = np.arange(n2)
            ns = T["ns"]
            fwd = m < ns
            sidx = np.where(fwd, m, n2 - 1 - m)  # segment index of each path position
            cellp = T["segc"][sidx]
            pa = np.where(fwd, m, n2 - m)
            pb = np.where(fwd, m + 1, n2 - 1 - m)
            Wn = np.zeros(
                (2, n2, Mc + 1)
            )  # node fields: point -> nodes (cell, cell + 1)
            for e2, pp in enumerate((pa, pb)):
                cl = T["pcl"][pp]
                al = alpha[pp]
                np.add.at(Wn, (e2, m, cl), 1 - al)
                np.add.at(Wn, (e2, m, cl + 1), al)
            Wc = np.zeros((2, n2, Mc))
            Wc[0, m, cellp] = 1.0
            Wc[1, m, cellp] = 1.0
            W = [Wn if self.ftype[f] == "node" else Wc for f in self.fields]
            gk = [g[keep] for g in gl]
            for fr in range(nf):
                Wr = W[fr].reshape(2 * n2, -1)
                pr = rpos[gk[fr]]  # (nW, nr)
                okr = pr >= 0
                for fc in range(nf):
                    Wcl = W[fc].reshape(2 * n2, -1)
                    Kf = K[:, fr, :, :, fc, :, :].reshape(nWk, 2 * n2, 2 * n2)
                    Kl = Wr.T @ Kf @ Wcl  # (nW, nr, nc)
                    PR = np.broadcast_to(pr[:, :, None], Kl.shape)
                    GC = np.broadcast_to(gk[fc][:, None, :], Kl.shape)
                    ok = np.broadcast_to(okr[:, :, None], Kl.shape)
                    np.add.at(M, (PR[ok], GC[ok]), Kl[ok])

    # ------------------------------------------------------------------ eigenproblem
    def _parity_maps(self, parity):
        """Unknown reduction.  parity None: all unknowns, rows = all.  'twisting' (phi, b even,
        A_par odd) / 'tearing' (phi, b odd, A_par even), theta0 = 0 and symmetric surface:
        half-domain unknowns u (theta >= 0), full = E u.  Returns (rowsel, E or None, r).
        """
        N = self.N
        if parity is None:
            f0 = "phi" if "phi" in self.fields else self.fields[0]
            o = self.off[f0]
            if self.ftype[f0] == "node":
                r = o + int(np.argmin(np.abs(self.edges - self.theta0)))
            else:
                r = o + int(np.argmin(np.abs(self.theta - self.theta0)))
            return np.arange(self.nunk), None, r
        if not self.symmetric:
            raise ValueError(
                "parity reduction needs theta0 = 0 and an up-down symmetric surface"
            )
        sg = {
            "twisting": dict(phi=1, apar=-1, bpar=1),
            "tearing": dict(phi=-1, apar=1, bpar=-1),
        }[parity]
        h = N // 2
        rows = []
        Erows, Ecols, Evals = [], [], []
        nred = 0
        rnorm = None
        for f in self.fields:
            o, s = self.off[f], sg[f]
            if self.ftype[f] == "node":
                idx = np.arange(h, N + 1) if s > 0 else np.arange(h + 1, N + 1)
                for j, n in enumerate(idx):
                    Erows += [o + n, o + (N - n)] if n != h else [o + n]
                    Ecols += [nred + j, nred + j] if n != h else [nred + j]
                    Evals += [1.0, float(s)] if n != h else [1.0]
                if f == ("phi" if parity == "twisting" else None):
                    rnorm = nred
            else:
                idx = np.arange(h, N)
                for j, k in enumerate(idx):
                    Erows += [o + k, o + (N - 1 - k)]
                    Ecols += [nred + j, nred + j]
                    Evals += [1.0, float(s)]
                if f == "apar" and parity == "tearing":
                    rnorm = nred
            rows.append(o + idx)
            nred += idx.size
        if rnorm is None:
            rnorm = 0
        from scipy.sparse import csr_matrix

        Em = csr_matrix((Evals, (Erows, Ecols)), shape=(self.nunk, nred))
        return np.concatenate(rows), Em, rnorm

    def matrix(self, omega, parity=None):
        rows, Em, r = self._parity_maps(parity)
        Mf = self.assemble(omega, rows)
        if Em is None:
            return Mf, r
        return np.asarray((Em.T @ Mf.T).T), r

    def D(self, omega, parity=None):
        """Bordered dispersion function 1/[M^-1]_rr and the solution of M x = e_r."""
        if not np.isfinite(omega) or complex(omega).imag <= 0:
            raise ValueError("the collisionless orbit response requires Im(omega) > 0")
        M, r = self.matrix(complex(omega), parity)
        if not np.all(np.isfinite(M)):
            raise FloatingPointError("non-finite collisionless response matrix")
        lu = lu_factor(M, check_finite=False)
        e = np.zeros(M.shape[0], complex)
        e[r] = 1.0
        xv = lu_solve(lu, e, check_finite=False)
        if not np.all(np.isfinite(xv)) or xv[r] == 0:
            raise FloatingPointError("non-finite or unnormalisable field response")
        self._last = (complex(omega), parity, xv)
        scale = np.abs(M) @ np.abs(xv)
        self._last_residual = float(np.max(np.abs(M @ xv) / np.maximum(scale, 1e-300)))
        return 1.0 / xv[r]

    def find_root(
        self,
        omega0,
        parity=None,
        tol=1e-7,
        maxit=40,
        verbose=False,
        dw=None,
        timeout=None,
    ):
        """Secant iteration on D(omega) from omega0 (model sign, positive imaginary part).

        Convergence requires both a small frequency step and satisfaction of every
        field equation. A stalled iteration at the positive-growth boundary is
        unresolved, not a proof of stability or a growing eigenmode.
        """
        t0 = time.monotonic()

        def evaluate(omega):
            if timeout is not None and time.monotonic() - t0 >= timeout:
                raise TimeoutError("exact root search exceeded its time limit")
            value = self.D(omega, parity)
            if timeout is not None and time.monotonic() - t0 >= timeout:
                raise TimeoutError("exact root search exceeded its time limit")
            return value

        x0 = complex(omega0)
        dw = dw if dw is not None else 0.02 * abs(x0) + 1e-3
        x1 = x0 + dw * (1 + 1j) / np.sqrt(2)
        f0, f1 = evaluate(x0), evaluate(x1)
        conv = False
        it = 0
        for it in range(1, maxit + 1):
            if f1 == f0 or not np.isfinite(f1):
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            # damp large steps
            step = x2 - x1
            if abs(step) > 0.5 * abs(x1) + 0.05:
                x2 = x1 + step * (0.5 * abs(x1) + 0.05) / abs(step)
            if x2.imag <= 1e-6:
                x2 = complex(x2.real, max(1e-6, 0.5 * x1.imag))
            x0, f0 = x1, f1
            x1, f1 = x2, evaluate(x2)
            if verbose:
                print(
                    f"   it {it}: omega {x1:.7g}  |D| {abs(f1):.3g}  ({time.monotonic() - t0:.1f} s)",
                    flush=True,
                )
            if abs(x1 - x0) < tol * max(abs(x1), 1e-3) and self._last_residual < tol:
                conv = True
                break
        return self.result(x1, parity, conv, it, time.monotonic() - t0)

    def eigenvector(self):
        """Fields of the last D() solve: dict phi, bpar on the nodes (self.edges), apar on the cells
        (self.theta)."""
        omega, par, xv = self._last
        _, Em, _ = self._parity_maps(par)
        full = xv if Em is None else Em @ xv
        return {
            f: full[self.off[f] : self.off[f] + self.size[f]].copy()
            for f in self.fields
        }

    def result(self, omega, parity, conv, it, sec):
        f = self.eigenvector()
        r = dict(
            omega=complex(omega),
            omega_gene=-complex(omega).real,
            gamma=complex(omega).imag,
            converged=bool(conv),
            iters=it,
            seconds=sec,
            parity=parity,
            theta_nodes=self.edges.copy(),
            theta_cells=self.theta.copy(),
            fields=f,
            N=self.N,
            nturns=self.nturns,
            npt=self.npt,
        )
        r["relative_residual"] = self._last_residual
        r["status"] = "growing_root" if conv else "unresolved"
        r["collision_model"] = "none"
        if "phi" in f:
            ph = f["phi"]
            pairs = np.stack([ph[:-1], ph[1:]], axis=1)
            norm = np.einsum("ci,cij,cj->", pairs.conj(), self.mJ, pairs).real
            moment = np.einsum("ci,cij,cj->", pairs.conj(), self.mJkperp2, pairs).real
            r["kperp2_phi"] = float(moment / norm) if norm > 0 else np.nan
            r["edge_phi"] = float(np.abs(ph[[0, -1]]).max() / np.abs(ph).max())
        if "apar" in f:
            A = f["apar"]
            w = np.abs(A) ** 2
            r["kperp2_apar"] = float(np.sum(w * self.K2) / np.sum(w * self.V))
            r["C_tear"] = float(abs(np.sum(A * self.VB)) / np.sum(np.abs(A) * self.VB))
        return r

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_deck(cls, deck, ky_ref, **kw):
        """From a gene_io.Deck at k_y rho_ref (deck units); beta, species, dpdx_term from the deck."""
        if float(deck.nml["general"].get("coll", 0.0)) != 0:
            warnings.warn(
                "ExactSolver is collisionless: the deck's collision operator is not included.",
                UserWarning,
                stacklevel=2,
            )
        sp = species_from_deck(deck)
        rs = deck.units["rho_s_over_rho_ref"]
        bp = deck.solver_kw.get("bp_e", 1.0)
        fields = kw.pop(
            "fields",
            ("phi", "apar")
            + (("bpar",) if deck.solver_kw.get("bpar") != "off" else ()),
        )
        S = cls(
            deck.geo, sp, ky_ref * rs, deck.params["beta"], bp=bp, fields=fields, **kw
        )
        S.deck_units = deck.units
        return S
