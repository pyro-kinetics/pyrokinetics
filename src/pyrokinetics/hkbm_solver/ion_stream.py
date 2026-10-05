"""
ion_stream.py -- kinetic ion response integrated ALONG THE FIELD LINE (parallel streaming and bounce motion) for a
given complex omega (Im omega > 0), projected on a Galerkin basis.  Replaces the local ion response of Zocco (3.6)
(omega >> k_par v_ti) and the plane-wave k_par surrogate.

For each energy E (generalised Gauss-Laguerre, weight E^1/2 e^-E), pitch lam = mu/E and direction sigma, the ion
gyrokinetic equation (GENE h, exp(-i omega t), g = T_i h/F0)
        (omega - omega_di) g + i v_par d_l g = (omega - omega_*i^T(E)) J0 chi,     v_par = sigma v_Ti sqrt(E (1 - lam B))
is integrated along the orbit in time (dl = J B dtheta, v_Ti = sqrt(2 T_i)): dg/dt = -i S + i (omega - omega_di) g,
cell by cell with the exact exponential integrator.  Passing ions (lam < 1/B_max): theta from -pi to pi (sigma = +1)
or back (sigma = -1), g = 0 entering the turn (the mode is localised, |phi(pi)| ~ 1e-3).  Trapped ions: one full
bounce orbit theta(phi) = zc + hw sin(phi) forward then back, periodic (g(t_b) = g(0)).
Moments (projected on test functions b_m with the Galerkin weight J dtheta):
   H = int d3v J0 h,  P = int d3v mu I1 h,  D = int d3v J0 omega_di h,  L = (1/T_i) int d3v F0 (omega - omega_*^T) J0 chi
with d3v = pi B E^1/2 dE dlam / (2 sqrt(1 - lam B)) per sigma.  FLR: J0 = exp(-b B lam E/2), I1 = exp(-b B lam E/4)
(consistent with the exp forms of solver.py).  Valid for Im omega > 0 only (real-E quadrature, no continuation).
Collisions (Solver coll > 0, coll_i): Krook detrapping of the TRAPPED ions, omega -> omega + i nu_D^i(E)/eps in the orbit
integration (S.ion_nu); passing ions collisionless.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import j0 as besselj0
from scipy.special import j1 as besselj1
from scipy.special import roots_genlaguerre, roots_legendre

SQPI = np.sqrt(np.pi)


def _phi1(x):
    small = np.abs(x) < 1e-2
    xs = np.where(small, 1.0, x)
    big = np.expm1(xs) / xs
    ser = 1 + x / 2 + x**2 / 6 + x**3 / 24
    return np.where(small, ser, big)


def _phi2(x):
    small = np.abs(x) < 1e-1
    xs = np.where(small, 1.0, x)
    big = (np.expm1(xs) - xs) / xs**2
    ser = 0.5 + x / 6 + x**2 / 24 + x**3 / 120 + x**4 / 720 + x**5 / 5040
    return np.where(small, ser, big)


class StreamingIons:
    def __init__(
        self,
        S,
        nE=24,
        nlp=24,
        nlt=32,
        nthp=384,
        nphi=160,
        vscale=1.0,
        ion_psi=False,
        bessel=False,
    ):
        """S: solver.Solver (geometry, basis, switches).  vscale: multiplies v_Ti (vscale -> 0 recovers the local
        drift-kinetic ions, the self-test).  bessel: exact J0(x), 2 J1(x)/x (x^2 = 2 b B lam E) instead of the
        exponential forms exp(-x^2/4), exp(-x^2/8) (test of the FLR approximation at high k_y).
        """
        self.S = S
        self.bessel = bessel
        self.ion_psi = (
            ion_psi and S.use_psi
        )  # ions respond to A_par = -(i/omega) d_l psi (chi_A = -v_par J0 A_par)
        geo, p = S.geo, S.p
        self.Ti = Ti = p["Ti"]
        self.vT = np.sqrt(2 * Ti) * vscale
        E, wE = roots_genlaguerre(nE, 0.5)
        self.E, self.wE = E, wE
        th = geo.theta

        def spl(f):
            fr = CubicSpline(
                np.append(th, np.pi), np.append(f.real, f.real[0]), bc_type="periodic"
            )
            fi = CubicSpline(
                np.append(th, np.pi), np.append(f.imag, f.imag[0]), bc_type="periodic"
            )
            return lambda z: fr(z) + 1j * fi(z)

        self.splines = dict(
            phi=[spl(f) for f in S.Bphi],
            bpar=[spl(f) for f in S.Bb],
            psi=[spl(f) for f in S.Bpsi],
        )
        orbits = []
        # ---- passing: lam = (1 - t^2)/Bmax, Gauss-Legendre in t on (0, 1)
        t, wt = roots_legendre(nlp)
        t = 0.5 * (t + 1)
        wt = 0.5 * wt
        lam_p = (1 - t**2) / geo.Bmax
        wlam_p = wt * 2 * t / geo.Bmax
        zc = -np.pi + (np.arange(nthp) + 0.5) * 2 * np.pi / nthp
        dz = 2 * np.pi / nthp
        z = np.broadcast_to(zc[None, :], (nlp, nthp))
        orbits.append(self._geom(z, np.full(z.shape, dz), lam_p, wlam_p, trapped=False))
        # ---- trapped: Gauss-Legendre in lam on (1/Bmax, 1/Bmin)
        x, wx = roots_legendre(nlt)
        lmin, lmax = 1 / geo.Bmax, 1 / geo.Bmin
        lam_t = lmin + (lmax - lmin) * 0.5 * (x + 1)
        wlam_t = wx * 0.5 * (lmax - lmin)
        ph = (np.arange(nphi) + 0.5) / nphi * np.pi - np.pi / 2
        zz, dd = [], []
        for l in lam_t:  # noqa: E741
            z1, z2 = geo.bounce_points(l)
            c, hw = 0.5 * (z1 + z2), 0.5 * (z2 - z1)
            zz.append(c + hw * np.sin(ph))
            dd.append(hw * np.cos(ph) * np.pi / nphi)
        orbits.append(
            self._geom(np.array(zz), np.array(dd), lam_t, wlam_t, trapped=True)
        )
        self.orbits = orbits

    def _geom(self, z, dz, lam, wlam, trapped):
        """Geometry on an orbit family: arrays (nlam, ncell)."""
        S = self.S
        geo = S.geo
        g = geo.g
        B = g.sB(z)
        J = g.sJ(z)
        K = g.sK(z)
        Kc = K - geo.dpdx / (2 * B)
        Kg = Kc + S.sw["bp_i"] * geo.dpdx / (2 * B)
        lB = lam[:, None] * B
        sq = np.sqrt(np.clip(1 - lB, 1e-14, None))
        o = dict(
            z=z,
            lam=lam,
            wlam=wlam,
            trapped=trapped,
            B=B,
            lB=lB,
            dl_over_v=J * B * dz / sq,  # dl / sqrt(1 - lam B)   (time x v_T sqrt(E))
            wdrift=S.ky * self.Ti * (lB * Kg + 2 * (1 - lB) * Kc) / B,  # omega_di / E
            wbeta=S.ky
            * self.Ti
            * lB
            * geo.dpdx
            / (2 * B)
            / B,  # beta' part of omega_di / E (bp_i = 1)
            bB=S.ky**2
            * geo.sgyy(z)
            * self.Ti
            / B**2
            * B
            * lam[:, None],  # b B lam (FLR argument / E)
            wgt=J * B * dz / sq,
        )  # moment weight J B dtheta/sqrt(1 - lam B)
        o["basis"] = {
            f: np.array([s(z) for s in self.splines[f]]) for f in ("phi", "bpar", "psi")
        }  # (nb, nlam, ncell)
        return o

    def moments(self, omega, split=False):
        """Projected ion moment matrices: dict[(moment, test_field, source_field)] -> (n_test, n_src) arrays.
        moment in 'H' (density, J0), 'P' (mu I1), 'D' (J0 omega_di), 'L' (local source moment);
        test_field in phi/bpar/psi (test basis), source_field in phi (chi = J0 phi) / bpar (chi = v_i T_i mu I1 dB_par).
        """
        S = self.S
        omega = complex(omega)
        Ti, vT = self.Ti, self.vT
        vl = S.sw["vl_i"]
        ai, bi = S.ai, S.bi
        pref = (
            (1 / Ti) * np.pi**-1.5 * (np.pi / 2)
        )  # (1/T_i) F0 normalisation x d3v factor per sigma
        out = {}
        outs = []
        for o in self.orbits:
            if split:
                out = {}
            E = self.E[:, None, None]  # (nE, 1, 1)
            wE = self.wE[:, None, None]
            lam = o["lam"][None, :, None]
            if self.bessel:
                xx = np.sqrt(2 * o["bB"][None] * E)
                J0 = besselj0(xx)
                I1 = np.where(
                    xx > 1e-8, 2 * besselj1(xx) / np.where(xx > 1e-8, xx, 1), 1.0
                )
            else:
                J0 = np.exp(-0.5 * o["bB"][None] * E)  # (nE, nlam, ncell)
                I1 = np.exp(-0.25 * o["bB"][None] * E)
            mu = lam * E
            wd = o["wdrift"][None] * E
            if vT > 0:
                dt = o["dl_over_v"][None] / (vT * np.sqrt(E))  # cell transit time
            else:
                dt = np.full(J0.shape, np.inf)
            omk = omega
            if o["trapped"] and getattr(S, "ion_nu", None) is not None:
                omk = (
                    omega + 1j * S.ion_nu(self.E)[:, None, None]
                )  # Krook detrapping of trapped ions (collisions)
            a = 1j * (omk - wd)
            x = a * dt
            # precession vs direct drive test (a_i, d_i): phi response = h[shift a r] + d (h[0] - h[r]), shift s: omega_di -> omega_di
            # - s beta'-part; each part's D moment with its own drift (the TCH is the moment of that part's GK equation)
            if getattr(S, "dec", {}).get("i"):
                a_i, d_i, rr = S.sw["a_i"], S.sw["d_i"], S.r_eff
                iparts = [(a_i * rr, 1.0)] + (
                    [(0.0, d_i), (rr, -d_i)] if d_i != 0 else []
                )
            else:
                iparts = [(0.0, 1.0)]
            fac = omega - ai - bi * E  # (nE,1,1)
            W = (
                pref * wE * o["wlam"][None, :, None] * o["wgt"][None]
            )  # quadrature weight per cell
            src_list = []
            for f in ("phi", "bpar"):
                bs = o["basis"][f]  # (nb, nlam, ncell)
                if f == "phi":
                    chi = J0[None] * bs[:, None]
                else:
                    chi = vl * Ti * (mu * I1)[None] * bs[:, None]
                src_list.append(
                    (f, fac[None] * chi)
                )  # S = (omega - omega_*^T) chi : (nb, nE, nlam, ncell)
            if self.ion_psi:
                # A_par source (omega - omega_*)(i/omega) J0 v_par d_l psi = c i J0 dpsi/dt (c = (omega - omega_*)/omega):
                # g = g' + c J0 psi with g' driven by S' = -(omega - omega_di) c J0 psi (J0 taken constant along the orbit
                # in the integration by parts)
                u = J0[None] * o["basis"]["psi"][:, None]
                c = (fac / omega)[None]
                src_list.append(("psi", -(omk - wd)[None] * c * u))
            for f, Ssrc in src_list:
                for kp, (s_, c_) in enumerate(iparts if f == "phi" else [(0.0, 1.0)]):
                    wds = wd - s_ * o["wbeta"][None] * E if s_ != 0 else wd
                    if vT > 0:
                        xs = 1j * (omk - wds) * dt if s_ != 0 else x
                        gbar = self._integrate(-1j * Ssrc, xs, dt, o["trapped"])
                    else:
                        gbar = (
                            Ssrc / (omk - wds)[None]
                        )  # local limit (both sigma identical)
                        gbar = 2 * gbar
                    if f == "psi":
                        gbar = gbar + 2 * c * u  # the local part c J0 psi, both sigma
                    gbar = c_ * gbar
                    # moments projected on each test basis
                    for mom, wfun in (("H", J0), ("P", mu * I1), ("D", J0 * wds)):
                        q = W[None] * wfun[None] * gbar  # (nb_src, nE, nlam, ncell)
                        for tf in ("phi", "bpar", "psi"):
                            tb = np.conj(o["basis"][tf])  # (nb_t, nlam, ncell)
                            val = np.einsum("tlc,selc->ts", tb, q)
                            key = (mom, tf, f)
                            out[key] = out.get(key, 0) + val
                    # local source moment L = (1/T_i) int F0 (omega - omega_*) J0 chi  (both sigma: factor 2);
                    # the A_par part of chi is odd in v_par and gives no L.  Once per source (the precession/direct-drive parts'
                    # coefficients sum to 1: h[a r] + d h[0] - d h[r])
                    if kp > 0:
                        continue
                    q = 2 * W[None] * J0[None] * Ssrc * (0 if f == "psi" else 1)
                    for tf in ("phi", "bpar", "psi"):
                        tb = np.conj(o["basis"][tf])
                        val = np.einsum("tlc,selc->ts", tb, q)
                        key = ("L", tf, f)
                        out[key] = out.get(key, 0) + val
            if split:
                outs.append(out)
        return outs if split else out

    def flux_moments(self, omega, fields):
        """Velocity moments of the ion h for one eigenmode, for the quasilinear fluxes.

        fields: dict(phi, bpar, psi) of basis coefficients (the eigenvector, same normalisation as
        the fields whose fluxes are wanted) and dapar: callable z -> A_par(z) (A_par = -(i/omega)
        d_l psi).  Returns dict of complex numbers, k = 0 (particles) and 1 (energy, E = v^2/v_Ti^2):
            P[k] = int J dtheta int d3v conj(h) J0 E^k phi
            B[k] = int J dtheta int d3v conj(h) mu I1 E^k dB_par
            A[k] = int J dtheta int d3v conj(h) v_par J0 E^k A_par     (v_par in c_s units)
        h is integrated along the orbits exactly as in moments() (same quadrature)."""
        S = self.S
        omega = complex(omega)
        assert not any(getattr(S, "dec", {}).values()), "not for the a_s/d_s split"
        Ti = self.Ti
        vT = np.sqrt(2 * Ti)
        vl = S.sw["vl_i"]
        ai, bi = S.ai, S.bi
        pref = (1 / Ti) * np.pi**-1.5 * (np.pi / 2)
        out = dict(
            P=np.zeros(2, complex), B=np.zeros(2, complex), A=np.zeros(2, complex)
        )
        for o in self.orbits:
            E = self.E[:, None, None]
            wE = self.wE[:, None, None]
            lam = o["lam"][None, :, None]
            if self.bessel:
                xx = np.sqrt(2 * o["bB"][None] * E)
                J0 = besselj0(xx)
                I1 = np.where(
                    xx > 1e-8, 2 * besselj1(xx) / np.where(xx > 1e-8, xx, 1), 1.0
                )
            else:
                J0 = np.exp(-0.5 * o["bB"][None] * E)
                I1 = np.exp(-0.25 * o["bB"][None] * E)
            mu = lam * E
            wd = o["wdrift"][None] * E
            dt = o["dl_over_v"][None] / (self.vT * np.sqrt(E))
            omk = omega
            if o["trapped"] and getattr(S, "ion_nu", None) is not None:
                omk = omega + 1j * S.ion_nu(self.E)[:, None, None]
            x = 1j * (omk - wd) * dt
            fac = omega - ai - bi * E
            bas = o["basis"]
            phi = np.einsum("n,nlc->lc", fields["phi"], bas["phi"])[None]
            bpar = np.einsum("n,nlc->lc", fields["bpar"], bas["bpar"])[None]
            chi = J0 * phi + vl * Ti * mu * I1 * bpar
            gf, gr = self._integrate(
                (-1j * fac * chi)[None], x, dt, o["trapped"], split=True
            )
            gf, gr = gf[0], gr[0]
            if self.ion_psi:
                u = J0 * np.einsum("n,nlc->lc", fields["psi"], bas["psi"])[None]
                c = fac / omega
                sp = -(omk - wd) * c * u
                hf, hr = self._integrate((-1j * sp)[None], x, dt, o["trapped"], True)
                gf = gf + hf[0] + c * u
                gr = gr + hr[0] + c * u
            W = pref * wE * o["wlam"][None, :, None] * o["wgt"][None]
            gs = np.conj(gf + gr)
            go = np.conj(gf - gr)
            vpar = vT * np.sqrt(E * np.clip(1 - o["lB"][None], 0, None))
            apar = fields["dapar"](o["z"])[None]
            for k in (0, 1):
                Ek = E**k
                out["P"][k] += np.sum(W * gs * J0 * Ek * phi)
                out["B"][k] += np.sum(W * gs * mu * I1 * Ek * bpar)
                out["A"][k] += np.sum(W * go * vpar * J0 * Ek * apar)
        return out

    @staticmethod
    def _integrate(s, x, dt, trapped, split=False):
        """Cell-averaged g, summed over both sigma, for dg/dt = s + a g (x = a dt per cell), s: (nb, nE, nlam, ncell).
        Passing: g = 0 entering, sigma = +1 runs over cells 0..n-1, sigma = -1 over n-1..0.
        Trapped: periodic over the bounce orbit (forward cells then backward cells).
        split=True returns the two directions separately, (g[sigma = +1], g[sigma = -1]).
        """
        ex = np.exp(x)  # |ex| <= 1 for Im omega > 0
        p1 = _phi1(x)
        p2 = _phi2(x)
        sdt = s * dt[None]
        n = x.shape[-1]

        def sweep(order, g0):
            g = g0
            gb = np.empty(s.shape, complex)
            for k in order:
                gb[..., k] = g * p1[..., k] + sdt[..., k] * p2[..., k]
                g = g * ex[..., k] + sdt[..., k] * p1[..., k]
            return gb, g

        fwd = range(n)
        bwd = range(n - 1, -1, -1)
        z0 = np.zeros(s.shape[:-1], complex)
        if not trapped:
            gf, _ = sweep(fwd, z0)
            gr, _ = sweep(bwd, z0)
            return (gf, gr) if split else gf + gr
        # trapped: one bounce = forward then backward; periodic solution g0 = C/(1 - prod ex)
        gf, g1 = sweep(fwd, z0)
        gr, g2 = sweep(bwd, g1)
        Ptot = np.prod(ex, axis=-1) ** 2
        g0 = g2 / (1 - Ptot)
        # add the homogeneous part g0 * (cumulative product) to the cell averages
        cf = np.concatenate(
            [np.ones(ex.shape[:-1] + (1,), complex), np.cumprod(ex, axis=-1)[..., :-1]],
            axis=-1,
        )
        Pf = np.prod(ex, axis=-1)
        cb = (
            Pf[..., None]
            * np.concatenate(
                [
                    np.ones(ex.shape[:-1] + (1,), complex),
                    np.cumprod(ex[..., ::-1], axis=-1)[..., :-1],
                ],
                axis=-1,
            )[..., ::-1]
        )
        gf = gf + g0[..., None] * cf * p1
        gr = gr + g0[..., None] * cb * p1
        return (gf, gr) if split else gf + gr
