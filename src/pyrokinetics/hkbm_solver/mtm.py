"""
Microtearing (tearing-parity) branch: the collisionless gyrokinetic MTM dispersion relation of
Chandran & Schekochihin, J. Plasma Phys. (2024), arXiv:2211.02103 (C&S below), evaluated in the
package's GENE geometry (Miller/MXH, extended over many ballooning turns).

Model (C&S section 2, k_perp rho_e << beta_e, |omega| ~ omega_*e, no delta B_par):
  * Ampere at theta ~ 1: A_par = C B / k_perp^2 (C&S 2.37); it enters only through
    psi_inf = (i omega/2) int J B A_par dtheta (2.43; solver units, see below).
  * dispersion relation (2.39)
        omega - omega_0 + i sqrt(pi) [v_Te/L + omega^2 B_max/(2 v_Te) int dtheta J Gamma dPhi] = 0,
    omega_0 = omega_*e (1 + eta_e/2), L = int J B^2 dtheta / (B_max k_perp^2 d_e^2) (2.40),
    Gamma(theta) the passing-electron integral (2.41);
  * dPhi = delta Phi / psi_inf from quasineutrality at |theta| >> 1 (2.46): Boltzmann ions with
    tau = T_i/T_e (the "hot-ion" form), passing-electron kernel W_p (2.47) and, optionally, the
    trapped-electron kernel W_tr (2.48, A8).  The integral equation is solved by GMRES; the W_p
    product uses the propagator recursion (O(N n_v) per product, stable for Im omega > 0).
  * root: secant iteration on the dispersion relation.
Collisions (not in C&S; option, default off): a Krook term in the electron response with the
energy-dependent electron deflection frequency, omega -> omega + i coll_scale nu_D^e(E) in the
propagators (time-of-flight phase and bounce average), the drive (omega - Omega_*e) unchanged,
as in the drift-kinetic microtearing literature (Drake & Lee 1977; Gladd et al. 1980).

Units/signs as solver.py: T_ref = T_e, n_ref = n_e, m_ref = m_i, frequencies in c_s/L_ref, k_y in
1/rho_s, exp(-i omega t) ("model sign"; omega_r > 0 = electron direction, omega_gene = -Re omega).
In these units v_Te = sqrt(2/m_e), rho_e = sqrt(2 m_e) rho_s at B_ref, d_e^2 = rho_e^2/beta_e,
dl = J B dtheta (GENE's Jacobian), A_par in rho_s B_ref, phi in T_e/e.  Electron drift and
diamagnetic frequencies as in ion_stream.py / solver.py with the electron charge:
    omega_De = -k_y [lam B K_y + 2 (1 - lam B) K_c] E / B,  K_c = K_y - dpdx/(2B),
    Omega_*e(E) = k_y [omn + omte (E - 3/2)],   E = v^2/v_Te^2, lam = mu/E (lam B = sin^2 pitch).
Turn j of the extended ballooning angle theta = z + 2 pi j is the central turn of a mode with
ballooning angle theta0 - 2 pi j (geometry.Geo.shifted): K_y + kappa K_x, g^yy + 2 kappa g^xy +
kappa^2 g^xx, kappa = -kx_per_theta0 (theta0 - 2 pi j).

    from pyrokinetics.hkbm_solver import mtm
    r = mtm.solve(0.285, params=dict(PARAMS, beta=0.14))      # STEP geometry
    r = mtm.solve_deck(Deck("run/parameters"), ky_ref)        # a GENE deck (deck units out)
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import j0 as besselj0
from scipy.special import roots_genlaguerre, roots_legendre

SQPI = np.sqrt(np.pi)


class MTMSolver:
    """The C&S microtearing dispersion function D(omega) at one k_y.

    geo      geometry.Geo (GENE z grid, nz divisible by 2 npt)
    params   dict(Ti, omn, omte, beta, me) in solver units (beta = beta_e at B_ref)
    ky       k_y rho_s;  theta0 ballooning angle
    npt      coarse theta points per 2 pi (dPhi, Gamma, W); the phases use the full GENE z grid
    extent   theta range = +-extent / (k_y rho_e s_k), s_k = d sqrt(g^yy)/dtheta at large theta
             (the width of dPhi is ~ 1/(k_perp rho_e)); or nturns (turns each side) directly
    nE, nlam, nlam_tr  energy (generalised Gauss-Laguerre), passing and trapped pitch nodes
    trapped  include the trapped-electron kernel W_tr (C&S A8)
    coll     GENE's coll in solver units (as Solver; nu_ei = 4 coll/sqrt(m_e)); coll_scale
             multiplies the Krook rate; ee: include e-e deflection in nu_D^e
    phi_term False drops the dPhi term (C&S: then Im omega < 0; a test)
    drift_sign  multiplies omega_De everywhere (diagnostic only: -1 reverses the magnetic drift)
    radial_drift  multiplies the secular (local-shear, K_x) part of omega_De, kx_per_theta0
                  (theta - theta0) K_x, in K_y (diagnostic)
    """

    def __init__(
        self,
        geo,
        params,
        ky,
        theta0=0.0,
        npt=32,
        extent=6.0,
        nturns=None,
        nE=24,
        nlam=24,
        nlam_tr=16,
        trapped=True,
        coll=0.0,
        coll_scale=1.0,
        ee=True,
        phi_term=True,
        drift_sign=1.0,
        radial_drift=1.0,
    ):
        self.geo, self.p, self.ky = geo, dict(params), float(ky)
        self.drift_sign = float(drift_sign)
        self.radial_drift = float(radial_drift)
        self.theta0 = float(theta0)
        self.npt, self.trapped, self.phi_term = int(npt), bool(trapped), phi_term
        p = self.p
        self.me, self.tau, self.beta = p["me"], p["Ti"], p["beta"]
        self.vte = np.sqrt(2.0 / self.me)
        self.rhoe = np.sqrt(2.0 * self.me)
        self.omega0 = self.ky * (p["omn"] + 0.5 * p["omte"])
        self.coll = float(coll)
        if self.coll > 0:
            from .solver import nu_ei_gene

            self.nu_ei = nu_ei_gene(self.coll, me=self.me)
        else:
            self.nu_ei = 0.0
        self.coll_scale, self.ee = float(coll_scale), ee
        self._turns(extent, nturns)
        self._grid()
        self._velocity(nE, nlam, nlam_tr)

    # ------------------------------------------------------------------ geometry
    def _turn_geometry(self, j):
        g = self.geo
        kap = -g.kx_per_theta0 * (self.theta0 - 2 * np.pi * j)
        if self.radial_drift == 1.0:
            Ky = g.Ky + kap * g.Kx
        else:  # periodic part + radial_drift x the secular (local-shear) part kx_per_theta0 (theta - theta0) K_x
            sec = g.kx_per_theta0 * g.z * g.Kx
            Ky = g.Ky - sec + self.radial_drift * (sec + kap * g.Kx)
        gyy = g.gyy + 2 * kap * g.gxy + kap**2 * g.gxx
        return Ky, gyy, kap

    def _turns(self, extent, nturns):
        if nturns is None:
            _, gyy1, _ = self._turn_geometry(1)
            _, gyy2, _ = self._turn_geometry(2)
            sk = (np.sqrt(gyy2.mean()) - np.sqrt(gyy1.mean())) / (2 * np.pi)
            sk = max(sk, 1e-3)
            thmax = extent / (self.ky * self.rhoe * sk)
            nturns = int(np.ceil(thmax / (2 * np.pi)))
            nturns = int(min(max(nturns, 3), 400))
        self.nturns = nturns

    def _grid(self):
        g = self.geo
        nz = g.z.size
        if nz % (2 * self.npt):
            raise ValueError(f"geometry nz = {nz} must be a multiple of 2 npt")
        nsub = nz // self.npt
        js = np.arange(-self.nturns, self.nturns + 1)
        th, B, J, Ky, gyy, Kx, kap = [], [], [], [], [], [], []
        for j in js:
            Kyj, gyyj, kp = self._turn_geometry(j)
            th.append(g.z + 2 * np.pi * j)
            B.append(g.B)
            J.append(g.J)
            Ky.append(Kyj)
            gyy.append(gyyj)
            Kx.append(g.Kx)
            kap.append(np.full(nz, kp))
        f = dict(
            theta=np.concatenate(th),
            B=np.concatenate(B),
            J=np.concatenate(J),
            Ky=np.concatenate(Ky),
            gyy=np.concatenate(gyy),
            Kx=np.concatenate(Kx),
            kap=np.concatenate(kap),
            turn=np.repeat(js, nz),
        )
        f["Kc"] = f["Ky"] - g.dpdx / (2 * f["B"])
        self.dz = 2 * np.pi / nz
        self.fine = f
        # coarse points: cell centres of npt cells per turn (theta = 0 is never a node)
        self.ic = np.arange(nsub // 2, f["theta"].size, nsub)
        self.dth = 2 * np.pi / self.npt
        self.c = {k: v[self.ic] for k, v in f.items()}
        self.i0 = int(np.argmin(np.abs(f["theta"])))  # theta = 0 on the fine grid
        self.Bmax = g.Bmax
        self.kperp2_f = self.ky**2 * f["gyy"]
        # Ampere: v_Te/L,  L = int J B^2 dtheta / (B_max k_perp^2 d_e^2), d_e^2 = rho_e^2/beta_e
        de2 = self.rhoe**2 / self.beta
        L = np.sum(f["J"] * f["B"] ** 2 / (self.kperp2_f * de2)) * self.dz / self.Bmax
        self.L = L
        self.vte_over_L = self.vte / L

    def _cumint(self, y):
        """int_0^theta y dz on the fine grid (trapezoid), zero at theta = 0 (the grid point i0)."""
        c = np.concatenate(
            [np.zeros((1,) + y.shape[1:]), np.cumsum(0.5 * (y[1:] + y[:-1]), 0)]
        )
        c *= self.dz
        return c - c[self.i0]

    def _velocity(self, nE, nlam, nlam_tr):
        f, c = self.fine, self.c
        E, wE = roots_genlaguerre(nE, 0.5)  # weight E^1/2 e^-E
        self.E, self.wE = E, wE
        t, wt = roots_legendre(nlam)
        t, wt = 0.5 * (t + 1), 0.5 * wt
        lam = (1 - t**2) / self.Bmax
        wlam = wt * 2 * t / self.Bmax
        self.lam, self.wlam = lam, wlam
        lB = lam[None, :] * f["B"][:, None]  # (Nf, nl)
        sq = np.sqrt(1 - lB)
        wd_hat = -self.ky * (lB * f["Ky"][:, None] + 2 * (1 - lB) * f["Kc"][:, None])
        wd_hat *= self.drift_sign / f["B"][:, None]  # omega_De / E
        JB = (f["J"] * f["B"])[:, None]
        self.tof = self._cumint(JB / sq)[
            self.ic
        ]  # (N, nl): v_Te sqrt(E) x time of flight
        self.dfl = self._cumint(JB * wd_hat / sq)[
            self.ic
        ]  # drift phase / E (x v_Te sqrt E)
        lBc = lam[None, :] * c["B"][:, None]
        self.sqc = np.sqrt(1 - lBc)
        al2 = 2 * self.me * self.kperp2_f[self.ic][:, None, None] * E[None, None, :]
        al2 = al2 * lam[None, :, None] / c["B"][:, None, None]
        self.J0c = besselj0(np.sqrt(al2))  # (N, nl, nE)
        self.nu = self._nu(E)
        if self.trapped:
            self._trapped_setup(nlam_tr)

    def _nu(self, E):
        if self.nu_ei <= 0:
            return np.zeros_like(E)
        from .solver import nuD_e

        return self.coll_scale * nuD_e(E, self.nu_ei, ee=self.ee)

    def _trapped_setup(self, nlam_tr):
        """Trapped orbits, one well per turn around the B minimum (B and J are periodic, so the
        well is the same index range in every turn; K_y, g^yy and the K_x phase change with the
        turn): coarse nodes inside the well, bounce time, bounce-averaged drift (per E) and the
        radial-drift phase P (C&S A5: the K_x part of omega_De integrated along the orbit from the
        bounce point; P = sqrt(E) prad / v_Te)."""
        f, g = self.fine, self.geo
        x, wx = roots_legendre(nlam_tr)
        lmin, lmax = 1 / g.Bmax, 1 / g.Bmin
        lam = lmin + (lmax - lmin) * 0.5 * (x + 1)
        wlam = wx * 0.5 * (lmax - lmin)
        nz = g.z.size
        nt = 2 * self.nturns + 1
        nsub = nz // self.npt
        icl = np.arange(
            nsub // 2, nz, nsub
        )  # coarse nodes within a turn (local fine index)
        B1 = g.B
        k0 = int(np.argmin(B1))
        sh = lambda a: a.reshape(nt, nz)  # noqa: E731
        Bt, Kyt, Kct = sh(f["B"]), sh(f["Ky"]), sh(f["Kc"])
        kapt, Kxt, kp2t = sh(f["kap"]), sh(f["Kx"]), sh(self.kperp2_f)
        orbits = []
        for il, lm in enumerate(lam):
            inside = lm * B1 < 1
            a = k0
            while a - 1 >= 0 and inside[a - 1]:
                a -= 1
            b = k0
            while b + 1 < nz and inside[b + 1]:
                b += 1
            sl = slice(a, b + 1)
            kc = np.nonzero((icl >= a) & (icl <= b))[0]
            if kc.size == 0:
                continue
            loc = icl[kc] - a
            sq = np.sqrt(1 - lm * B1[sl])
            JB = (g.J * g.B)[sl]
            lB = lm * Bt[:, sl]
            ds = -self.ky * self.drift_sign
            wd = ds * (lB * Kyt[:, sl] + 2 * (1 - lB) * Kct[:, sl]) / Bt[:, sl]
            wrad = (
                ds * self.radial_drift * kapt[:, sl] * Kxt[:, sl] * (2 - lB) / Bt[:, sl]
            )
            dt = JB / sq * self.dz
            tb = dt.sum()
            wdb = (dt * wd).sum(1) / tb  # (nt,)
            prad = np.cumsum(dt * wrad, 1) - 0.5 * dt * wrad  # (nt, nwell)
            orbits.append(
                dict(
                    il=il,
                    k=np.arange(nt)[:, None] * self.npt
                    + kc[None, :],  # (nt, nk) coarse
                    sq=sq[loc],
                    tb=tb,
                    wdb=wdb,
                    prad=prad[:, loc],
                    B=B1[icl[kc]],
                    J=g.J[icl[kc]],
                    kp2=kp2t[:, icl[kc]],
                )
            )
        self.tr_lam, self.tr_wlam, self.orbits = lam, wlam, orbits

    # ------------------------------------------------------------------ kernels at omega
    def _phases(self, omega):
        """I_v(theta_k) (N, nl, nE): ((omega + i nu(E)) tof - E dfl) / (v_Te sqrt E)."""
        E = self.E
        w = omega + 1j * self.nu
        return (w[None, None, :] * self.tof[:, :, None] - E * self.dfl[:, :, None]) / (
            self.vte * np.sqrt(E)
        )[None, None, :]

    def setup(self, omega):
        c = self.c
        E, wE = self.E, self.wE
        I = self._phases(omega)  # noqa: E741
        sg = np.sign(c["theta"])[:, None, None]
        Q = omega - self.ky * (self.p["omn"] + self.p["omte"] * (E - 1.5))  # (nE,)
        base = self.J0c / self.sqc[:, :, None]  # (N, nl, nE)
        wv = self.wlam[:, None] * wE[None, :]  # (nl, nE)
        Gam = np.einsum("kle,le->k", base * np.exp(1j * sg * I), wv * Q / omega) * (
            c["B"] / SQPI
        )
        Gam *= np.sign(c["theta"])
        self.Gamma = Gam
        # passing kernel factors (flattened velocity index v = (l, e))
        N = c["theta"].size
        tauf = self.tau / (1 + self.tau)
        cv = (1j * tauf / (2 * SQPI)) * wv * Q / (self.vte * np.sqrt(E))[None, :]
        self._left = (base * c["B"][:, None, None]).reshape(N, -1) * cv.reshape(1, -1)
        self._right = (base * (c["J"] * c["B"] * self.dth)[:, None, None]).reshape(
            N, -1
        )
        self._blocks(I.reshape(N, -1))
        # trapped kernel: per pitch, arrays over (turn, node, E)
        self._tr = []
        if self.trapped:
            Qe = omega - self.ky * (self.p["omn"] + self.p["omte"] * (E - 1.5))
            sE = np.sqrt(E)
            for o in self.orbits:
                lm = self.tr_lam[o["il"]]
                al2 = (
                    2 * self.me * o["kp2"][:, :, None] * E * lm / o["B"][None, :, None]
                )
                cosP = np.cos(o["prad"][:, :, None] * sE / self.vte)
                h = besselj0(np.sqrt(al2)) * cosP  # (nt, nk, nE)
                avg = omega + 1j * self.nu[None, :] - E[None, :] * o["wdb"][:, None]
                coef = (
                    -(tauf / SQPI) * self.tr_wlam[o["il"]] * wE * Qe / (avg * o["tb"])
                )
                left = h * (o["B"] / o["sq"])[None, :, None] * coef[:, None, :]
                right = h * (o["J"] * o["B"] * self.dth / o["sq"])[None, :, None]
                self._tr.append((o["k"], left, right))
        self._omega = omega

    def _blocks(self, If):
        """Propagator factors for the blocked W_p product: blocks of one turn (npt nodes); inside
        a block the propagator exp(i (I_k - I_k')) is split about the block midpoint (bounded
        factors), between blocks it is carried as F (forward) and H (backward)."""
        N = If.shape[0]
        b = self.npt
        st = np.arange(0, N, b)
        en = np.minimum(st + b, N)
        mid = (st + en) // 2
        bi = np.repeat(np.arange(st.size), np.diff(np.append(st, N)))
        Im = If[mid[bi]]
        self._Ep = np.exp(1j * (If - Im))
        self._Em = np.exp(-1j * (If - Im))
        self._Ts = np.exp(1j * (If - If[st[bi]]))
        nxt = np.minimum(en, N - 1)
        self._Te = np.exp(1j * (If[nxt[bi]] - If))
        self._Bse = np.exp(1j * (If[nxt] - If[st]))
        self._Bme = np.exp(1j * (If[nxt] - If[mid]))
        self._bl = list(zip(st, en))

    def apply_W(self, x):
        """(W_p + W_tr) x on the coarse grid."""
        L, R = self._left, self._right
        rx = R * x[:, None]
        Ep, Em, Ts, Te = self._Ep, self._Em, self._Ts, self._Te
        tot = np.empty(R.shape, complex)
        F = np.zeros(R.shape[1], complex)
        nb = len(self._bl)
        for j, (s, e) in enumerate(self._bl):  # forward: sum over theta' < theta
            q = Em[s:e] * rx[s:e]
            c = np.cumsum(q, 0)
            ex = np.vstack([np.zeros((1, q.shape[1])), c[:-1]])
            tot[s:e] = Ts[s:e] * F + Ep[s:e] * ex
            if j < nb - 1:
                F = self._Bse[j] * F + self._Bme[j] * c[-1]
        H = np.zeros(R.shape[1], complex)
        for j in range(nb - 1, -1, -1):  # backward: sum over theta' > theta
            s, e = self._bl[j]
            q = Ep[s:e] * rx[s:e]
            c = np.cumsum(q[::-1], 0)[::-1]  # sum_{k' >= k}
            ex = np.vstack([c[1:], np.zeros((1, q.shape[1]))])
            G = Em[s:e] * ex
            if j < nb - 1:
                G += Te[s:e] * H
            tot[s:e] += G
            H = rx[s] + G[0]
        tot += rx  # diagonal (theta' = theta)
        out = np.einsum("kv,kv->k", L, tot)
        for k, left, right in self._tr:
            s = np.einsum("tke,tk->te", right, x[k])
            np.add.at(out, k, np.einsum("tke,te->tk", left, s))
        return out

    def solve_phi(self, omega, tol=1e-8):
        """dPhi(theta_k) at omega from (2.46) (GMRES)."""
        self.setup(omega)
        N = self.c["theta"].size
        A = LinearOperator((N, N), matvec=lambda v: v + self.apply_W(v), dtype=complex)
        rhs = self.tau / (1 + self.tau) * self.Gamma
        x0 = getattr(self, "_phi_prev", None)
        x, info = gmres(A, rhs, x0=x0, rtol=tol, atol=0.0, restart=60, maxiter=4)
        self._phi_prev = x
        self.gmres_info = info
        return x

    def D(self, omega):
        """The dispersion function (2.39) at omega (Im omega > 0, or > -min nu with collisions)."""
        omega = complex(omega)
        if self.phi_term:
            ph = self.solve_phi(omega)
            S = np.sum(self.c["J"] * self.Gamma * ph) * self.dth
            self.phi = ph
        else:
            S = 0.0
            self.phi = None
        return (
            omega
            - self.omega0
            + 1j * SQPI * (self.vte_over_L + omega**2 * self.Bmax / (2 * self.vte) * S)
        )

    def cold_ion(self):
        """C&S (2.56-2.57), the tau << 1 limit: omega_r = omega_0 and
        gamma = -sqrt(pi) [v_Te/L + tau omega_0^2 B_max/(2 v_Te) int J Re Gamma^2 |_omega_0].
        """
        self.setup(complex(self.omega0))
        S = np.sum(self.c["J"] * (self.Gamma**2).real) * self.dth
        g = -SQPI * (
            self.vte_over_L + self.tau * self.omega0**2 * self.Bmax / (2 * self.vte) * S
        )
        return complex(self.omega0, g)

    # ------------------------------------------------------------------ root
    def solve(
        self,
        omega0=None,
        tol=1e-7,
        maxit=40,
        verbose=False,
        im_floor=None,
        timeout=None,
    ):
        """Secant iteration on D(omega).  omega0: seed (model sign, c_s/L_ref); default
        omega_0 (1 + 0.1 i).  Iterates are kept above im_floor (default 1e-4 omega_0 - min nu):
        the propagators need Im(omega) + nu > 0; three iterates pinned at the floor stop the
        search (no growing root).  timeout: wall-clock seconds.  Returns a result dict (see
        fields())."""
        t0 = time.time()
        w0 = self.omega0
        if im_floor is None:
            im_floor = 1e-4 * abs(w0) - (self.nu.min() if self.nu_ei > 0 else 0.0)
        x0 = complex(omega0) if omega0 is not None else w0 * (1 + 0.1j)
        x1 = x0 * (1 + 0.02) + 0.01j * abs(w0)
        f0, f1 = self.D(x0), self.D(x1)
        conv, it, floored = False, 0, 0
        for it in range(1, maxit + 1):
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if x2.imag < im_floor:
                x2 = complex(x2.real, im_floor)
                floored += 1
            x0, f0 = x1, f1
            x1, f1 = x2, self.D(x2)
            if verbose:
                print(f"  it {it}: omega {x1:.6g} |D| {abs(f1):.3g}", flush=True)
            if abs(x1 - x0) < tol * abs(w0) and abs(f1) < 1e-5 * abs(w0):
                conv = True
                break
            if floored >= 3:
                break
            if timeout is not None and time.time() - t0 > timeout:
                break
        growing = conv and x1.imag > im_floor * 1.5 and x1.imag > 0
        return self.fields(
            dict(
                omega=x1,
                omega_gene=-x1.real,
                gamma=x1.imag,
                converged=conv,
                growing=bool(growing),
                iters=it,
                residual=abs(f1),
                seconds=time.time() - t0,
                omega0=w0,
            )
        )

    def fields(self, r):
        """Eigenfunctions and validity numbers for the last D() evaluation (A_par = B/k_perp^2
        normalised to A_par(theta=0) = 1 in rho_s B_ref; phi in T_e/e on the same normalisation).
        """
        f, c = self.fine, self.c
        omega = r["omega"]
        A = f["B"] / self.kperp2_f
        A = A / A[self.i0]
        psi_inf = 0.5j * omega * np.sum(f["J"] * f["B"] * A) * self.dz
        phi = self.phi * psi_inf if self.phi is not None else None
        kp2_c = self.kperp2_f[self.ic]
        r["theta"] = c["theta"].copy()
        r["phi"] = phi
        r["apar_theta"], r["apar"] = f["theta"], A
        r["psi_inf"] = psi_inf
        wA = f["J"] * np.abs(A) ** 2
        kA = float(np.sum(wA * self.kperp2_f) / np.sum(wA))
        if phi is not None:
            wp = c["J"] * np.abs(phi) ** 2
            kP = float(np.sum(wp * kp2_c) / np.sum(wp))
            amp = float(np.abs(A).max() / np.abs(phi).max())
            # C_tear-like parity number: |int J B A| / int J B |A| = 1 by construction (A > 0)
        else:
            kP, amp = np.nan, np.nan
        r["kperp2"] = dict(phi=kP, apar=kA)
        r["apar_over_phi"] = amp
        kpe0 = np.sqrt(self.kperp2_f[self.i0]) * self.rhoe
        r["validity"] = dict(
            kperp_rho_e=float(kpe0),
            kperp_rho_e_over_beta_e=float(kpe0 / self.beta),
            nu_ei_over_omega=float(self.nu_ei / abs(self.omega0)),
            beta_e=float(self.beta),
            nturns=self.nturns,
            edge_phi=(
                float(np.abs(phi[[0, -1]]).max() / np.abs(phi).max())
                if phi is not None
                else np.nan
            ),
        )
        r["parity"] = "tearing"
        r["branch"] = "mtm"
        return r


# production resolution (run_linear/solve_deck): passing electrons only.  The trapped kernel
# (C&S A8) has the bounce-averaged precession resonance 1/<omega - omega_De>_b, which near the real
# axis needs far finer pitch/energy grids (STEP k_y 0.1, beta 0.12: gamma +0.006 with 16 trapped
# pitches, -0.0009 with 8); C&S find it changes gamma by 10 % (n = 50) to 0.6 % (n = 400).
PROD = dict(npt=16, nE=16, nlam=16, nlam_tr=12, trapped=False)
LOWRES = dict(npt=16, nE=10, nlam=10, nlam_tr=8)


def find_root(
    geo, params, ky, omega0=None, timeout=20.0, quick=True, verbose=False, **kw
):
    """MTM root at k_y rho_s (solver units) in two stages: a low-resolution secant search
    (LOWRES) from omega0 or omega_0 (1 + 0.1 i) gives the no-root verdict (iterates pinned at the
    real axis: no growing root) or the seed of the production search (PROD updated by kw:
    MTMSolver options).  Returns the result dict with 'stage' and 'no_root_verdict' (True when
    the low-resolution search found no growing root)."""
    t0 = time.time()
    kw = dict(PROD, **kw)
    if quick:
        lk = dict(kw)
        lk.update(LOWRES)
        Ml = MTMSolver(geo, params, ky, **lk)
        rl = Ml.solve(omega0=omega0, maxit=15, timeout=timeout, verbose=verbose)
        if not rl["growing"]:
            rl.update(no_root_verdict=True, stage="low", seconds=time.time() - t0)
            rl["solver"] = Ml
            return rl
        omega0 = rl["omega"]
    M = MTMSolver(geo, params, ky, **kw)
    left = None if timeout is None else max(timeout - (time.time() - t0), 1.0)
    r = M.solve(omega0=omega0, timeout=left, verbose=verbose)
    r.update(no_root_verdict=False, stage="full", seconds=time.time() - t0)
    r["solver"] = M
    return r


def solve(ky, params=None, miller=None, omega0=None, **kw):
    """MTM root at k_y rho_s on the STEP deck geometry (default) or a Geo/miller.dat path."""
    from . import solver as S
    from .geometry import Geo

    geo = (
        S.step_geo()
        if miller is None
        else (miller if isinstance(miller, Geo) else Geo.from_file(miller))
    )
    p = dict(S.PARAMS if params is None else params)
    sk = {k: kw.pop(k) for k in ("tol", "maxit", "verbose") if k in kw}
    M = MTMSolver(geo, p, ky, **kw)
    r = M.solve(omega0=omega0, **sk)
    r["solver"] = M
    return r


def solve_deck(deck, ky_ref, omega0=None, theta0=0.0, coll=0.0, **kw):
    """MTM root for a gene_io.Deck at k_y rho_ref (deck units).  omega0: seed in GENE sign, deck
    units (omega + i gamma).  coll: 0 (default: collisionless, as C&S), "deck" (the deck's coll in the Krook model) or a
    number (GENE coll in solver units).  Returns the solver dict with omega, gamma
    in deck units (GENE sign), ky, ky_rho_s, kperp2 in 1/rho_ref^2."""
    u = deck.units
    rs, cs = u["rho_s_over_rho_ref"], u["c_s_over_c_ref"]
    ky_s = ky_ref * rs
    cl = deck.solver_kw.get("coll", 0.0) if coll == "deck" else float(coll)
    seed = None
    if omega0 is not None:
        w = complex(omega0)
        seed = complex(-w.real, w.imag) / cs
    r = find_root(
        deck.geo, deck.params, ky_s, omega0=seed, theta0=theta0, coll=cl, **kw
    )
    r.update(
        ky=ky_ref,
        ky_rho_s=ky_s,
        theta0=theta0,
        omega_solver=r["omega"],
        gamma_solver=r["gamma"],
        omega=-r["omega"].real * cs,
        gamma=r["gamma"] * cs,
        kperp2_ref={k: v / rs**2 for k, v in r["kperp2"].items()},
    )
    return r


# ---------------------------------------------------------------------------- run_linear branch
MTM_ORDERING_MAX = (
    0.3  # k_perp rho_e / beta_e above which the C&S ordering (2.53) is doubtful
)


def _record(deck, r, ky_ref, theta0, n=None, fields=True):
    """One MTM root in the deck's GENE normalisation, in the layout of quasilinear._record."""
    u = deck.units
    rs = u["rho_s_over_rho_ref"]
    rec = dict(
        ky=ky_ref,
        ky_rho_s=ky_ref * rs,
        n=n,
        theta0=theta0,
        branch="mtm",
        parity="tearing",
        omega=np.nan,
        gamma=np.nan,
        converged=False,
        hkbm_like=False,
        checks=None,
        weights=None,
        weights_solver=None,
        error=r.get("error") if isinstance(r, dict) else None,
        no_root_verdict=bool(r.get("no_root_verdict", False)),
        seconds=r.get("seconds"),
    )
    if "omega" not in r:
        return rec
    v = r["validity"]
    checks = dict(
        converged=bool(r["converged"]),
        growing=bool(r["growing"]),
        electron_direction=bool(r["omega_solver"].real > 0),
        ordering=bool(v["kperp_rho_e_over_beta_e"] < MTM_ORDERING_MAX),
        phi_decays=bool(v["edge_phi"] < 0.1),
    )
    rec["checks"] = checks
    rec["validity"] = v
    if not (checks["converged"] and checks["growing"]):
        return rec
    rec.update(
        omega=float(r["omega"]),
        gamma=float(r["gamma"]),
        gamma_solver=float(r["gamma_solver"]),
        converged=True,
        mtm_like=all(checks.values()),
    )
    kp = r["kperp2"]
    amp = dict(phi=1.0, apar=float(r["apar_over_phi"]))
    rec["kperp2_avg"] = kp["phi"] / rs**2
    rec["giacomin"] = dict(
        kperp2=dict(kp),
        amplitude=amp,
        Lambda_hat=float(r["gamma_solver"]) * sum(amp[f] / kp[f] for f in amp),
        Q_i_over_Q=None,
        Q_e_over_Q=None,
        Gamma_over_Q=None,
    )
    if fields:
        phi = r["phi"]
        k = int(np.argmax(np.abs(phi)))
        p0 = phi[k]
        M = r["solver"]
        rec.update(
            theta=r["theta"].copy(),
            phi=phi / p0,
            apar_theta=r["apar_theta"].copy(),
            apar=r["apar"] * rs**2 / (p0 * u["T_e"] * rs),
            kperp2=M.kperp2_f[M.ic] / rs**2,
            jacobian=M.c["J"].copy(),
            bmag=M.c["B"].copy(),
        )
    return rec


def _warm(warm, ky, theta0):
    if not warm:
        return None
    c = [
        m
        for m in warm
        if m.get("branch") == "mtm"
        and m.get("converged")
        and abs(m.get("theta0", 0.0) - theta0) < 1e-9
    ]
    if not c:
        return None
    m = min(c, key=lambda m: abs(np.log(m["ky"] / ky)))
    if abs(np.log(m["ky"] / ky)) > 0.5:
        return None
    return complex(m["omega"], m["gamma"]) * ky / m["ky"]


def run_linear_mtm(
    source,
    ky=None,
    n=None,
    theta0=(0.0,),
    rho_star=None,
    timeout=None,
    omega0=None,
    fields=True,
    verbose=False,
    warm=None,
    mtm_kw=None,
):
    """MTM roots (C&S) on a (k_y, theta0) grid, records as quasilinear.run_linear's with
    branch 'mtm', parity 'tearing', checks (converged, growing, electron_direction, ordering:
    k_perp rho_e/beta_e < MTM_ORDERING_MAX, phi_decays), validity numbers and the Giacomin block
    (kperp2 phi/apar in rho_s units, amplitude apar/phi; no flux weights).  mtm_kw: options of
    find_root/MTMSolver (e.g. coll='deck', trapped=False)."""
    import tempfile
    import warnings

    from .gene_io import Deck
    from .quasilinear import KY_DEFAULT, _rho_star, _source_deck

    with tempfile.TemporaryDirectory() as tmp:
        path = _source_deck(source, tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deck = Deck(path)
    rs = deck.units["rho_s_over_rho_ref"]
    if n is not None:
        rst = _rho_star(source, deck, rho_star)
        ns = [float(x) for x in np.atleast_1d(n)]
        kys = [x * rst / abs(float(deck.geo.h["Cy"])) for x in ns]
    else:
        kys = (
            [k / rs for k in KY_DEFAULT]
            if ky is None
            else [float(k) for k in np.atleast_1d(ky)]
        )
        ns = [None] * len(kys)
    kw = dict(mtm_kw or {})
    out = []
    for th0 in [float(t) for t in np.atleast_1d(theta0)]:
        prev = None
        for j in np.argsort(kys):
            kyj = kys[j]
            seed = _warm(warm, kyj, th0)
            if seed is None and prev is not None and prev["converged"]:
                seed = complex(prev["omega"], prev["gamma"]) * kyj / prev["ky"]
            if seed is None and omega0 is not None and not out:
                seed = omega0
            t0 = time.time()
            try:
                r = solve_deck(
                    deck,
                    kyj,
                    omega0=seed,
                    theta0=th0,
                    timeout=timeout if timeout is not None else 20.0,
                    **kw,
                )
                r["seconds"] = time.time() - t0
                rec = _record(deck, r, kyj, th0, ns[j], fields)
            except Exception as e:  # reported per mode
                rec = _record(deck, dict(error=repr(e)), kyj, th0, ns[j])
            rec["seconds"] = time.time() - t0
            out.append((th0, j, rec))
            prev = rec
            if verbose:
                print(
                    "  MTM k_y %-8.4g theta0 %-5.3g gamma %+.5f omega %+.5f %s(%.1f s)"
                    % (
                        kyj,
                        th0,
                        rec["gamma"],
                        rec["omega"],
                        "" if rec["converged"] else "no growing root ",
                        rec["seconds"],
                    ),
                    flush=True,
                )
    out.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in out]


def run_linear_modes(source, modes=("hkbm", "mtm"), mtm_kw=None, **kw):
    """quasilinear.run_linear with several branches: the hKBM records (branch 'hkbm', parity
    'twisting') followed by the MTM records (branch 'mtm', parity 'tearing'), each list in
    run_linear's order (theta0 outer, k_y inner).  Each branch is solved independently; the
    caller picks the root it wants (e.g. the most unstable, or the one whose checks pass).
    """
    from .quasilinear import run_linear

    modes = tuple(modes)
    bad = set(modes) - {"hkbm", "mtm"}
    if bad:
        raise ValueError(f"unknown modes {sorted(bad)} (use 'hkbm', 'mtm')")
    mtm_options = dict(mtm_kw or {})
    backend = mtm_options.pop("backend", "cs")
    if backend == "lorentz_ei":
        from .mtm_collisional_ql import run_linear_mtm as mtm_runner
    elif backend == "cs":
        mtm_runner = run_linear_mtm
    else:
        raise ValueError("MTM backend must be 'cs' or experimental 'lorentz_ei'")
    out = []
    if "hkbm" in modes:
        hkbm_kw = dict(kw)
        if backend == "lorentz_ei" and hkbm_kw.get("warm"):
            hkbm_kw["warm"] = [
                m for m in hkbm_kw["warm"] if m.get("branch", "hkbm") == "hkbm"
            ]
        for m in run_linear(source, modes=("hkbm",), **hkbm_kw):
            m.setdefault("branch", "hkbm")
            m.setdefault("parity", "twisting")
            out.append(m)
    if "mtm" in modes:
        mk = {
            k: kw[k]
            for k in (
                "ky",
                "n",
                "theta0",
                "rho_star",
                "timeout",
                "fields",
                "verbose",
                "warm",
            )
            if k in kw
        }
        out += mtm_runner(source, mtm_kw=mtm_options, **mk)
    return out
