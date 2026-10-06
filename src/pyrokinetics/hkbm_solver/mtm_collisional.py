"""Experimental C&S-shaped MTM with a collisional kinetic-electron response.

This is NOT Chandran & Schekochihin's collisionless scalar dispersion with
omega shifted by i*nu. We return to their electron equation (2.25), start with
A_parallel proportional to B/k_perp**2 (optionally relax with nA>1), and close
hot-ion quasineutrality and Galerkin projections of Ampere's law.
Both passing and trapped electrons are
resolved on (theta, xi=v_parallel/v, E=v**2/vte**2). A Lorentz electron-ion
operator couples pitch angles, including the trapped/passing boundary.

With g=-h_e/F0e, the equations in the package's electron reference units are
  [S - i(omega-omega_d) - C] g = -i(omega-omega_star) J0 (phi-v_parallel A),
  (1+1/tau) phi = <J0 g>,
  <A, k_perp**2 A - beta/2 <v_parallel J0 g>>_J = 0.
S = v/(JB) [xi d_theta -(1-xi**2)/2 (d_theta ln B) d_xi].
C = nu_D(E)/2 d_xi[(1-xi**2)d_xi] - nu_D(E) a0**2(1+xi**2)/4,
where a0=k_perp*v/|Omega_e|. The last term is the gyroaveraged Lorentz
finite-Larmor-radius correction, not a phenomenological damping rate.

No electron-electron energy diffusion/field-particle terms, kinetic ions or
delta B_parallel. The prescribed A shape and hot-ion closure need validation
at finite collisionality. In particular, this is not GENE's Sugama operator.
The zero-collision limit is this pre-asymptotic fixed-A model, not an algebraic
identity with the original C&S asymptotic dispersion. Unresolved != stable.
"""

import time
import warnings

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legder, legval, legvander
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.special import j0, roots_genlaguerre, roots_legendre

from .exact import ExtendedGeometry
from .solver import nu_ei_gene


def pitch_operators(nxi):
    """Gauss-Legendre nodal differentiation and unit-rate Lorentz operator.

    C P_l = -l(l+1)/2 P_l. The quadrature-weighted operator is self-adjoint,
    conserves the angular average and dissipates every non-isotropic harmonic.
    """
    if int(nxi) != nxi or nxi < 4 or nxi % 2:
        raise ValueError("nxi must be an even integer >= 4")
    xi, weights = roots_legendre(nxi)
    ell = np.arange(nxi)
    V = legvander(xi, nxi - 1)
    inverse = ((2 * ell + 1)[:, None] / 2) * V.T * weights
    deriv = (
        np.column_stack([legval(xi, legder(np.eye(nxi)[degree])) for degree in ell])
        @ inverse
    )
    collision = (V * (-ell * (ell + 1) / 2)) @ inverse
    return xi, weights, deriv, collision


def theta_derivatives(n, spacing, order=2):
    """Upwind derivatives; incoming endpoint rows are replaced.

    At the first interior point use the first-order incoming stencil. Both
    signs are reflected copies, preserving tearing parity exactly. Interior
    upwind truncation is a numerical error, not a physical collision term.
    """
    if n < 5 or spacing <= 0 or order not in (2, 3):
        raise ValueError("need at least five theta nodes and positive spacing")
    rows, cols, vals = [], [], []
    for k in range(1, n):
        if k == 1:
            offsets, coefficients = (0, 1), (1.0, -1.0)
        elif order == 3 and k < n - 1:
            # Third-order biased stencil, positive Fourier-symbol real part.
            offsets, coefficients = (2, 1, 0, -1), (1 / 6, -1.0, 0.5, 1 / 3)
        else:
            offsets, coefficients = (0, 1, 2), (1.5, -2.0, 0.5)
        for offset, coefficient in zip(offsets, coefficients):
            rows.append(k)
            cols.append(k - offset)
            vals.append(coefficient / spacing)
    positive = csr_matrix((vals, (rows, cols)), shape=(n, n))
    reverse = eye(n, format="csr")[::-1]
    negative = -reverse @ positive @ reverse
    return positive, negative


class CollisionalMTMSolver:
    """Experimental fixed-A, hot-ion electron-ion Lorentz MTM eigenproblem.

    Units: omega in c_s/L_ref, ky in 1/rho_s, electron-direction Re(omega)>0.
    ``from_deck`` converts ky and coll; results retain internal units and also
    expose omega_gene_deck/gamma_deck when constructed from a Deck.
    ``frequency_model='constant'`` is a labelled diagnostic; default is the
    physical electron-ion deflection rate nu_ei/E**1.5.
    """

    def __init__(
        self,
        geo,
        params,
        ky,
        *,
        coll=0.0,
        npt=32,
        nturns=4,
        nE=12,
        nxi=16,
        frequency_model="coulomb",
        collision_flr=True,
        drift_sign=1.0,
        bp=1.0,
        theta_order=2,
        nA=1,
        a_width=1.0,
    ):
        self.p, self.geo = dict(params), geo
        self.ky, self.coll = float(ky), float(coll)
        self.npt, self.nturns = int(npt), int(nturns)
        self.nE, self.nxi = int(nE), int(nxi)
        self.frequency_model = frequency_model
        self.collision_flr = bool(collision_flr)
        self.drift_sign = float(drift_sign)
        self.bp = float(bp)
        self.theta_order, self.nA, self.a_width = (
            int(theta_order),
            int(nA),
            float(a_width),
        )
        if theta_order not in (2, 3) or nA != self.nA or nA < 1 or a_width <= 0:
            raise ValueError("theta_order=2 or 3, integer nA>=1 and a_width>0 required")
        if (
            not np.isfinite([ky, coll, drift_sign, bp, a_width]).all()
            or ky <= 0
            or coll < 0
        ):
            raise ValueError("finite ky>0, coll>=0 and drift_sign required")
        if npt != self.npt or npt < 8 or npt % 2 or nturns != self.nturns or nturns < 0:
            raise ValueError("even npt>=8 and integer nturns>=0 required")
        if nE != self.nE or nE < 2:
            raise ValueError("integer nE>=2 required")
        if frequency_model not in ("coulomb", "constant"):
            raise ValueError("frequency_model must be 'coulomb' or 'constant'")
        if any(not np.isfinite(self.p[k]) for k in ("me", "Ti", "beta", "omn", "omte")):
            raise ValueError("finite plasma parameters required")
        self.me, self.tau, self.beta = (float(self.p[k]) for k in ("me", "Ti", "beta"))
        if min(self.me, self.tau, self.beta) <= 0:
            raise ValueError("me, Ti and beta must be positive")
        self.vte = np.sqrt(2 / self.me)
        self.nu_ei = nu_ei_gene(self.coll, me=self.me)
        self.xi, self.wxi, Dxi, Cxi = pitch_operators(nxi)
        self.E, wE = roots_genlaguerre(nE, 0.5)
        self.wE = wE * (2 / np.sqrt(np.pi))
        self.nu = self.nu_ei * (
            self.E**-1.5 if frequency_model == "coulomb" else np.ones(nE)
        )
        self.extended = ExtendedGeometry(geo, nturns=nturns)
        if not self.extended.updown:
            raise ValueError("currently requires up-down symmetric geometry, theta0=0")
        extent = np.pi * (2 * nturns + 1)
        self.theta = np.linspace(-extent, extent, (2 * nturns + 1) * npt + 1)
        self.N = self.theta.size
        self.mid = self.N // 2
        self.dtheta = 2 * np.pi / npt
        self.B = self.extended.sB(self.theta)
        self.J = self.extended.sJ(self.theta)
        Ky = self.extended.sKy(self.theta)
        Kc = Ky - geo.dpdx / (2 * self.B)
        Kg = Kc + self.bp * geo.dpdx / (2 * self.B)
        self.kperp2 = ky**2 * self.extended.sgyy(self.theta)
        if min(self.B.min(), self.J.min(), self.kperp2.min()) <= 0:
            raise ValueError("positive B, Jacobian and kperp2 required")
        self.A = self.B / self.kperp2
        self.A /= self.A[self.mid]
        self.wtheta = self.J * self.dtheta
        self.wtheta[[0, -1]] *= 0.5
        self.ampere_norm = np.sum(self.wtheta * self.kperp2 * self.A**2)
        # nA=1 is precisely the prescribed C&S shape. Extra even rational
        # Chebyshev functions relax it while preserving the theta^-2 tail.
        # Weighted Gram-Schmidt makes the vacuum Ampere matrix the identity.
        self.A_basis = []
        mapped = self.theta / np.sqrt(self.theta**2 + a_width**2)
        metric = self.wtheta * self.kperp2
        for m in range(nA):
            coefficients = np.zeros(2 * m + 1)
            coefficients[-1] = 1.0
            basis = self.A * chebval(mapped, coefficients)
            for _ in range(2):
                for previous in self.A_basis:
                    basis -= (
                        np.sum(metric * previous * basis) / self.ampere_norm * previous
                    )
            norm = np.sum(metric * basis**2)
            if norm < 1e-20 * self.ampere_norm:
                raise ValueError(
                    "A basis is unresolved/linearly dependent on this grid"
                )
            self.A_basis.append(basis * np.sqrt(self.ampere_norm / norm))
        self.A_basis = np.asarray(self.A_basis)
        self.incoming = np.r_[
            np.flatnonzero(self.xi > 0),
            (self.N - 1) * nxi + np.flatnonzero(self.xi < 0),
        ]
        self.qn_coefficient = 1 + 1 / self.tau
        Dpos, Dneg = theta_derivatives(self.N, self.dtheta, theta_order)
        stream = kron(Dpos, diags(np.maximum(self.xi, 0)))
        stream += kron(Dneg, diags(np.minimum(self.xi, 0)))
        stream = diags(np.repeat(1 / (self.J * self.B), nxi)) @ stream
        dlnB = self.extended.sB(self.theta, 1) / self.B
        mirror = ((1 - self.xi**2)[:, None] / 2) * Dxi
        stream -= kron(diags(dlnB / (self.J * self.B)), csr_matrix(mirror))
        self.stream = stream.tocsc()
        self.lorentz = kron(eye(self.N), csr_matrix(Cxi)).tocsc()
        self.drift = (
            -ky
            * drift_sign
            / self.B[:, None]
            * ((1 - self.xi**2) * Kg[:, None] + 2 * self.xi**2 * Kc[:, None])
        )
        self.bases, self.j0, self.vpar, self.sources = [], [], [], []
        for E, nu in zip(self.E, self.nu):
            velocity = self.vte * np.sqrt(E) * self.xi
            a02 = 2 * self.me * E * self.kperp2 / self.B**2
            gyro = j0(np.sqrt(a02[:, None] * (1 - self.xi**2)))
            flr_loss = (
                nu * a02[:, None] * (1 + self.xi**2) / 4
                if collision_flr
                else np.zeros_like(gyro)
            )
            base = self.vte * np.sqrt(E) * self.stream - nu * self.lorentz
            base += diags((1j * E * self.drift + flr_loss).ravel())
            self.bases.append(base.tocsc())
            self.j0.append(gyro)
            self.vpar.append(velocity)
            self.sources.append(ky * (self.p["omn"] + self.p["omte"] * (E - 1.5)))
        self._cached_omega = None
        self._phi_previous = None
        self._basis_previous = [None] * self.nA

    @classmethod
    def from_deck(cls, deck, ky_ref, **options):
        options.setdefault("coll", deck.solver_kw["coll"])
        options.setdefault("bp", deck.solver_kw.get("bp_e", 1.0))
        warnings.warn(
            "Experimental C&S-shaped Lorentz-ei MTM: Boltzmann ions, no Bpar, "
            "no ee collisions; not a Sugama-equivalent or validated MTM backend.",
            UserWarning,
            stacklevel=2,
        )
        result = cls(
            deck.geo, deck.params, ky_ref * deck.units["rho_s_over_rho_ref"], **options
        )
        result.deck_units = dict(deck.units)
        return result

    def factor(self, omega):
        omega = complex(omega)
        if not np.isfinite([omega.real, omega.imag]).all() or omega.imag <= 0:
            raise ValueError("only finite growing-half-plane omega is supported")
        if self._cached_omega == omega:
            return
        factors = []
        for base in self.bases:
            matrix = (base - 1j * omega * eye(base.shape[0])).tolil()
            for row in self.incoming:
                matrix.rows[row] = [int(row)]
                matrix.data[row] = [1.0 + 0j]
            factors.append(splu(matrix.tocsc()))
        self._factors, self._cached_omega = factors, omega

    def response(self, phi, amplitude=0.0, *, kinetic_residual=False, apar=None):
        """Density/current moments of g with incoming g=0, at the last omega."""
        density = np.zeros(self.N, complex)
        current = np.zeros(self.N, complex)
        boundary, residual = 0.0, 0.0
        omega = self._cached_omega
        apar = amplitude * self.A if apar is None else np.asarray(apar)
        for k, (lu, gyro, vp, ws, weight) in enumerate(
            zip(self._factors, self.j0, self.vpar, self.sources, self.wE)
        ):
            rhs = (
                -1j
                * (omega - ws)
                * gyro
                * (np.asarray(phi)[:, None] - apar[:, None] * vp)
            ).ravel()
            rhs[self.incoming] = 0
            g = lu.solve(rhs).reshape(self.N, self.nxi)
            density += weight * ((gyro * g) @ (self.wxi / 2))
            current += weight * ((gyro * g * vp) @ (self.wxi / 2))
            if kinetic_residual:
                applied = self.bases[k] @ g.ravel() - 1j * omega * g.ravel()
                applied[self.incoming] = g.ravel()[self.incoming]
                residual = max(
                    residual,
                    float(
                        np.linalg.norm(applied - rhs) / max(np.linalg.norm(rhs), 1e-30)
                    ),
                )
                edge = np.linalg.norm(g[[0, -1]], axis=1).max()
                peak = np.linalg.norm(g, axis=1).max()
                boundary = max(boundary, float(edge / max(peak, 1e-30)))
        return density, current, residual, boundary

    def _expand(self, half):
        """Odd phi, with centre and endpoint values fixed to zero."""
        phi = np.zeros(self.N, complex)
        phi[self.mid + 1 : -1] = half
        phi[1 : self.mid] = -np.asarray(half)[::-1]
        return phi

    def _quasineutrality(self, apar, previous, qn_tol):
        """Linear phi response to one trial A; factorisation already cached."""
        sl = slice(self.mid + 1, -1)
        rhs, _, _, _ = self.response(np.zeros(self.N), apar=apar)

        def matvec(half):
            phi = self._expand(half)
            density, _, _, _ = self.response(phi)
            return (self.qn_coefficient * phi - density)[sl]

        operator = LinearOperator(
            (self.mid - 1, self.mid - 1), matvec=matvec, dtype=complex
        )
        kwargs = dict(x0=previous, atol=0.0, restart=60, maxiter=10)
        try:
            half, info = gmres(operator, rhs[sl], rtol=qn_tol, **kwargs)
        except TypeError:  # SciPy < 1.12
            half, info = gmres(operator, rhs[sl], tol=qn_tol, **kwargs)
        phi = self._expand(half)
        density, current, kinetic_error, edge_g = self.response(
            phi, apar=apar, kinetic_residual=True
        )
        qn = self.qn_coefficient * phi - density
        qn_error = float(
            np.linalg.norm(qn[1:-1])
            / max(
                np.linalg.norm(self.qn_coefficient * phi[1:-1]),
                np.linalg.norm(density[1:-1]),
                1e-30,
            )
        )
        if kinetic_error > 1e-7:
            raise RuntimeError(
                f"kinetic response unresolved: residual {kinetic_error:g}"
            )
        if info != 0 or qn_error > max(1e-7, 10 * qn_tol):
            raise RuntimeError(
                f"quasineutrality unresolved: GMRES {info}, residual {qn_error:g}"
            )
        return phi, current, half.copy()

    def evaluate(self, omega, *, qn_tol=1e-9):
        """QN elimination followed by the A-basis Ampere Schur complement.

        The first A coefficient is fixed to 1. Additional Ampere projections
        determine the other coefficients; the remaining projection is D(omega).
        nA=1 recovers the prescribed C&S-shape projection exactly.
        """
        self.factor(omega)
        phis, currents = [], []
        for m, basis in enumerate(self.A_basis):
            phi, current, previous = self._quasineutrality(
                basis, self._basis_previous[m], qn_tol
            )
            self._basis_previous[m] = previous
            phis.append(phi)
            currents.append(current)
        responses = self.kperp2 * self.A_basis - self.beta / 2 * np.asarray(currents)
        matrix = (self.A_basis * self.wtheta) @ responses.T / self.ampere_norm
        coefficients = np.ones(self.nA, complex)
        if self.nA > 1:
            coefficients[1:] = np.linalg.solve(matrix[1:, 1:], -matrix[1:, 0])
        phi = coefficients @ np.asarray(phis)
        apar = coefficients @ self.A_basis
        if not np.isfinite(phi).all() or not np.isfinite(apar).all():
            raise FloatingPointError("non-finite field response")
        density, current, kinetic_error, edge_g = self.response(
            phi, apar=apar, kinetic_residual=True
        )
        qn = self.qn_coefficient * phi - density
        qn_error = float(
            np.linalg.norm(qn[1:-1])
            / max(
                np.linalg.norm((self.qn_coefficient * phi)[1:-1]),
                np.linalg.norm(density[1:-1]),
                1e-30,
            )
        )
        if max(kinetic_error, qn_error) > 1e-7:
            raise RuntimeError("combined kinetic/QN residual exceeds tolerance")
        ampere = self.kperp2 * apar - self.beta / 2 * current
        dispersion = (matrix @ coefficients)[0]
        scale_phi = max(float(np.abs(phi).max()), 1e-30)
        # Measure the last interior turn, not the imposed zero endpoint.
        edge_phi = float(
            np.max(np.abs(phi[np.abs(self.theta) >= abs(self.theta[-1]) - 2 * np.pi]))
            / scale_phi
        )
        self.last = dict(
            omega=complex(omega),
            phi=phi,
            apar=apar,
            current=current,
            apar_coefficients=coefficients,
            dispersion=complex(dispersion),
            qn_residual=qn_error,
            kinetic_residual=kinetic_error,
            edge_phi=edge_phi,
            edge_g=edge_g,
            ampere_shape_residual=float(
                np.linalg.norm(ampere)
                / max(
                    np.linalg.norm(self.kperp2 * apar),
                    np.linalg.norm(self.beta / 2 * current),
                    1e-30,
                )
            ),
        )
        core = abs(self.theta) <= np.pi
        self.last["ampere_core_residual"] = float(
            np.linalg.norm(ampere[core])
            / max(
                np.linalg.norm((self.kperp2 * apar)[core]),
                np.linalg.norm((self.beta / 2 * current)[core]),
                1e-30,
            )
        )
        self.last["ampere_midplane_residual"] = complex(
            ampere[self.mid] / max(abs(self.kperp2[self.mid] * apar[self.mid]), 1e-30)
        )
        return complex(dispersion)

    def solve(self, omega0=None, *, maxit=30, timeout=120.0, tol=1e-7, verbose=False):
        """Secant search with independent QN/kinetic/projected-Ampere checks.

        Only a positive-growth root can be accepted; a failed search is
        unresolved, never a stability verdict. Timeout is checked between
        evaluations, so one sparse factorisation/QN solve may overrun it.
        """
        start = time.monotonic()
        if maxit < 1 or int(maxit) != maxit or timeout <= 0 or tol <= 0:
            raise ValueError("positive integer maxit, timeout and tolerance required")
        wscale = max(abs(self.ky * (self.p["omn"] + 0.5 * self.p["omte"])), 1e-3)
        w0 = complex(omega0) if omega0 is not None else wscale * (1 + 0.05j)
        if w0.imag <= 0:
            raise ValueError("root seed must have positive imaginary part")
        w1 = w0 + 0.01 * wscale * (1 + 1j)
        history, status, converged, floored = [], "unresolved", False, 0
        f0, f1 = self.evaluate(w0), self.evaluate(w1)
        for it in range(maxit):
            history.append(dict(omega=[w1.real, w1.imag], residual=abs(f1)))
            if verbose:
                print(f"MTM Lorentz: {it} omega={w1:.8g} |D|={abs(f1):.3g}", flush=True)
            if abs(f1) < tol and abs(w1 - w0) < tol * max(wscale, abs(w1)):
                converged, status = True, "growing_root"
                break
            if time.monotonic() - start > timeout:
                status = "timeout"
                break
            if abs(f1 - f0) < 1e-30:
                break
            step = -f1 * (w1 - w0) / (f1 - f0)
            if abs(step) > 2 * wscale:
                step *= 2 * wscale / abs(step)
            w2 = w1 + step
            if w2.imag <= 1e-7 * wscale:
                w2 = complex(w2.real, max(0.25 * w1.imag, 1e-7 * wscale))
                floored += 1
                if floored >= 8:
                    break
            w0, f0 = w1, f1
            w1, f1 = w2, self.evaluate(w2)
        r = dict(self.last)
        r.update(
            status=status,
            converged=converged,
            gamma=w1.imag,
            omega_gene=-w1.real,
            relative_residual=abs(f1),
            seconds=time.monotonic() - start,
            history=history,
            theta=self.theta.copy(),
            collision_model="lorentz_ei",
            collision_parameter=self.coll,
            frequency_model=self.frequency_model,
            collision_flr=self.collision_flr,
            model="cs_shaped_hot_ion" if self.nA == 1 else "cs_basis_hot_ion",
            drift_sign=self.drift_sign,
            bp=self.bp,
            nA=self.nA,
            a_width=self.a_width,
            theta_order=self.theta_order,
            resolution=dict(npt=self.npt, nturns=self.nturns, nE=self.nE, nxi=self.nxi),
            collision_physics_match=False,
            validated=False,
            parity="tearing",
        )
        r["domain_warning"] = bool(r["edge_phi"] >= 0.01 or r["edge_g"] >= 0.01)
        r["apar_shape_warning"] = bool(r["ampere_core_residual"] >= 0.1)
        r["resolution_converged"] = False  # requires a separate refinement study
        if hasattr(self, "deck_units"):
            cs = self.deck_units["c_s_over_c_ref"]
            r.update(omega_gene_deck=r["omega_gene"] * cs, gamma_deck=r["gamma"] * cs)
        return r
