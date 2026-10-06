"""Legacy reduced collisions projected onto the full-GK orbit response.

This is NOT a full gyrokinetic/Sugama collision operator. Passing particles
remain collisionless. Trapped ions get the old energy-dependent Krook loss.
The old bounce-averaged electron Lorentz operator acts on

    H = bounce_average(G - (1 - omega_star/omega) psi),
    d_l psi = i omega A_parallel, psi(left boundary) = 0.

The subtraction matters: the old model scatters H, not the whole G. The
electron update is a low-rank, self-consistent response correction; it is not
an eigenvalue damping shift. With R the collisionless orbit resolvent,
U = <R source>, h = <R 1>, Q = <(1-wstar/w) psi>, and L=-C/nu >= 0,

    (I + i nu diag(h) L) H = U - Q,
    delta G = R 1 (-i nu L H).

Supported initially for twisting parity on single-well periodic geometry.
The old H=0 passing-boundary closure is not a validated MTM collision model.
"""

import numpy as np

from .geometry import Geometry
from .solver import Solver as ReducedSolver
from .solver import nu_ei_gene, nuD_e, nuD_i


def periodic_probes(x, dt, rows, cols, interpolation):
    """Bounce-averaged source response and response to a constant unit drive.

    Leading axes are (energy, well), last axis is the directed orbit segment.
    ``cols`` includes the -i(omega-wstar) source coefficient; row factors include
    the field moment weights. ``interpolation[f]`` has shape (2, nsegment, ndof).
    Returns h (energy,well), and lists U/V of (energy,well,ndof), with no energy sum.
    Uses the same linear-in-time elements as ExactSolver's collisionless response.
    """
    from .exact import _phis

    p1, p2, p3, _ = _phis(x)
    phase = np.concatenate(
        [np.zeros(x.shape[:-1] + (1,), complex), np.cumsum(x, -1)], -1
    )
    nseg = x.shape[-1]
    lower = np.arange(nseg)[:, None] > np.arange(nseg)[None, :]
    exponent = phase[..., :-1, None] - phase[..., None, 1:]
    exponent += np.where(lower, 0, phase[..., -1, None, None])
    propagator = np.exp(exponent) / (-np.expm1(phase[..., -1, None, None]))
    rho = dt / dt.sum(-1, keepdims=True)
    incoming_unit = np.einsum("...ij,...j->...i", propagator, -1j * dt * p1)
    h = np.sum(rho * (p1 * incoming_unit - 1j * dt * p2), -1)
    averaged_propagator = np.einsum("...i,...ij->...j", rho * p1, propagator)
    U, V = [], []
    for row, col, W in zip(rows, cols, interpolation):
        ua = col * dt * (averaged_propagator * (p1 - p2) + rho * (p2 - p3))
        ub = col * dt * (averaged_propagator * p2 + rho * p3)
        va = row * (p2 * incoming_unit - 1j * dt * p3)
        vb = row * ((p1 - p2) * incoming_unit - 1j * dt * (p2 - p3))
        U.append(ua @ W[0] + ub @ W[1])
        V.append(va @ W[0] + vb @ W[1])
    return h, U, V


class LegacyCollisions:
    """Reuse the reduced solver's Lorentz matrix and collision frequencies."""

    def __init__(self, solver, coll, ee=True, ions=True, eps=None):
        self.solver, self.coll, self.ions = solver, float(coll), bool(ions)
        species = solver.species
        electrons = [s for s in species if s.Z < 0]
        ion_species = [s for s in species if s.Z > 0]
        if (
            len(species) != 2
            or len(electrons) != 1
            or len(ion_species) != 1
            or electrons[0].Z != -1
            or ion_species[0].Z != 1
        ):
            raise ValueError(
                "legacy collisions require one singly charged ion and electrons"
            )
        electron, ion = electrons[0], ion_species[0]
        if electron.T != 1 or electron.n != 1 or ion.m != 1 or ion.n != 1:
            raise ValueError(
                "legacy collisions require the reduced solver's e/i reference units"
            )
        if not solver.symmetric or not solver.trapped:
            raise ValueError(
                "legacy collisions require symmetric, trapped-particle geometry"
            )
        if len(solver.trp) != len(solver.lt):
            raise ValueError(
                "legacy collisions currently require one magnetic well per period"
            )
        turns = solver.trp[0]["turns"]
        if any(not np.array_equal(t["turns"], turns) for t in solver.trp):
            raise ValueError(
                "legacy collisions require complete wells at every trapped pitch"
            )
        # Reuse the operator exactly, on this solver's pitch grid. No duplicated
        # deflection-rate formula or subtly different trapped-boundary condition.
        reduced = ReducedSolver.__new__(ReducedSolver)
        reduced.geo = Geometry(solver.geo, nth=128)
        reduced.p = dict(me=electron.m, Ti=ion.T)
        reduced.coll, reduced.coll_model = coll, "lorentz"
        reduced.coll_ee, reduced.coll_i, reduced.coll_eps = ee, ions, eps
        reduced.nE_e, reduced.lam = 96, solver.lt
        reduced._setup_collisions()
        self.lo, self.diag, self.up = reduced.L_lo, reduced.L_dg, reduced.L_up
        self.eps = reduced.eps_coll
        self.nu = nuD_e(solver.E, nu_ei_gene(coll, me=electron.m), ee=ee)
        self.kappa0 = reduced.kappa0
        # Primitive of cell-constant A_parallel, zero at the left boundary.
        self.integral = (
            np.arange(solver.N)[None, :] < np.arange(solver.N + 1)[:, None]
        ) * solver.VB
        self.batches = {}

    def ion_rate(self, species):
        if not self.ions or species.Z < 0:
            return np.zeros_like(self.solver.E)
        return (
            nuD_i(
                self.solver.E,
                self.coll,
                species.T,
                mi=species.m,
                Z=species.Z,
                ni=species.n,
            )
            / self.eps
        )

    def begin(self):
        self.batches = {}

    def collect(
        self, T, keep, x, dt, rows, cols, W, Wnode, global_indices, rpos, omega, species
    ):
        """Collect per-pitch probes, then couple all pitches within each well."""
        solver = self.solver
        h, U, V = periodic_probes(x, dt, rows, cols, W)
        turns = T["turns"][keep]
        rho = dt[0] / dt[0].sum(-1, keepdims=True)
        node_average = rho @ ((Wnode[0] + Wnode[1]) / 2)
        ws = (
            -(species.T / species.Z)
            * solver.ky
            * (species.omn + species.omt * (solver.E - 1.5))
        )
        nrow = int(np.sum(rpos >= 0))
        for iw, turn in enumerate(turns):
            if int(turn) not in self.batches:
                self.batches[int(turn)] = dict(
                    h=np.zeros((solver.E.size, solver.lt.size), complex),
                    U=np.zeros((solver.E.size, solver.lt.size, solver.nunk), complex),
                    V=np.zeros((solver.E.size, nrow, solver.lt.size), complex),
                    present=np.zeros(solver.lt.size, bool),
                )
            batch = self.batches[int(turn)]
            il = T["il"]
            batch["present"][il] = True
            batch["h"][:, il] = h[:, iw]
            for f, uf, vf, gi in zip(solver.fields, U, V, global_indices):
                batch["U"][:, il, gi[iw]] += uf[:, iw]
                pr = rpos[gi[iw]]
                use = pr >= 0
                batch["V"][:, pr[use], il] += vf[:, iw, use]
            if "apar" in solver.fields:
                nodes = T["k1"][keep][iw] + np.arange(T["Mc"] + 1)
                primitive_average = node_average[iw] @ self.integral[nodes]
                cols_a = solver.off["apar"] + np.arange(solver.N)
                batch["U"][:, il, cols_a] -= (
                    1j * (omega - ws)[:, None] * primitive_average[None, :]
                )

    def apply(self, matrix):
        """Add V[-i nu L (I+i nu h L)^-1 (U-Q)] to the field matrix."""
        for batch in self.batches.values():
            if not batch["present"].all():
                raise ValueError("incomplete pitch set in a collisional well")
            h, U, V = batch["h"], batch["U"], batch["V"]
            z = 1j * self.nu[:, None] * h
            H = ReducedSolver._tridiag(
                -z * self.lo, 1 + z * self.diag, -z * self.up, U.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            force = self.diag[None, :, None] * H
            force[:, 1:] -= self.lo[None, 1:, None] * H[:, :-1]
            force[:, :-1] -= self.up[None, :-1, None] * H[:, 1:]
            force *= -1j * self.nu[:, None, None]
            active_rows = np.flatnonzero(np.any(V != 0, axis=(0, 2)))
            active_cols = np.flatnonzero(np.any(force != 0, axis=(0, 1)))
            correction = np.einsum(
                "erl,elc->rc",
                V[:, active_rows],
                force[:, :, active_cols],
                optimize=True,
            )
            matrix[np.ix_(active_rows, active_cols)] += correction
