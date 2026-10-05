"""
Field-line geometry for the hKBM solver: the arrays of GENE's miller.dat (B, Jacobian, K_y,
g^yy, R) on GENE's z grid, splined periodically onto the solver's theta grid (the central
ballooning turn, z = chi in [-pi, pi)).

GENE conventions (prefactors.F90 get_curv, dgdxy_terms.F90, geometry.F90 set_curvature):
K_y = (dBdx - ga3/ga1 dBdz)/C_xy, ga1 = gxx gyy - gxy^2, ga3 = gxy gyz - gyy gxz; the magnetic
drift is omega_d = k_y (T/q) [(mu B + 2 v_par^2)/B K_y - v_par^2/B^2 dpdx] ('full_drift'),
dpdx = dpdx_pm/C_xy the pressure term of the drift (GENE's my_dpdx).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .miller import read_miller_dat


class Geo:
    """GENE z-grid geometry (miller.dat columns + header), with periodic splines of B, K_y, J.

    Build it from a miller.dat file (``Geo.from_file``) or from the dict returned by
    ``miller.miller_from_namelist`` (``Geo.from_miller``)."""

    def __init__(self, header, cols):
        self.h = dict(header)
        n = cols["B"].size
        self.z = -np.pi + 2 * np.pi * np.arange(n) / n
        gxx, gxy, gxz, gyy, gyz = (
            cols["gxx"],
            cols["gxy"],
            cols["gxz"],
            cols["gyy"],
            cols["gyz"],
        )
        self.B, dBdx, dBdz, self.J = (
            cols["B"],
            cols["dBdx"],
            cols["dBdz"],
            cols["jacobian"],
        )
        self.R = cols["R"]
        ga1 = gxx * gyy - gxy**2
        ga2 = gxx * gyz - gxy * gxz
        ga3 = gxy * gyz - gyy * gxz
        cxy = self.h.get("Cxy", 1.0)
        self.Ky = (dBdx - ga3 / ga1 * dBdz) / cxy
        # GENE's K_x = -(dBdy + ga2/ga1 dBdz)/C_xy (dBdy = 0 in a local equilibrium)
        self.Kx = -(ga2 / ga1 * dBdz) / cxy
        self.theta0 = 0.0
        self.dpdx = (
            self.h.get("my_dpdx", 0.0) / cxy
        )  # absent when dpdx_pm = 0 (no pressure term)
        self.gxx, self.gxy, self.gyy = gxx, gxy, gyy
        zp = np.append(self.z, np.pi)

        def sp(f):
            return CubicSpline(zp, np.append(f, f[0]), bc_type="periodic")

        self.sB, self.sK, self.sJ = sp(self.B), sp(self.Ky), sp(self.J)
        zz = np.linspace(-np.pi, np.pi, 20001)
        Bz = self.sB(zz)
        self.Bmax, self.Bmin = Bz.max(), Bz.min()
        self.zmin = zz[np.argmin(Bz)]

    @property
    def kx_per_theta0(self):
        """d(k_x/k_y)/d(theta0): the mean slope of the secular part of g^xy/g^xx over the turn
        (= C_y q0 shat/r0 in GENE's Miller coordinates), so that k_x = -k_y kx_per_theta0 theta0
        puts the minimum of k_perp^2 at theta = theta0 (ballooning angle theta0 = k_x/(shat k_y)).
        """
        r = self.gxy / self.gxx
        return float((r[-1] - r[0]) / (self.z[-1] - self.z[0]))

    def shifted(self, theta0):
        """This geometry for a mode with ballooning angle theta0 (GENE kx_center != 0): the
        solver sees k_y times effective K_y and g^yy,
            K_y -> K_y + (k_x/k_y) K_x,   g^yy -> g^yy + 2 (k_x/k_y) g^xy + (k_x/k_y)^2 g^xx,
        with k_x/k_y = -kx_per_theta0 theta0 (GENE's omega_d ~ K_x k_x + K_y k_y and
        k_perp^2 = g^xx k_x^2 + 2 g^xy k_x k_y + g^yy k_y^2 on the central turn)."""
        import copy

        if theta0 == 0:
            return self
        g = copy.copy(self)
        kap = -self.kx_per_theta0 * float(theta0)
        g.Ky = self.Ky + kap * self.Kx
        g.gyy = self.gyy + 2 * kap * self.gxy + kap**2 * self.gxx
        zp = np.append(self.z, np.pi)
        g.sK = CubicSpline(zp, np.append(g.Ky, g.Ky[0]), bc_type="periodic")
        g.theta0 = float(theta0)
        g.kx_over_ky = kap
        return g

    @classmethod
    def from_file(cls, path):
        h, cols = read_miller_dat(path)
        return cls(h, cols)

    @classmethod
    def from_miller(cls, g):
        """g: dict from miller.miller_from_namelist (columns in L_ref units; header as GENE writes it)."""
        lf = g.get("Lref", 0.0) or 1.0
        cols = {
            k: np.asarray(g[k], float)
            for k in (
                "gxx",
                "gxy",
                "gxz",
                "gyy",
                "gyz",
                "gzz",
                "B",
                "dBdx",
                "dBdy",
                "dBdz",
                "jacobian",
                "R",
                "Z",
            )
        }
        cols["R"] = cols["R"] * lf
        cols["Z"] = cols["Z"] * lf
        h = dict(
            q0=g["q0"],
            shat=g["shat"],
            s0=g["x0"] ** 2,
            minor_r=g.get("minor_r", 1.0),
            major_R=g["major_R"],
            trpeps=g["trpeps"],
            beta=g.get("beta", 0.0),
            Cy=g["C_y"],
            Cxy=g["C_xy"],
            gridpoints=float(g["z"].size),
        )
        if g.get("my_dpdx", 0.0) != 0.0:
            h["my_dpdx"] = g["my_dpdx"]
        return cls(h, cols)


class Geometry:
    """Central ballooning turn theta in [-pi, pi) (nth points) of a Geo object: B, J, K_y, g^yy,
    the pressure term dpdx and the curvature coefficient K_c = K_y - dpdx/(2B)."""

    def __init__(self, geo, nth=1024):
        if isinstance(geo, str):
            geo = Geo.from_file(geo)
        self.g = g = geo
        self.theta = th = -np.pi + 2 * np.pi * np.arange(nth) / nth
        self.dth = 2 * np.pi / nth
        zp = np.append(g.z, np.pi)

        def sp(f):
            return CubicSpline(zp, np.append(f, f[0]), bc_type="periodic")

        self.sgyy = sp(g.gyy)
        self.B, self.J, self.Ky, self.gyy = g.sB(th), g.sJ(th), g.sK(th), self.sgyy(th)
        self.dpdx = g.dpdx
        self.Kc = self.Ky - self.dpdx / (2 * self.B)  # GENE curvature drift coefficient
        self.Bmax, self.Bmin = g.Bmax, g.Bmin

    def bounce_points(self, lam):
        """lam = mu/E; trapped iff 1/Bmax < lam < 1/Bmin. Returns (z1, z2) around the outboard well."""
        g = self.g

        def f(z):
            return 1 - lam * g.sB(z)

        z0 = g.zmin

        def edge(sgn):
            zs = z0 + sgn * np.linspace(0, np.pi, 4001)
            v = f(zs)
            i = np.argmax(v <= 0)
            if v[i] > 0:
                return z0 + sgn * np.pi
            return (
                brentq(f, zs[i - 1], zs[i]) if sgn > 0 else brentq(f, zs[i], zs[i - 1])
            )

        return edge(-1), edge(+1)
