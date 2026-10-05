"""
Energy integrals of the hKBM dispersion relation: the plasma dispersion function and the
resonant Maxwellian moments M_n(omega, w) = int_0^inf E^(n+1/2) e^-E/(omega - w E) dE with
their Landau continuation (Zocco, Rodriguez & Edmiston, J. Plasma Phys. 92, E39 (2026),
App. B), and the plane-wave streaming surrogate Mint_kpar.
"""

import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import wofz

SQPI = np.sqrt(np.pi)


# ------------------------------------------------------------------ plasma dispersion function, energy integrals
def Zfun(z):
    """Plasma dispersion function Z(z) = i sqrt(pi) w(z); entire (analytic continuation built in)."""
    return 1j * SQPI * wofz(z)


def zeta_of(omega, w):
    """zeta = sqrt(omega/w) on the branch that gives (i) the real-E integral for Im omega > 0 (Im zeta > 0) and
    (ii) its Landau continuation from Im omega > 0 across the real omega axis at Re omega (Zocco App. B:
    'branch cut placed on the positive real line, Im zeta -> +inf when Im omega -> +inf').
    """
    omega = complex(omega)
    if omega.imag == 0:
        omega += 1e-14j  # exactly real omega: take the limit from the upper half plane
    s = omega / w
    z = np.sqrt(s + 0j)  # principal: cut on the negative real s axis
    sgn = np.where(w > 0, 1.0, -1.0)
    flip = (omega.imag < 0) & (
        s.real < 0
    )  # path from Re omega + i0 to omega crosses the negative real s axis
    return z * sgn * np.where(flip, -1.0, 1.0)


def Mint(n, omega, w):
    """M_n(omega, w) = int_0^inf E^(n+1/2) e^-E / (omega - w E) dE, n = 0, 1, 2 (w real array, omega complex),
    = -P_{n+1}(zeta)/(w zeta) with P_m the Z-type moments (Zocco B10-B13); series when w -> 0.
    """
    w = np.asarray(w, float)
    out = np.empty(w.shape, complex)
    small = np.abs(w) < 1e-4 * abs(omega)
    wb = np.where(small, 1.0, w)
    z = zeta_of(omega, wb)
    out[:] = -Pmom(n + 1, z, np.zeros(z.shape, bool)) / (wb * z)
    if small.any():  # 1/(omega - wE) = sum_k w^k E^k / omega^(k+1)
        ws = w[small]
        ser = sum((ws / omega) ** k * Gamma(n + k + 1.5) for k in range(6)) / omega
        out[small] = ser
    return out


def Pmom(m, z, lower):
    """P_m(z) = int_-inf^inf u^(2m) e^-u^2 /(u - z) du as a real-line integral: sqrt(pi) Z(z) for Im z > 0;
    for a root that sits in the LOWER half plane on the physical side (lower=True) the real-line value is
    sqrt(pi)[Z(z) - 2 i sqrt(pi) e^-z^2] = -sqrt(pi) Z(-z); both expressions are then continued analytically.
    """
    # Z(z) - 2 i sqrt(pi) e^{-z^2} = -Z(-z): no exponential, no overflow for large |Im z|
    P0 = SQPI * np.where(lower, -Zfun(-z), Zfun(z))
    P = P0
    for k in range(1, m + 1):
        P = z * Gamma(k - 0.5) + z**2 * P
    # large |z|: the recursion cancels catastrophically (terms ~ z^(2m+1) against a result ~ 1/z); use the
    # asymptotic series -sum_k Gamma(m + k + 1/2) z^-(2k+1) (the pole term 2 i sqrt(pi) e^-z^2 is negligible there)
    big = np.abs(z) > 30
    if np.any(big):
        zb = np.where(big, z, 1.0)
        Pa = -sum(Gamma(m + k + 0.5) * zb ** (-(2 * k + 1)) for k in range(6))
        P = np.where(big, Pa, P)
    return P


def Mint_kpar(n, omega, w, a):
    """(1/2) sum_sigma int_0^inf E^(n+1/2) e^-E / (omega - w E - sigma a sqrt(E)) dE  (a = k_par v_T sqrt(1-lam B) >= 0)
    = int_-inf^inf u^(2n+2) e^-u^2 / (omega - w u^2 - a u) du = -[P_{n+1}(u1) - P_{n+1}(u2)] / (w (u1 - u2)),
    u_{1,2} = (-a +- sqrt(a^2 + 4 w omega))/(2w).  Exact (real-E) for Im omega > 0; below the real axis the bounded
    real-E value is returned, NOT the Landau continuation (see the comment in the code): use only gamma >= 0.
    """
    omega = complex(omega)
    if omega.imag == 0:
        omega += 1e-14j
    w = np.asarray(w, float)
    a = np.asarray(a, float)
    small = np.abs(w) < 1e-6 * abs(omega)
    wb = np.where(small, 1e-6 * abs(omega) * np.sign(w + (w == 0)), w)
    disc = a**2 + 4 * wb * omega
    sq = np.sqrt(disc + 0j)
    u1 = (-a + sq) / (2 * wb)
    u2 = (-a - sq) / (2 * wb)
    # half plane of each root on the physical side: Im u_j = Im omega / (2 w u_j + a) at Im omega -> 0+
    # (2 w u1 + a = +sq, 2 w u2 + a = -sq); for Im omega < 0 the same flags are kept (continuation)
    if omega.imag > 0:
        low1 = u1.imag < 0
        low2 = u2.imag < 0  # exact real-E integral (flags = actual half planes)
    else:
        # Im omega < 0: the Landau continuation of the streaming resonance for v_par -> 0 particles moves the
        # root by ~ Im omega/(k_par v_par) and e^{-u^2} grows like e^{+(Im u)^2} (e^{+86} at gamma = -0.1 for
        # lam B -> 1): that is the plane-wave k_par surrogate breaking down for trapped-ion-like phase space.
        # We therefore do NOT continue: the bounded real-E value is used below the axis (non-analytic there),
        # so with ion_model='kpar' only roots with gamma >= 0 are meaningful.
        low1 = u1.imag < 0
        low2 = u2.imag < 0
    out = -(Pmom(n + 1, u1, low1) - Pmom(n + 1, u2, low2)) / (wb * (u1 - u2))
    return out
