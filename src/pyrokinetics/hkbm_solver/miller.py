"""
GENE's local Miller field-line geometry in Python.

A line-by-line port of GENE's ``miller_geometry.F90`` (subroutine ``get_miller``, case
``magn_geometry = 'miller'``; Miller et al., Phys. Plasmas 5, 973 (1998); Candy, PPCF 51,
105009 (2009)), together with the parts of ``geometry.F90`` that decide the pressure term
(``amhd``, ``dpdx_pm``) and the curvature coefficient written by ``set_curvature``.  It
returns the same arrays GENE writes to ``miller.dat`` (16 columns on GENE's z grid) and can
write that file in GENE's format.

Grids and numerics are GENE's own: 500 (n_pol + 2) points in the poloidal angle, 500
(n_pol + 1) points in the straight-field-line angle, third-order Lagrange interpolation
(``lag_interp.F90``), second-order finite differences, the trapezoidal running integrals of
``dlp_int_ind`` and the rescaling of chi.  The result agrees with GENE's ``miller.dat`` to
round-off (see ``tests/hkbm_solver/test_miller.py``).

Normalisation (GENE): lengths in L_ref, B in B_ref (B0 = 1 at R = major_R), z = chi in
[-pi, pi), x = r (dx/dr = 1), C_y = dPsi/dr sign(Ip), C_xy = |B0 dPsi/dr / C_y| = 1.

Only the up-down symmetric Miller parametrisation with elongation, triangularity and
squareness (kappa, delta, zeta and their shears, drR, drZ) is implemented; n_pol = 1 and
edge_opt = 0 (GENE's defaults).
"""

import numpy as np

COLUMNS = (
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
    "phi",
    "Z",
    "dxdR",
    "dxdZ",
)


# --------------------------------------------------------------------------- GENE numerics
def _linspace(lo, hi, n):
    """GENE's linspace: out(i) = min + (i - 1) (max - min)/(n - 1), i = 1..n."""
    i = np.arange(n, dtype=float)
    return lo + i * (hi - lo) / (n - 1)


def lag3interp(y_in, x_in, x_out):
    """Third-order Lagrange interpolation, a literal port of GENE's ``lag3interp_array_dp``
    (lag_interp.F90), including its stencil search and its extrapolation at both ends.
    """
    y_in = np.asarray(y_in, float)
    x_in = np.asarray(x_in, float)
    x_out = np.asarray(x_out, float)
    n_in, n_out = x_in.size, x_out.size
    # 1-based indices as in the Fortran source
    if x_in[n_in - 1] > x_in[0]:
        jstart, jstep = 3, 1
    else:
        jstart, jstep = n_in - 2, -1
    if x_out[n_out - 1] > x_out[0]:
        order = range(1, n_out + 1)
    else:
        order = range(n_out, 0, -1)
    y_out = np.empty(n_out)
    j1 = jstart
    for j in order:
        x = x_out[j - 1]
        while x >= x_in[j1 - 1] and j1 < n_in - 1 and j1 > 2:
            j1 += jstep
        j2 = j1 + jstep
        j0 = j1 - jstep
        jm = j1 - 2 * jstep
        x2, x1, x0, xm = x_in[j2 - 1], x_in[j1 - 1], x_in[j0 - 1], x_in[jm - 1]
        aintm = (x - x0) * (x - x1) * (x - x2) / ((xm - x0) * (xm - x1) * (xm - x2))
        aint0 = (x - xm) * (x - x1) * (x - x2) / ((x0 - xm) * (x0 - x1) * (x0 - x2))
        aint1 = (x - xm) * (x - x0) * (x - x2) / ((x1 - xm) * (x1 - x0) * (x1 - x2))
        aint2 = (x - xm) * (x - x0) * (x - x1) / ((x2 - xm) * (x2 - x0) * (x2 - x1))
        y_out[j - 1] = (
            aintm * y_in[jm - 1]
            + aint0 * y_in[j0 - 1]
            + aint1 * y_in[j1 - 1]
            + aint2 * y_in[j2 - 1]
        )
    return y_out


def _deriv_fd(y, x):
    """GENE's deriv_fd: centred differences inside, one-sided at both ends, uniform dx."""
    out = np.empty_like(y)
    out[1:-1] = 0.5 * (y[2:] - y[:-2])
    out[0] = y[1] - y[0]
    out[-1] = y[-1] - y[-2]
    return out / (x[1] - x[0])


def _dlp_int_ind(var, dlp, h):
    """Running trapezoidal integral of var dlp (GENE's dlp_int_ind for ind = 1..np)."""
    f = var * dlp
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1])) * h
    return out


# --------------------------------------------------------------------------- input handling
def pressure_terms(geometry, species, beta, n_spec=None):
    """amhd and dpdx_pm as GENE resolves them (geometry.F90 initialize_geometry/check_geometry,
    profiles.F90 calc_dpdx_pm).

    geometry: dict with lower-case keys of the &geometry namelist; species: list of dicts
    (omn, omt, temp, dens, passive).  Returns (amhd, dpdx_pm) with the values GENE uses.
    """
    q0 = abs(float(geometry["q0"]))
    major_R = float(geometry.get("major_r", 1.0))
    n_spec = len(species) if n_spec is None else n_spec
    active = [s for s in species if not s.get("passive", False)]
    grad = sum(
        (float(s["omn"]) + float(s["omt"])) * float(s["temp"]) * float(s["dens"])
        for s in active
    )
    if n_spec == 1:
        grad *= 2.0
    amhd = float(geometry.get("amhd", 0.0))
    if amhd < 0.0:
        amhd = q0**2 * major_R * beta * grad
    dpdx_term = str(geometry.get("dpdx_term", "")).strip().lower()
    dpdx_pm = float(geometry.get("dpdx_pm", -2.0))
    if dpdx_term == "curv_eq_gradb":
        dpdx_pm = 0.0
    if dpdx_pm == -1111.0:
        dpdx_pm = -2.0
    if dpdx_pm == -1.0:
        dpdx_pm = beta * grad
    elif dpdx_pm == -2.0:
        dpdx_pm = amhd / (q0**2 * major_R)
    return amhd, dpdx_pm


# --------------------------------------------------------------------------- the geometry
def get_miller(
    q0,
    shat,
    amhd,
    major_R=1.0,
    trpeps=None,
    rho=None,
    kappa=1.0,
    s_kappa=0.0,
    delta=0.0,
    s_delta=0.0,
    zeta=0.0,
    s_zeta=0.0,
    drR=0.0,
    drZ=0.0,
    major_Z=0.0,
    sign_Ip_CW=1,
    sign_Bt_CW=1,
    nz0=512,
    n_pol=1,
):
    """Port of GENE's get_miller for magn_geometry = 'miller'.  Returns a dict with the 16
    miller.dat columns on GENE's z grid (R and Z still in units of L_ref, phi = 0) and the
    scalars q0 (signed), trpeps, rho (= x0), C_y, C_xy, drPsi."""
    sign_Ip_CW = int(np.sign(sign_Ip_CW)) or 1
    sign_Bt_CW = int(np.sign(sign_Bt_CW)) or 1
    pi = np.arccos(-1.0)
    n_pol_ext, n_pol_s = n_pol + 2, n_pol + 1
    npt, np_s = 500 * n_pol_ext, 500 * n_pol_s
    if rho is None or rho < 0.0:
        rho = trpeps * major_R
    if rho <= 0.0:
        raise ValueError("flux surface radius not defined (trpeps or rho)")
    trpeps = rho / major_R
    q0 = sign_Ip_CW * sign_Bt_CW * abs(q0)
    R0 = major_R
    B0 = 1.0 * sign_Bt_CW
    F = R0 * B0
    Z0 = major_Z
    mu_0 = 4.0 * pi
    theta = _linspace(-pi * n_pol_ext, pi * n_pol_ext - 2 * pi * n_pol_ext / npt, npt)
    d_inv = np.arcsin(delta)

    def surface(th):
        R = R0 + rho * np.cos(th + d_inv * np.sin(th))
        Z = Z0 + kappa * rho * np.sin(th + zeta * np.sin(2 * th))
        R_rho = (
            drR
            + np.cos(th + d_inv * np.sin(th))
            - s_delta * np.sin(th) * np.sin(th + d_inv * np.sin(th))
        )
        Z_rho = (
            drZ
            + kappa * s_zeta * np.cos(th + zeta * np.sin(2 * th)) * np.sin(2 * th)
            + kappa * np.sin(th + zeta * np.sin(2 * th))
            + kappa * s_kappa * np.sin(th + zeta * np.sin(2 * th))
        )
        R_th = -(rho * (1 + d_inv * np.cos(th)) * np.sin(th + d_inv * np.sin(th)))
        Z_th = (
            kappa
            * rho
            * (1 + 2 * zeta * np.cos(2 * th))
            * np.cos(th + zeta * np.sin(2 * th))
        )
        R_thth = -(
            rho * (1 + d_inv * np.cos(th)) ** 2 * np.cos(th + d_inv * np.sin(th))
        ) + d_inv * rho * np.sin(th) * np.sin(th + d_inv * np.sin(th))
        Z_thth = -4 * kappa * rho * zeta * np.cos(th + zeta * np.sin(2 * th)) * np.sin(
            2 * th
        ) - kappa * rho * (1 + 2 * zeta * np.cos(2 * th)) ** 2 * np.sin(
            th + zeta * np.sin(2 * th)
        )
        return R, Z, R_rho, Z_rho, R_th, Z_th, R_thth, Z_thth

    thetaShift = 0.0
    iBmax = 0
    bMaxShift = True
    while bMaxShift:
        thAdj = theta + thetaShift
        R, Z, R_rho, Z_rho, R_theta, Z_theta, R_theta_theta, Z_theta_theta = surface(
            thAdj
        )
        dlp = (R_theta**2 + Z_theta**2) ** 0.5
        Rc = dlp**3 * (R_theta * Z_theta_theta - Z_theta * R_theta_theta) ** (-1)
        cosu = Z_theta / dlp
        J_r = R * (R_rho * Z_theta - R_theta * Z_rho)
        tmp = J_r / R**2
        drPsi = (
            sign_Ip_CW
            * F
            / (2.0 * pi * n_pol_ext * q0)
            * np.sum(tmp)
            * 2
            * pi
            * n_pol_ext
            / npt
        )
        Bp = sign_Ip_CW * drPsi / J_r * dlp
        Bphi = F / R
        B = np.sqrt(Bphi**2 + Bp**2)
        bMaxShift = False
        if thetaShift == 0.0 and abs(drZ) > np.finfo(float).eps:
            for i in range(1, 500):
                if B[i] > B[iBmax]:
                    iBmax = i
            if iBmax != 0:
                bMaxShift = True
                thetaShift = theta[iBmax] - theta[0]

    dx_drho = 1.0
    dxPsi = drPsi / dx_drho
    C_y = dxPsi * sign_Ip_CW
    C_xy = abs(B0 * dxPsi / C_y)
    dq_dx = shat * q0 / rho / dx_drho
    dq_dpsi = dq_dx / dxPsi
    pprime = -amhd / q0**2 / R0 / (2 * mu_0) * B0**2 / drPsi
    dpdx_pm_geom = amhd / q0**2 / R0 / dx_drho

    psi1 = R * Bp * sign_Ip_CW
    h = 2 * pi * n_pol_ext / npt
    t0 = (2.0 / Rc - 2.0 * cosu / R) / (R * psi1**2)
    t1 = B**2 * R / psi1**3
    t2 = mu_0 * R / psi1**3
    t3 = 1.0 / (R * psi1)
    D0 = -F * _dlp_int_ind(t0, dlp, h)
    D1 = -_dlp_int_ind(t1, dlp, h) / F
    D2 = -_dlp_int_ind(t2, dlp, h) * F
    D3 = -_dlp_int_ind(t3, dlp, h) * F
    D0_full = -F * np.sum(t0 * dlp) * h
    D1_full = -np.sum(t1 * dlp) * h / F
    D2_full = -np.sum(t2 * dlp) * h * F
    mid = npt // 2  # Fortran index np/2 + 1
    ffprime = (
        -(sign_Ip_CW * dq_dpsi * 2.0 * pi * n_pol_ext + D0_full + D2_full * pprime)
        / D1_full
    )
    D0 = D0 - D0[mid]
    D1 = D1 - D1[mid]
    D2 = D2 - D2[mid]
    nu = D3 - D3[mid]
    nu1 = psi1 * (D0 + D1 * ffprime + D2 * pprime)

    chi = -nu / q0
    chi = chi * (np.max(theta) - np.min(theta)) / (np.max(chi) - np.min(chi))
    if not (np.all(np.diff(chi) >= 0) or np.all(np.diff(chi) <= 0)):
        raise ValueError("Miller geometry: non-monotonic straight-field-line angle")

    chi_s = _linspace(-pi * n_pol_s, pi * n_pol_s - 2 * pi * n_pol_s / np_s, np_s)
    if sign_Ip_CW < 0:
        # chi decreases with theta: GENE interpolates on the reversed arrays (theta_s is then a
        # decreasing function of chi_s) and keeps theta_s_reverse for the interpolations below
        theta_s = lag3interp(theta[::-1], chi[::-1], chi_s)
        theta_s_rev = theta_s[::-1]
    else:
        theta_s = lag3interp(theta, chi, chi_s)
    dtheta_dchi_s = _deriv_fd(theta_s, chi_s)
    thAdj_s = theta_s + thetaShift
    R_s = R0 + rho * np.cos(thAdj_s + d_inv * np.sin(thAdj_s))
    R_theta_s = -(
        dtheta_dchi_s
        * rho
        * (1 + d_inv * np.cos(thAdj_s))
        * np.sin(thAdj_s + d_inv * np.sin(thAdj_s))
    )
    Z_s = Z0 + kappa * rho * np.sin(thAdj_s + zeta * np.sin(2 * thAdj_s))
    Z_theta_s = (
        dtheta_dchi_s
        * kappa
        * rho
        * (1 + 2 * zeta * np.cos(2 * thAdj_s))
        * np.cos(thAdj_s + zeta * np.sin(2 * thAdj_s))
    )

    if sign_Ip_CW < 0:
        # GENE interpolates onto theta_s_reverse and reverses the result
        def interp(f):
            return lag3interp(f, theta, theta_s_rev)[::-1]

    else:

        def interp(f):
            return lag3interp(f, theta, theta_s)

    nu1_s, Bp_s, dlp_s, Rc_s = interp(nu1), interp(Bp), interp(dlp), interp(Rc)

    psi1_s = R_s * Bp_s * sign_Ip_CW
    dBp_dchi_s = _deriv_fd(Bp_s, chi_s)
    Bphi_s = F / R_s
    B_s = np.sqrt(Bphi_s**2 + Bp_s**2)
    cosu_s = Z_theta_s / dlp_s / dtheta_dchi_s
    sinu_s = -R_theta_s / dlp_s / dtheta_dchi_s

    dB_drho_s = (
        -1.0 / B_s * (F**2 / R_s**3 * cosu_s + Bp_s**2 / Rc_s + mu_0 * psi1_s * pprime)
    )
    dB_dl_s = (
        1.0 / B_s * (Bp_s * dBp_dchi_s / dtheta_dchi_s / dlp_s + F**2 / R_s**3 * sinu_s)
    )
    dnu_drho_s = nu1_s
    dnu_dl_s = -F / (R_s * psi1_s)
    grad_nu_s = np.sqrt(dnu_drho_s**2 + dnu_dl_s**2)

    gxx = (psi1_s / dxPsi) ** 2
    gxy = -psi1_s / dxPsi * C_y * sign_Ip_CW * nu1_s
    gxz = -psi1_s / dxPsi * (nu1_s + psi1_s * dq_dpsi * chi_s) / q0
    gyy = C_y**2 * (grad_nu_s**2 + 1 / R_s**2)
    gyz = sign_Ip_CW * C_y / q0 * (grad_nu_s**2 + dq_dpsi * nu1_s * psi1_s * chi_s)
    gzz = (
        1.0
        / q0**2
        * (
            grad_nu_s**2
            + 2.0 * dq_dpsi * nu1_s * psi1_s * chi_s
            + (dq_dpsi * psi1_s * chi_s) ** 2
        )
    )
    jac = 1.0 / np.sqrt(
        gxx * gyy * gzz
        + 2.0 * gxy * gyz * gxz
        - gxz**2 * gyy
        - gyz**2 * gxx
        - gzz * gxy**2
    )
    dBdx = (
        jac
        * C_y
        / (q0 * R_s)
        * (
            F / (R_s * psi1_s) * dB_drho_s
            + (nu1_s + dq_dpsi * chi_s * psi1_s) * dB_dl_s
        )
    )
    dBdz = 1.0 / B_s * (Bp_s * dBp_dchi_s - F**2 / R_s**3 * R_theta_s)
    dxdR_s = dx_drho / drPsi * psi1_s * cosu_s
    dxdZ_s = dx_drho / drPsi * psi1_s * sinu_s

    chi_out = _linspace(-pi * n_pol, pi * n_pol - 2 * pi * n_pol / nz0, nz0)
    out = {}
    for name, arr in (
        ("gxx", gxx),
        ("gxy", gxy),
        ("gxz", gxz),
        ("gyy", gyy),
        ("gyz", gyz),
        ("gzz", gzz),
        ("B", B_s),
        ("jacobian", jac),
        ("dBdx", dBdx),
        ("dBdz", dBdz),
        ("R", R_s),
        ("Z", Z_s),
        ("dxdR", dxdR_s),
        ("dxdZ", dxdZ_s),
    ):
        out[name] = lag3interp(arr, chi_s, chi_out)
    out["dBdy"] = np.zeros(nz0)
    out["phi"] = np.zeros(nz0)
    out.update(
        z=chi_out,
        q0=q0,
        shat=shat,
        trpeps=trpeps,
        rho=rho,
        x0=rho,
        C_y=C_y,
        C_xy=C_xy,
        drPsi=drPsi,
        dpdx_pm_geom=dpdx_pm_geom,
        thetaShift=thetaShift,
        major_R=major_R,
        sign_Ip_CW=sign_Ip_CW,
        sign_Bt_CW=sign_Bt_CW,
    )
    return out


def curvature(g):
    """GENE's curvature coefficients (geometry.F90 set_curvature) from the miller.dat columns:
    K_x = -(dBdy + ga2/ga1 dBdz)/C_xy, K_y = (dBdx - ga3/ga1 dBdz)/C_xy."""
    ga1 = g["gxx"] * g["gyy"] - g["gxy"] ** 2
    ga2 = g["gxx"] * g["gyz"] - g["gxy"] * g["gxz"]
    ga3 = g["gxy"] * g["gyz"] - g["gyy"] * g["gxz"]
    K_x = -(g["dBdy"] + ga2 / ga1 * g["dBdz"]) / g["C_xy"]
    K_y = (g["dBdx"] - ga3 / ga1 * g["dBdz"]) / g["C_xy"]
    return K_x, K_y


def miller_from_namelist(nml, nz0=512):
    """Run get_miller from a GENE parameters namelist (f90nml Namelist or nested dict with
    the GENE group names).  Returns (geometry dict, header dict) where the header holds what
    GENE writes at the top of miller.dat (my_dpdx is the resolved dpdx_pm)."""
    geo = {k.lower(): v for k, v in nml["geometry"].items()}
    gen = {k.lower(): v for k, v in nml["general"].items()}
    spec = nml["species"]
    spec = list(spec) if isinstance(spec, list) else [spec]
    spec = [{k.lower(): v for k, v in s.items()} for s in spec]
    beta = float(gen.get("beta", 0.0))
    amhd, dpdx_pm = pressure_terms(geo, spec, beta)
    mg = str(geo.get("magn_geometry", "")).strip().lower()
    if mg != "miller":
        raise ValueError(f"magn_geometry = '{mg}' is not supported (only 'miller')")
    for key in ("thetak", "thetad"):
        if float(geo.get(key, 0.0)) != 0.0:
            raise ValueError(f"Miller tilt '{key}' is not supported")
    if float(geo.get("edge_opt", 0.0)) != 0.0:
        raise ValueError("edge_opt != 0 is not supported")
    if float(gen.get("n_pol", geo.get("n_pol", 1))) != 1:
        raise ValueError("n_pol != 1 is not supported")
    sign_ip = int(gen.get("sign_ip_cw", geo.get("sign_ip_cw", 1)) or 1)
    sign_bt = int(gen.get("sign_bt_cw", geo.get("sign_bt_cw", 1)) or 1)
    g = get_miller(
        q0=float(geo["q0"]),
        shat=float(geo["shat"]),
        amhd=amhd,
        major_R=float(geo.get("major_r", 1.0)),
        trpeps=float(geo.get("trpeps", 0.0)),
        rho=float(geo["rho"]) if "rho" in geo else None,
        kappa=float(geo.get("kappa", 1.0)),
        s_kappa=float(geo.get("s_kappa", 0.0)),
        delta=float(geo.get("delta", 0.0)),
        s_delta=float(geo.get("s_delta", 0.0)),
        zeta=float(geo.get("zeta", 0.0)),
        s_zeta=float(geo.get("s_zeta", 0.0)),
        drR=float(geo.get("drr", 0.0)),
        drZ=float(geo.get("drz", 0.0)),
        major_Z=float(geo.get("major_z", 0.0)),
        sign_Ip_CW=sign_ip,
        sign_Bt_CW=sign_bt,
        nz0=nz0,
    )
    units = {k.lower(): v for k, v in nml["units"].items()} if "units" in nml else {}
    Lref = float(units.get("lref", 0.0) or 0.0)
    Bref = float(units.get("bref", 0.0) or 0.0)
    g["Lref"], g["Bref"] = Lref, Bref
    g["amhd"], g["my_dpdx"] = amhd, dpdx_pm
    g["minor_r"] = float(geo.get("minor_r", 1.0))
    g["beta"] = beta
    K_x, K_y = curvature(g)
    g["K_x"], g["K_y"] = K_x, K_y
    return g


def write_miller_dat(g, path):
    """Write g (from get_miller/miller_from_namelist) as GENE's miller.dat (geometry.F90
    write_geometry, x-local case)."""
    Lref = g.get("Lref", 0.0)
    lf = Lref if Lref > 0 else 1.0
    e = lambda v: "%24.16E" % v  # noqa: E731  (Fortran ES24.16)
    lines = [
        "&parameters",
        "gridpoints = %5d" % g["z"].size,
        "q0 = " + e(g["q0"]),
        "shat = " + e(g["shat"]),
        "s0   = " + e(g["x0"] ** 2),
        "minor_r= " + e(g.get("minor_r", 1.0)),
        "major_R= " + e(g["major_R"]),
        "trpeps = " + e(g["trpeps"]),
        "beta = " + e(g.get("beta", 0.0)),
        "edge_opt = " + e(0.0),
        "Lref = " + e(Lref),
        "Bref = " + e(g.get("Bref", 0.0)),
    ]
    if g.get("my_dpdx", 0.0) != 0.0:
        lines.append("my_dpdx = " + e(g["my_dpdx"]))
    lines += [
        "Cy = " + e(g["C_y"]),
        "Cxy = " + e(g["C_xy"]),
        "magn_geometry = 'miller'",
        "",
        "sign_Ip_CW = %3d" % g["sign_Ip_CW"],
        "sign_Bt_CW = %3d" % g["sign_Bt_CW"],
        "/",
    ]
    cols = []
    for c in COLUMNS:
        v = g[c]
        cols.append(v * lf if c in ("R", "Z") else v)
    A = np.array(cols).T
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for row in A:
            f.write("".join("%24.16E" % v for v in row) + "\n")


def read_miller_dat(path):
    """Read a GENE miller.dat: (header dict, columns dict)."""
    import re

    txt = open(path).read()
    hdr, body = txt.split("/", 1)
    h = {}
    for k, v in re.findall(r"(\w+)\s*=\s*([^\n]+)", hdr):
        v = v.strip().strip("'")
        try:
            h[k] = float(v)
        except ValueError:
            h[k] = v
    a = np.array(
        [
            [float(x) for x in ln.split()]
            for ln in body.strip().splitlines()
            if ln.strip()
        ]
    )
    return h, {c: a[:, i] for i, c in enumerate(COLUMNS)}
