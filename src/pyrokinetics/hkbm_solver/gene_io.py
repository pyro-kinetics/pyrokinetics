"""
GENE-compatible front end of the hKBM solver: read a GENE ``parameters`` file, solve the
local linear eigenproblem at each k_y, and write GENE's output files so that GENE readers
(pyrokinetics' ``GKOutputReaderGENE`` in particular) load the result unchanged.

Input (Fortran namelist, read with f90nml): &box kymin, nky0, nz0 (resolution of the output
grid only), kx_center; &general beta, bpar, coll, collision_op, nonlinear, ExBrate/pfsrate;
&geometry (magn_geometry = 'miller' and its parameters, amhd, dpdx_pm, dpdx_term, sign_Ip_CW,
sign_Bt_CW); &species (name, charge, mass, temp, dens, omn, omt, and the delta B_par switches
bpar_vlasov, bpar_field, bpar_source of the hKBM GENE branch); &units (optional, echoed).

Output (GENE's names: ``<name>.dat`` for a single k_y, ``<name>_0001`` ... for several k_y, one
GENE-like run per k_y as a GENE scan writes them, plus ``scan.log``):
  parameters  the input echoed as GENE writes parameters.dat (resolved amhd, dpdx_pm, nx0 = 1,
              nky0 = 1, kymin = k_y) with GENE's &info block
  omega       "k_y gamma omega" (GENE's format and sign: omega > 0 = ion diamagnetic direction)
  field       GENE's binary field file (phi, A_par, B_par on GENE's z grid) holding a short time
              series of the eigenmode growing at gamma (a linear GENE run's late-time behaviour)
  nrg         zero fluxes at the same times (the solver computes no fluxes)
  miller      GENE's miller.dat of the geometry used
  hkbm        JSON with the full-precision eigenvalue, convergence data and the unit factors

Units: everything in and out is in the deck's own GENE normalisation (T_ref, n_ref, m_ref,
B_ref, L_ref).  Internally the solver uses T_ref = T_e, n_ref = n_e, m_ref = m_i (same B_ref,
L_ref); the conversion factors are applied here (``Deck.units``).
"""

import json
import re
import struct
import warnings
from pathlib import Path

import f90nml
import numpy as np
from scipy.interpolate import CubicSpline

from . import solver as S
from .geometry import Geo, Geometry
from .miller import miller_from_namelist, write_miller_dat


class UnsupportedDeck(ValueError):
    """The deck asks for physics the hKBM solver does not have."""


def _lower(d):
    return {str(k).lower(): v for k, v in d.items()}


def _species_list(nml):
    sp = nml["species"]
    sp = list(sp) if isinstance(sp, list) else [sp]
    return [_lower(s) for s in sp]


def _bool(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in (".true.", "t", "true")
    return bool(v)


def _scan_values(text, key):
    """GENE scan syntax on a parameter line: 'key = v !scanlist: a, b, c' or '!scan: a:step:b'."""
    for line in text.splitlines():
        m = re.match(
            r"\s*%s\s*=\s*([^!]*)!\s*(scanlist|scan)\s*:\s*(.*)$" % key, line, re.I
        )
        if not m:
            continue
        kind, spec = m.group(2).lower(), m.group(3).strip()
        if kind == "scanlist":
            return [float(x) for x in spec.replace(",", " ").split()]
        a, step, b = (float(x) for x in spec.split(":"))
        n = int(round((b - a) / step)) + 1
        return [a + i * step for i in range(n)]
    return None


class Deck:
    """A GENE parameters file translated into solver inputs.

    Attributes: nml (f90nml.Namelist), kys (k_y rho_ref values to solve), solver_kw (Solver
    options), params (plasma parameters in solver units), geo (geometry.Geo), units (factors
    ref -> solver), warnings (list of str)."""

    def __init__(self, path, nz_geo=512):
        self.path = Path(path)
        self.text = self.path.read_text()
        self.nml = f90nml.reads(self.text)
        self.warnings = []
        self._check_and_translate(nz_geo)

    def warn(self, msg):
        self.warnings.append(msg)
        warnings.warn(msg, stacklevel=3)

    # ------------------------------------------------------------------ translation
    def _check_and_translate(self, nz_geo):
        nml = self.nml
        for grp in ("box", "general", "geometry", "species"):
            if grp not in nml:
                raise UnsupportedDeck(f"GENE parameters file has no &{grp} namelist")
        box, gen, geo = (
            _lower(nml["box"]),
            _lower(nml["general"]),
            _lower(nml["geometry"]),
        )
        spec = _species_list(nml)
        if _bool(gen.get("nonlinear"), False):
            raise UnsupportedDeck("nonlinear = .true.: the hKBM solver is linear only")
        ext = _lower(nml["external_contr"]) if "external_contr" in nml else {}
        for k in ("exbrate", "pfsrate", "omega0_tor"):
            if float(gen.get(k, ext.get(k, 0.0)) or 0.0) != 0.0:
                raise UnsupportedDeck(
                    f"{k} != 0: flow and flow shear are not in the hKBM solver"
                )
        if float(box.get("kx_center", 0.0) or 0.0) != 0.0:
            raise UnsupportedDeck(
                "kx_center != 0: the solver has the k_x = 0 (theta_0 = 0) ballooning mode only"
            )
        if int(box.get("n_spec", len(spec))) != len(spec):
            raise UnsupportedDeck(
                "n_spec does not match the number of &species namelists"
            )
        if len(spec) != 2:
            raise UnsupportedDeck(
                f"{len(spec)} species: the hKBM solver needs exactly two kinetic species, one ion species "
                "(charge +1) and electrons (charge -1); impurities and adiabatic species are not supported"
            )
        ele = [s for s in spec if float(s["charge"]) < 0]
        ion = [s for s in spec if float(s["charge"]) > 0]
        if len(ele) != 1 or len(ion) != 1:
            raise UnsupportedDeck(
                "need one electron species (charge < 0) and one ion species (charge > 0)"
            )
        e, i = ele[0], ion[0]
        if float(e["charge"]) != -1.0 or float(i["charge"]) != 1.0:
            raise UnsupportedDeck(
                "charges must be -1 (electrons) and +1 (singly charged ions)"
            )
        for s in spec:
            if _bool(s.get("passive"), False):
                raise UnsupportedDeck("passive species are not supported")
            if _bool(s.get("no_trap"), False):
                raise UnsupportedDeck("no_trap is not supported")
            if float(s.get("omegator", 0.0) or 0.0) != 0.0 or float(
                s.get("prof_type", 0) or 0
            ) not in (0,):
                raise UnsupportedDeck(
                    "species rotation or radial profiles are not supported (local, no flow)"
                )
            if str(s.get("f0_type", "maxwellian")).strip().lower() not in (
                "maxwellian",
                "",
            ):
                raise UnsupportedDeck("only Maxwellian backgrounds are supported")
            for key in ("bpar_vlasov_terms", "bpar_vlasov_pitch"):
                if str(s.get(key, "all")).strip().lower() != "all":
                    raise UnsupportedDeck(
                        f"{key} = '{s[key]}' is not supported (only 'all')"
                    )
            if str(s.get("bpar_source", "full")).strip().lower() not in (
                "full",
                "none",
            ):
                raise UnsupportedDeck(
                    f"bpar_source = '{s['bpar_source']}' is not supported ('full' or 'none')"
                )
        ne, ni = float(e["dens"]), float(i["dens"])
        if abs(ne - ni) > 1e-6 * ne:
            raise UnsupportedDeck(
                f"n_e = {ne} != n_i = {ni}: not quasineutral with one singly charged ion"
            )
        if abs(float(e["omn"]) - float(i["omn"])) > 1e-6 * max(
            1.0, abs(float(e["omn"]))
        ):
            raise UnsupportedDeck(
                "omn differs between ions and electrons (not quasineutral)"
            )
        beta = float(gen.get("beta", 0.0) or 0.0)
        if beta <= 0:
            raise UnsupportedDeck(
                "beta <= 0: the solver is electromagnetic (A_par always on) and needs beta > 0"
            )
        if (
            str(gen.get("magn_geometry", geo.get("magn_geometry", ""))).strip().lower()
            != "miller"
        ):
            raise UnsupportedDeck("only magn_geometry = 'miller' is supported")
        # units: solver reference T_e, n_e, m_i (B_ref, L_ref unchanged)
        Te, me, mi = float(e["temp"]), float(e["mass"]), float(i["mass"])
        Ti = float(i["temp"])
        rs = np.sqrt(Te * mi)  # rho_s / rho_ref
        cs = np.sqrt(Te / mi)  # c_s / c_ref
        self.units = dict(
            rho_s_over_rho_ref=rs, c_s_over_c_ref=cs, T_e=Te, n_e=ne, m_i=mi
        )
        self.params = dict(
            Ti=Ti / Te,
            omn=float(e["omn"]),
            omte=float(e["omt"]),
            omti=float(i["omt"]),
            beta=beta * ne * Te,
            me=me / mi,
            q0=abs(float(geo["q0"])),
            R0=float(geo.get("major_r", 1.0)),
        )
        # collisions: GENE's nu_ei (collisions_common.F90 compute_nuei) in c_ref/L_ref -> solver coll
        cop = str(gen.get("collision_op", "none")).strip().lower()
        coll = float(gen.get("coll", 0.0) or 0.0)
        if cop == "none" or coll <= 0:
            coll_s = 0.0
        else:
            nu_ei_ref = 4 * ni / Te**1.5 / me**0.5 * coll
            nu_ei_s = nu_ei_ref / cs
            coll_s = nu_ei_s * np.sqrt(self.params["me"]) / 4
            if cop not in ("landau", "sugama", "pitch-angle", "lorentz"):
                self.warn(
                    f"collision_op = '{cop}': unknown to the solver; its own pitch-angle scattering is used"
                )
            else:
                self.warn(
                    f"collision_op = '{cop}': the solver uses its own reduced model (bounce-averaged "
                    "Lorentz pitch-angle scattering of trapped electrons, Krook detrapping of trapped ions) "
                    "at GENE's nu_ei"
                )
        self.nu_ei_ref = 4 * ni / Te**1.5 / me**0.5 * coll if coll_s > 0 else 0.0
        # electromagnetic switches
        sw = {}
        if not _bool(gen.get("bpar"), False):
            bpar = "off"
            self.warn(
                "bpar = .false.: delta B_par off (GENE default); set bpar = .true. for the hKBM"
            )
        else:
            bpar = "field"
        dterm = str(geo.get("dpdx_term", "")).strip().lower()
        if dterm in ("gradb_eq_curv", "off"):
            sw.update(bp_e=0.0, bp_i=0.0)
        elif dterm not in ("full_drift", "on", "curv_eq_gradb", ""):
            raise UnsupportedDeck(f"dpdx_term = '{dterm}' is not supported")
        elif dterm == "":
            self.warn(
                "dpdx_term not set: GENE's choice for bpar/beta decks ('full_drift') is used"
            )
        for tag, s in (("e", e), ("i", i)):
            sw["vl_" + tag] = 1.0 if _bool(s.get("bpar_vlasov"), True) else 0.0
            sw["fd_" + tag] = 1.0 if _bool(s.get("bpar_field"), True) else 0.0
            sw["src_" + tag] = (
                0.0
                if str(s.get("bpar_source", "full")).strip().lower() == "none"
                else 1.0
            )
        for k in ("hyp_z", "hyp_v", "hyp_x", "hyp_y"):
            if float(gen.get(k, 0.0) or 0.0) > 0:
                self.warn(
                    f"{k} > 0 is ignored (the solver has no numerical dissipation)"
                )
        if float(gen.get("debye2", 0.0) or 0.0) != 0.0:
            self.warn("debye2 is ignored")
        # geometry (GENE's Miller, computed in Python at nz_geo points)
        self.miller = miller_from_namelist(nml, nz0=nz_geo)
        self.geo = Geo.from_miller(self.miller)
        self.solver_kw = dict(
            S.DEFAULT, coll=coll_s, bpar=bpar, params=self.params, **sw
        )
        # k_y list (deck units)
        kymin = float(box["kymin"])
        nky0 = int(box.get("nky0", 1))
        scan = _scan_values(self.text, "kymin")
        if scan is not None:
            self.kys = scan
        elif nky0 > 1:
            self.kys = [kymin * (j + 1) for j in range(nky0)]
        else:
            self.kys = [kymin]
        self.nz0 = int(box.get("nz0", 64))

    # ------------------------------------------------------------------ solving
    def solve(self, ky_ref, omega0=None, verbose=False, scan="auto"):
        """Solve at one k_y (deck units).  omega0: seed (deck units, GENE sign omega + i gamma).

        Seeds tried in turn: omega0 (if given; accepted as soon as it converges to a growing
        root), the STEP hKBM root scaled to this k_y (accepted likewise), then drift-wave-like
        seeds in both directions (the most unstable converged root is kept).  scan: 'auto' (a
        coarse search of the smallest singular value of D(omega) when no seed converges to a
        growing root), True (always also scan) or False."""
        rs, cs = self.units["rho_s_over_rho_ref"], self.units["c_s_over_c_ref"]
        ky = ky_ref * rs
        geom = Geometry(self.geo, nth=1024)
        Sv = S.Solver(ky, geom=geom, **self.solver_kw)
        first = []
        if omega0 is not None:
            first.append(complex(omega0) / cs)
        k = min(S.GENE_SEEDS, key=lambda x: abs(np.log(x / ky)))
        g = S.GENE_SEEDS[k]
        first.append(complex(g[0] * ky / k, g[1]))
        wst = ky * max(abs(self.params["omn"]), 0.5)
        generic = [
            complex(0.5 * wst, 0.3 * wst),
            complex(-0.5 * wst, 0.3 * wst),
            complex(wst, 0.1 * wst),
        ]
        tried = []

        def attempt(sd):
            r = Sv.solve(complex(-sd.real, sd.imag))
            tried.append(r)
            if verbose:
                print(
                    "  seed %+.4f%+.4fj: omega %+.5f gamma %+.5f converged %s"
                    % (sd.real, sd.imag, r["omega_gene"], r["gamma"], r["converged"]),
                    flush=True,
                )
            return r["converged"] and r["gamma"] > 0 and np.isfinite(r["omega"])

        best = None
        for sd in first:
            if attempt(sd):
                best = tried[-1]
                break
        if best is None or scan is True:
            if best is None:
                for sd in generic:
                    attempt(sd)
            if scan == "auto":
                scan = not any(r["converged"] and r["gamma"] > 0 for r in tried)
            if scan:
                for sd in self._scan_seeds(Sv, ky, verbose):
                    attempt(sd)
            good = [
                r
                for r in tried
                if r["converged"] and r["gamma"] > 0 and np.isfinite(r["omega"])
            ]
            if good:
                best = max(good, key=lambda r: r["gamma"])
        if best is None:
            best = dict(tried[-1])
            best["converged"] = False
        best["solver"] = Sv
        best["ky_ref"] = ky_ref
        best["ky_solver"] = ky
        best["omega_ref"] = best["omega_gene"] * cs
        best["gamma_ref"] = best["gamma"] * cs
        return best

    @staticmethod
    def _scan_seeds(Sv, ky, verbose, nmax=3):
        """Local minima of sigma_min/sigma_max of the scaled D(omega) on a coarse grid (GENE sign
        omega in [-3, 3] x max(ky omn, 0.05) ... plus the Alfvenic range up to |omega| = 2;
        gamma > 0 only, where the streaming-ion response is valid)."""
        wr = np.unique(
            np.concatenate(
                [
                    np.linspace(-2.0, 2.0, 11),
                    np.linspace(-0.6, 0.6, 9) * max(ky, 0.1) / 0.2,
                ]
            )
        )
        wi = np.array([0.02, 0.08, 0.2])
        A = np.empty((wi.size, wr.size))
        for j, y in enumerate(wi):
            for i, x in enumerate(wr):
                D, _ = Sv.matrix(complex(-x, y))
                sv = np.linalg.svd(Sv.scales()[:, None] * D, compute_uv=False)
                A[j, i] = sv[-1] / sv[0]
        mins = []
        for j in range(wi.size):
            for i in range(wr.size):
                nb = A[max(j - 1, 0) : j + 2, max(i - 1, 0) : i + 2]
                if A[j, i] <= nb.min():
                    mins.append((A[j, i], complex(wr[i], wi[j])))
        mins.sort(key=lambda t: t[0])
        if verbose:
            print(
                "  scan minima (GENE sign):",
                ", ".join("%+.3f%+.3fj" % (m[1].real, m[1].imag) for m in mins[:nmax]),
            )
        return [m[1] for m in mins[:nmax]]


# ---------------------------------------------------------------------------- output
def _fields_on_z(res, deck, nz0):
    """phi, A_par, B_par of a solver result on GENE's z grid (nz0 points), in the deck's GENE
    normalisation, raw GENE convention (time dependence exp(-i omega_model t))."""
    Sv = res["solver"]
    geo = Sv.geo
    th = geo.theta
    z = -np.pi + 2 * np.pi * np.arange(nz0) / nz0
    omega = res["omega"]  # model sign, solver units
    tp = np.append(th, np.pi)

    def spl(f):
        return (
            CubicSpline(tp, np.append(f.real, f.real[0]), bc_type="periodic"),
            CubicSpline(tp, np.append(f.imag, f.imag[0]), bc_type="periodic"),
        )

    pr, pi_ = spl(res["phi"])
    br, bi = spl(res["bpar"])
    sr, si = spl(res["psi"])
    phi = pr(z) + 1j * pi_(z)
    bpar = br(z) + 1j * bi(z)
    dpsi = sr(z, 1) + 1j * si(z, 1)
    JB = geo.g.sJ(z) * geo.g.sB(z)
    apar = -1j / omega * dpsi / JB  # A_par = -(i/omega) d_l psi, dl = J B dz
    u = deck.units
    rs = u["rho_s_over_rho_ref"]
    phi = phi * u["T_e"] * rs  # (T_e/e)(rho_s/L) -> (T_ref/e)(rho_ref/L)
    apar = apar * rs**2  # B_ref rho_s^2/L -> B_ref rho_ref^2/L
    bpar = bpar * rs  # B_ref rho_s/L -> B_ref rho_ref/L
    return z, phi, apar, bpar


def _fmt_val(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, str):
        return "'%s'" % v
    if isinstance(v, (int, np.integer)):
        return "%d" % v
    if isinstance(v, list):
        return " ".join(_fmt_val(x) for x in v)
    return "%.16G" % v


def _group(name, items):
    lines = ["&" + name]
    for k, v in items:
        lines.append("%s = %s" % (k, _fmt_val(v)))
    lines.append("/")
    return "\n".join(lines) + "\n\n"


def write_parameters(path, deck, ky, nsteps, dt):
    """GENE parameters.dat for one k_y (GENE's layout, as parameters_IO.F90 writes it)."""
    nml = deck.nml
    box = _lower(nml["box"])
    gen = _lower(nml["general"])
    geo = _lower(nml["geometry"])
    m = deck.miller
    out = []
    par = [(k, v) for k, v in _lower(nml.get("parallelization", {})).items()] or [
        ("n_procs_s", 1)
    ]
    out.append(_group("parallelization", par))
    bx = dict(
        n_spec=2,
        nx0=1,
        nky0=1,
        nz0=deck.nz0,
        nv0=int(box.get("nv0", 32)),
        nw0=int(box.get("nw0", 16)),
        kymin=float(ky),
        lv=float(box.get("lv", 3.0)),
        lw=float(box.get("lw", 9.0)),
        adapt_lx=True,
        x0=float(m["x0"]),
        ky0_ind=1,
        mu_grid_type=str(box.get("mu_grid_type", "eq_vperp")),
    )
    out.append(_group("box", bx.items()))
    out.append(
        _group(
            "in_out",
            [
                ("diagdir", "./"),
                ("read_checkpoint", False),
                ("write_checkpoint", False),
                ("istep_field", 1),
                ("istep_mom", 0),
                ("istep_nrg", 1),
                ("istep_omega", 1),
                ("istep_vsp", 0),
                ("istep_schpt", 0),
                ("istep_energy", 0),
                ("write_std", True),
            ],
        )
    )
    g = [
        ("nonlinear", False),
        ("comp_type", "IV"),
        ("timescheme", "RK4"),
        ("dt_max", float(dt)),
        ("timelim", 1),
        ("ntimesteps", int(nsteps)),
        ("simtimelim", float(nsteps * dt)),
        ("beta", float(gen.get("beta"))),
        ("debye2", 0.0),
        ("bpar", _bool(gen.get("bpar"), False)),
        ("collision_op", str(gen.get("collision_op", "none"))),
    ]
    if str(gen.get("collision_op", "none")).strip().lower() != "none":
        g.append(("coll", float(gen.get("coll", 0.0))))
    g += [("init_cond", "alm"), ("hyp_z", float(gen.get("hyp_z", 0.0) or 0.0))]
    out.append(_group("general", g))
    ge = [
        ("magn_geometry", "miller"),
        ("q0", abs(float(geo["q0"]))),
        ("shat", float(geo["shat"])),
        ("amhd", float(m["amhd"])),
        ("major_R", float(m["major_R"])),
        ("minor_r", float(m.get("minor_r", 1.0))),
        ("trpeps", float(m["trpeps"])),
    ]
    for k in (
        "kappa",
        "delta",
        "zeta",
        "s_kappa",
        "s_delta",
        "s_zeta",
        "drr",
        "drz",
        "major_z",
    ):
        if k in geo or k in (
            "kappa",
            "delta",
            "zeta",
            "s_kappa",
            "s_delta",
            "s_zeta",
            "drr",
        ):
            name = {"drr": "drR", "drz": "drZ", "major_z": "major_Z"}.get(k, k)
            ge.append((name, float(geo.get(k, 1.0 if k == "kappa" else 0.0))))
    ge += [
        ("rhostar", float(geo.get("rhostar", -1.0))),
        ("dpdx_term", str(geo.get("dpdx_term", "full_drift"))),
        ("dpdx_pm", float(m["my_dpdx"])),
        ("norm_flux_projection", False),
        ("sign_Ip_CW", int(m["sign_Ip_CW"])),
        ("sign_Bt_CW", int(m["sign_Bt_CW"])),
    ]
    out.append(_group("geometry", ge))
    for s in _species_list(nml):
        items = [
            ("name", str(s["name"])),
            ("omn", float(s["omn"])),
            ("omt", float(s["omt"])),
            ("mass", float(s["mass"])),
            ("temp", float(s["temp"])),
            ("dens", float(s["dens"])),
            ("charge", float(s["charge"])),
        ]
        for k in ("bpar_vlasov", "bpar_field"):
            if k in s:
                items.append((k, _bool(s[k], True)))
        if "bpar_source" in s:
            items.append(("bpar_source", str(s["bpar_source"])))
        out.append(_group("species", items))
    nu = ("nu_ei = %10.6f\n" % deck.nu_ei_ref) if deck.nu_ei_ref > 0 else ""
    out.append(
        "&info\n"
        "probdir = '%s'\n"
        "step_time  =     0.0000\n"
        "number of computed time steps = %7d\n"
        "time for initial value solver =      0.000\n"
        "calc_dt = F\n"
        "init_time =     0.0000\n"
        "n_fields = 3\n"
        "n_moms   =  0\n"
        "nrgcols  = 10\n"
        "%s"
        "Zeff_species =   1.000000\n"
        "PRECISION  = DOUBLE\n"
        "ENDIANNESS = LITTLE\n"
        "RELEASE = hkbm-solver\n"
        "/\n\n" % (str(deck.path.parent.resolve()), nsteps, nu)
    )
    if "units" in nml:
        out.append(
            _group("units", [(k, float(v)) for k, v in _lower(nml["units"]).items()])
        )
    Path(path).write_text("".join(out))


def write_run(outdir, suffix, deck, res, nt=11):
    """Write the GENE files of one solved k_y.  suffix: '.dat' or '_0001' etc."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ky = res["ky_ref"]
    z, phi, apar, bpar = _fields_on_z(res, deck, deck.nz0)
    gam, om = res["gamma_ref"], res["omega_ref"]
    w_model = complex(-om, gam)  # raw GENE fields evolve as exp(-i w_model t)
    rate = max(abs(om), abs(gam), 1e-6)
    dt = 0.05 / rate  # 20 samples per 1/|omega| (pyrokinetics differentiates in t)
    nsteps = nt - 1
    t = np.arange(nt) * dt
    nz = deck.nz0
    nrm = np.max(np.abs(phi))
    phi, apar, bpar = phi / nrm, apar / nrm, bpar / nrm
    fs = nz * 16
    with open(outdir / ("field" + suffix), "wb") as f:
        for tk in t:
            f.write(struct.pack("=idi", 8, float(tk), 8))
            a = np.exp(-1j * w_model * tk)
            for fld in (phi, apar, bpar):
                f.write(struct.pack("=i", fs))
                f.write(np.ascontiguousarray((fld * a).astype(np.complex128)).tobytes())
                f.write(struct.pack("=i", fs))
    with open(outdir / ("nrg" + suffix), "w") as f:
        for tk in t:
            f.write("%13.6f\n" % tk)
            for _ in range(2):
                f.write("".join("%12.4E" % 0.0 for _ in range(10)) + "\n")
    with open(outdir / ("omega" + suffix), "w") as f:
        f.write("%7.3f%11.4f%11.4f\n" % (ky, gam, om))
    write_parameters(outdir / ("parameters" + suffix), deck, ky, nsteps, dt)
    write_miller_dat(deck.miller, outdir / ("miller" + suffix))
    info = dict(
        ky=ky,
        gamma=gam,
        omega=om,
        converged=bool(res["converged"]),
        iterations=res.get("iterations"),
        residual=res.get("fres"),
        seconds=res.get("seconds"),
        ky_solver_units=res["ky_solver"],
        gamma_solver_units=res["gamma"],
        omega_solver_units=res["omega_gene"],
        r0=[res["r0"].real, res["r0"].imag],
        frac_phi2_within_0p5rad=res["frac05"],
        units=deck.units,
        warnings=deck.warnings,
        note="GENE sign: omega > 0 = ion diamagnetic direction; deck normalisation (c_ref/L_ref, rho_ref)",
    )
    (outdir / ("hkbm.json" if suffix == ".dat" else "hkbm%s.json" % suffix)).write_text(
        json.dumps(info, indent=1)
    )


def run(
    parameters_path, outdir=None, omega0=None, verbose=False, nz_geo=512, scan="auto"
):
    """Read a GENE parameters file, solve each k_y, write GENE output files.

    parameters_path: the GENE 'parameters' file (or the run directory containing it).
    outdir: where to write (default: the directory of the parameters file, where
    pyrokinetics' GENE reader looks for the output of Pyro(gk_file=<dir>/parameters)).
    omega0: optional seed (GENE sign omega + i gamma, deck units) for the first k_y; later k_y
    are seeded by the previous root (continuation in k_y).
    scan: 'auto' | True | False, see Deck.solve.
    Returns a list of dicts (ky, gamma, omega in the deck's units, converged, files)."""
    p = Path(parameters_path)
    if p.is_dir():
        p = p / "parameters"
    deck = Deck(p, nz_geo=nz_geo)
    outdir = Path(outdir) if outdir is not None else p.parent
    single = len(deck.kys) == 1
    results = []
    prev = None
    for n, ky in enumerate(deck.kys, start=1):
        seed = (
            omega0
            if n == 1
            else (
                complex(prev["omega_ref"], prev["gamma_ref"]) * ky / prev["ky_ref"]
                if prev is not None and prev["converged"]
                else None
            )
        )
        res = deck.solve(ky, omega0=seed, verbose=verbose, scan=scan)
        suffix = ".dat" if single else "_%04d" % n
        write_run(outdir, suffix, deck, res)
        if verbose:
            print(
                "k_y %.4g: gamma %+.5f omega %+.5f converged %s (%.1f s)"
                % (
                    ky,
                    res["gamma_ref"],
                    res["omega_ref"],
                    res["converged"],
                    res["seconds"],
                )
            )
        results.append(
            dict(
                ky=ky,
                gamma=res["gamma_ref"],
                omega=res["omega_ref"],
                converged=res["converged"],
                suffix=suffix,
                result=res,
            )
        )
        prev = res
    if not single:
        with open(outdir / "scan.log", "w") as f:
            f.write("#Run  |   kymin       /Eigenvalue1\n")
            for n, r in enumerate(results, start=1):
                f.write(
                    "%04d | %12.6E | %10.4f %10.4f\n"
                    % (n, r["ky"], r["gamma"], r["omega"])
                )
    return results
