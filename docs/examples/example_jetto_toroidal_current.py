r"""
Compare the pyrokinetics toroidal current decomposition against a JETTO run.

JETTO carries more current information than SCENE or TRANSP, so it is a good
target for checking every piece of the decomposition at once:

    JSP key   meaning                                  pyrokinetics counterpart
    -------   --------------------------------------   ------------------------
    JZ        total toroidal current density [A/m^2]   Jphi_eff, on rho_tor
    JZBS      bootstrap toroidal current density       Jphi_bs_eff, on rho_tor
    JZNB      NBI driven toroidal current density      part of Jphi_ext_eff
    JPBS      <Jbs.B> / B0 [A/m^2]                     JbsdotB / B0
    CUR       cumulative total plasma current [A]      radial integral below -> Ip
    CUBS      cumulative bootstrap current [A]         radial integral below -> Ip_bs
    CUNB      cumulative NBI current [A]               part of Ip_ext
    FT        trapped fraction                         trapped_fraction
    RHO       JETTO radial coordinate [m]              see below

The radial coordinate
---------------------
JETTO's Jz is the same *kind* of quantity as our Jphi_eff -- both are

    (1 / 2 pi x) dIp/dx

(Tholerus, "Notes on the calculation of the JETTO current", eqs 43 and 51) --
but they use different radial labels x:

    pyrokinetics   x = r,     the minor radius (metric.rho)     [the default]
    JETTO          x = rho_J = sqrt(Psi_tor / (pi B_geo))    (Tholerus eq 1)

They differ by an exact factor obtained from the chain rule:

    Jz = Jphi_eff * (r / rho_J) * (dr / drho_J) = Jphi_eff * d(r^2)/d(rho_J^2)

This is *not* a constant: on a strongly shaped spherical tokamak it runs from
about 0.45 near the axis to 0.24 near the edge, so a single number will not do.

The library applies it. Passing ``radial_coordinate="rho_tor"`` to a
``BootstrapModel`` builds Jphi_eff, and dIp/drho with it, on rho_J directly, so
nothing is rescaled in this script. Substituting Psi_tor into the chain rule
above collapses the factor to

    d(r^2)/d(rho_J^2) = 2 pi r B_geo (dr/dpsi) / q

in which Psi_tor cancels, leaving only single-flux-surface quantities -- so no
radial scan is needed to build it. The model exposes it as ``label_factor``, and
the label itself as ``rho_label``.

The conversion is pinned by B_geo, because Jphi_eff scales as 1/c^2 under
rho -> c rho: it is a change of convention, not merely a relabelling. B_geo is
taken from ``Equilibrium.B_0``, which is why "rho_tor" needs a global
equilibrium -- pyrokinetics' own Bref = F / Rmaj is a per-flux-surface quantity
and is not B_geo. The check below compares that B_0 against JETTO's own, since a
wrong B_geo would silently rescale every comparison here.

Note that pyrokinetics' own ``Equilibrium.rho_tor`` is built from the same
Psi_tor, just normalised differently:

    pyro   rho_tor = sqrt(Psi_tor / Psi_tor(LCFS))     dimensionless, 0 -> 1
    JETTO  rho_J   = sqrt(Psi_tor / (pi B_geo))        metres

so rho_J = a_tor * rho_tor with a_tor = sqrt(psi_tor(1.0) / (pi B_geo)).

The radially *integrated* currents need no such conversion, since Ip is
independent of the radial label used to parametrise it.

Why the radial integral is done here, and not with integrate_toroidal_current
------------------------------------------------------------------------------
This script merges the fast and impurity species into the main ion on every flux
surface. ``neoclassical.integrate_toroidal_current`` runs its own radial scan,
calling ``load_local`` on each surface, and ``load_local`` rebuilds
``local_species`` from the global ``Kinetics`` -- so the merge would be silently
discarded, and the integrated currents would describe the full JETTO species set
while every density plotted here describes the merged one. To keep the whole
script on a single plasma, the ``dIp_*_drho`` values are collected during the one
scan below and integrated afterwards. The formula is the library function's,
verified to agree with it bit for bit on an unmerged case, and it costs half as
much because each surface is only fitted once.

How much the merge changes the answer is equilibrium dependent. It enters through
Z_eff and hence through <Jbs.B>; since the external current is the residual
<J.B> - <Jbs.B>, the bootstrap and external parts move by nearly equal and
opposite amounts and the total is barely affected. On the jun1126 case the
integrated bootstrap moves by 1%, the remainder by -3%, and the total by 0.08%,
while the Ampere law column does not move at all, being built from geometry
alone. Local shifts can be much larger -- on an SPR68 case <Jbs.B> moves by -17%
at psi_n = 0.3 and +5.6% at 0.85 -- so do not rely on the cancellation.

Point ``BASE_DIR`` at a JETTO run directory holding ``jetto.jsp`` and
``jetto.eqdsk_out``. It falls back to the shipped templates so the script runs
out of the box, but note the template equilibrium is *not* the one that
produced the template JSP, so the fallback numbers are not meaningful --
use it only to check the script executes.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # write PDFs, no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jetto_tools.binary import read_binary_file  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402

from pyrokinetics import Pyro, template_dir  # noqa: E402
from pyrokinetics.diagnostics.neoclassical import Redl2021, Sauter1999  # noqa: E402
from pyrokinetics.units import ureg  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# JETTO run directory holding jetto.jsp and jetto.eqdsk_out. Override without
# editing this file by passing it as the first argument, or by setting
# JETTO_RUN_DIR in the environment:
#     python example_jetto_toroidal_current.py /path/to/jetto/run
BASE_DIR = "."
if len(sys.argv) > 1:
    BASE_DIR = sys.argv[1]
elif os.environ.get("JETTO_RUN_DIR"):
    BASE_DIR = os.environ["JETTO_RUN_DIR"]

TIME_INDEX = -1

# Take every downsize'th flux surface. Cost is dominated by the MXH fit inside
# load_local (~1-3 s per surface), not by the current calculation (~0.1 s).
# A coarse grid also degrades the radial integral, so if the integrated total
# disagrees with the Ampere law value, refine this before suspecting physics.
DOWNSIZE = int(os.environ.get("JETTO_DOWNSIZE", 1))

# Restrict the radial scan: near the axis the inverse aspect ratio tends to
# zero and the trapped fraction degenerates, near the separatrix the MXH fit
# degrades. Both would quietly corrupt the radial integral.
PSI_N_MIN = float(os.environ.get("JETTO_PSI_N_MIN", 0.05))
PSI_N_MAX = float(os.environ.get("JETTO_PSI_N_MAX", 0.98))

# Prefix for the written PDFs, so successive runs can be told apart without
# overwriting each other, e.g. JETTO_PLOT_PREFIX=merged_
PLOT_PREFIX = os.environ.get("JETTO_PLOT_PREFIX", "")

# Merge fast and impurity species into the main ion, as a JETTO run usually
# carries alphas. Everything not in TO_KEEP is absorbed into MERGE_INTO.
MERGE_INTO = "deuterium"
TO_KEEP = ["deuterium", "electron"]

jsp_file = os.path.join(BASE_DIR, "jetto.jsp")
eqdsk_file = os.path.join(BASE_DIR, "jetto.eqdsk_out")

USING_FALLBACK = False
if not os.path.exists(jsp_file):
    jsp_file = template_dir / "jetto.jsp"
    USING_FALLBACK = True
if not os.path.exists(str(eqdsk_file)):
    eqdsk_file = template_dir / "test.geqdsk"
    USING_FALLBACK = True

FALLBACK_BANNER = (
    "\n"
    + "!" * 74
    + "\n!! Running on the shipped TEMPLATE files, because BASE_DIR did not hold"
    "\n!! both jetto.jsp and jetto.eqdsk_out."
    "\n!!"
    "\n!! The template equilibrium is NOT the one that produced the template"
    "\n!! JSP, so every comparison below is meaningless. This only proves the"
    "\n!! script executes. Set BASE_DIR to a real JETTO run.\n" + "!" * 74
)

if USING_FALLBACK:
    print(FALLBACK_BANNER)

# --------------------------------------------------------------------------
# JETTO reference data
# --------------------------------------------------------------------------
jsp = read_binary_file(jsp_file)
ti = TIME_INDEX


def jsp_profile(key):
    """Profile at the chosen time index"""
    return np.asarray(jsp[key])[ti, :]


# Normalised poloidal flux, from the poloidal flux profile itself
jetto_psi = jsp_profile("PSI")
jetto_psi = jetto_psi - jetto_psi[0]
jetto_psin = jetto_psi / jetto_psi[-1]

# Cell centred grid (JZ, CUR, ...) and cell boundary grid (JP, JPBS, FT)
x = np.asarray(jsp["XVEC1"])[0, :]
xb = np.asarray(jsp["XVEC2"])[0, :]

jetto_jz = jsp_profile("JZ")  # total toroidal current density
jetto_jzbs = jsp_profile("JZBS")  # bootstrap
jetto_cur = jsp_profile("CUR")  # cumulative total current
jetto_cubs = jsp_profile("CUBS")  # cumulative bootstrap current
jetto_cunb = jsp_profile("CUNB")  # cumulative NBI current

# These live on the boundary grid, put them on the centred grid
jetto_jpbs = np.interp(x, xb, jsp_profile("JPBS"))
jetto_ftrap = np.interp(x, xb, jsp_profile("FT"))

# JETTO's own radial coordinate, rho_J = sqrt(Psi_tor / (pi B_geo)), in metres
jetto_rho = jsp_profile("RHO") if "RHO" in jsp else None

print(f"\nJETTO run: {jsp_file}")
print(f"  total plasma current  CUR  = {jetto_cur[-1] / 1e6:8.4f} MA")
print(f"  bootstrap current     CUBS = {jetto_cubs[-1] / 1e6:8.4f} MA")
print(f"  NBI current           CUNB = {jetto_cunb[-1] / 1e6:8.4f} MA")
print(f"  bootstrap fraction         = {jetto_cubs[-1] / jetto_cur[-1]:8.4f}")

# --------------------------------------------------------------------------
# Pyrokinetics
# --------------------------------------------------------------------------
pyro = Pyro(
    eq_file=eqdsk_file,
    eq_type="GEQDSK",
    eq_kwargs={"psi_n_lcfs": 0.999},
    kinetics_file=jsp_file,
    kinetics_type="JETTO",
    kinetics_kwargs={"time_index": TIME_INDEX},
)

# --------------------------------------------------------------------------
# B_geo and R_geo: JETTO vs pyrokinetics
# --------------------------------------------------------------------------
# a_tor below, and radial_coordinate="rho_tor" inside the library, both depend
# on B_geo, so a wrong B_geo silently rescales the whole
# comparison. Check it against JETTO's own value before relying on it.
eq = pyro.eq
B_geo_pyro = abs(eq.B_0.to("tesla").m)

# R_geo is the centre of the midplane, (R_max + R_min) / 2 on the boundary
lcfs_R = eq.flux_surface(0.999)["R"].data
lcfs_R = np.asarray(getattr(lcfs_R, "magnitude", lcfs_R), dtype=float)
R_geo = 0.5 * (lcfs_R.max() + lcfs_R.min())
R_major_lcfs = eq.R_major(0.999).to("m").m

# The field actually on the boundary, which includes plasma diamagnetism. This
# is NOT B_geo: Tholerus eq (1) specifies the *vacuum* field, so a difference
# here of order beta_poloidal is expected and is not an inconsistency.
B_on_surface = abs(eq.F(0.999).to("m*tesla").m) / R_geo

jss_file = os.path.join(BASE_DIR, "jetto.jss")
B_geo_jetto = None
if os.path.exists(jss_file):
    jss = read_binary_file(jss_file)
    if "B0" in jss:
        B_geo_jetto = abs(np.asarray(jss["B0"]).ravel()[TIME_INDEX])

print("\nB_geo and R_geo, JETTO vs pyrokinetics")
if B_geo_jetto is not None:
    print(f"  B_geo  JETTO jetto.jss B0              = {B_geo_jetto:8.4f} T")
print(f"  B_geo  pyro eq.B_0 (eqdsk bcentr)      = {B_geo_pyro:8.4f} T")
print(f"  F(LCFS)/R_geo, on-surface not vacuum   = {B_on_surface:8.4f} T")
print(f"  R_geo = (Rmax + Rmin) / 2              = {R_geo:8.4f} m")
print(f"  eq.R_major(LCFS)                       = {R_major_lcfs:8.4f} m")

if B_geo_jetto is not None and abs(B_geo_pyro / B_geo_jetto - 1) > 0.01:
    print("  WARNING: pyro and JETTO disagree on B_geo by more than 1%.")
    print("           a_tor, and the library's rho_tor label, both use B_geo.")
if abs(R_geo / R_major_lcfs - 1) > 0.01:
    print("  NOTE: R_geo differs from R_major(LCFS). eq.B_0 is the vacuum field")
    print("        at the eqdsk rcentr, which only equals the field at R_geo")
    print("        when rcentr is close to R_geo.")

# JETTO's rho if the JSP did not carry it: rho_J = a_tor * rho_tor
if jetto_rho is None:
    B_geo = B_geo_jetto if B_geo_jetto is not None else B_geo_pyro
    a_tor = np.sqrt(abs(eq.psi_tor(1.0).to("weber").m) / (np.pi * B_geo))
    jetto_rho = a_tor * eq.rho_tor(jetto_psin)
    print(f"\n  RHO absent from the JSP, reconstructed with a_tor = {a_tor:.4f} m")
else:
    B_geo = B_geo_jetto if B_geo_jetto is not None else B_geo_pyro
    a_tor = np.sqrt(abs(eq.psi_tor(1.0).to("weber").m) / (np.pi * B_geo))
    reconstructed = a_tor * eq.rho_tor(jetto_psin)
    err = np.max(np.abs(reconstructed[1:] - jetto_rho[1:]) / jetto_rho[1:])
    print(f"\n  a_tor = sqrt(psi_tor(LCFS) / (pi B_geo)) = {a_tor:8.4f} m")
    print(f"  JETTO RHO at the LCFS                    = {jetto_rho[-1]:8.4f} m")
    print(f"  max relative error of a_tor * rho_tor    = {err:8.4f}")

psi_ns = jetto_psin[::DOWNSIZE]

jphi_fsa = psi_ns * 0.0
jphi_fsa_bs = psi_ns * 0.0
jphi_fsa_ext = psi_ns * 0.0
jphi_fsa_psdia = psi_ns * 0.0

jphi_area = psi_ns * 0.0

# Built on rho_J, not on r, via radial_coordinate="rho_tor" below
jphi_eff = psi_ns * 0.0
jphi_eff_bs = psi_ns * 0.0
jphi_eff_ext = psi_ns * 0.0
jphi_eff_psdia = psi_ns * 0.0
label_factor = psi_ns * 0.0

jbsdotb_b0 = psi_ns * 0.0
pyro_ftrap = psi_ns * 0.0
sauter_jbsdotb_b0 = psi_ns * 0.0
valid = np.zeros(len(psi_ns), dtype=bool)

# Collected during the same scan so the radial integral below describes the
# merged plasma. rho is rho_J in metres, the label dIp_drho is built on and the
# integration variable.
rho_m = psi_ns * 0.0
dIp_drho_bs = psi_ns * 0.0
dIp_drho_ext = psi_ns * 0.0
dIp_drho_psdia = psi_ns * 0.0
Ip_ampere = psi_ns * 0.0


def merge_to_main_ion(p):
    """Absorb every species outside TO_KEEP into MERGE_INTO, in place.

    LocalSpecies is rebuilt from the global kinetics by every load_local, so this
    has to be re-applied after each one. Failures propagate to the caller, which
    skips the surface: one whose merge did not complete must not end up in the
    same arrays as the good ones.
    """
    merge = [s for s in p.local_species.names if s not in TO_KEEP]
    if merge:  # nothing to do on a run that already carries only e + main ion
        p.local_species.merge_species(
            MERGE_INTO,
            merge,
            keep_base_species_z=True,
            keep_base_species_mass=True,
            update_zeff=True,
        )
    p.local_species.enforce_quasineutrality(MERGE_INTO)
    p.enforce_consistent_beta_prime()


B0 = np.abs(pyro.eq.B_0)
print(f"\nB0 from equilibrium = {B0:~.4g}")

print("\nScanning flux surfaces ...")
for i, psi_n in enumerate(psi_ns):
    if not (PSI_N_MIN < psi_n < PSI_N_MAX):
        continue
    try:
        pyro.load_local(psi_n=psi_n, local_geometry="MXH")
    except Exception as exc:
        print(f"  psi_n={psi_n:.3f} skipped ({type(exc).__name__})")
        continue

    try:
        merge_to_main_ion(pyro)
    except Exception as exc:
        # Skip rather than fall through, so a partially merged surface cannot
        # join the good ones, and rather than abort, so one bad surface does not
        # throw away the whole scan. valid[i] stays False either way.
        print(f"  psi_n={psi_n:.3f} skipped, merge failed ({type(exc).__name__})")
        continue

    # radial_coordinate="rho_tor" puts Jphi_eff and dIp/drho straight onto
    # JETTO's rho_J, so nothing below needs rescaling. See the module docstring.
    redl = Redl2021(pyro, radial_coordinate="rho_tor")

    # A third candidate convention: the AREA weighted toroidal current density,
    # dIp/dA. <J_phi> is volume weighted (2 pi R dl / |grad rho|) whereas an area
    # integral needs dl / |grad rho|, so the two differ by the correlation of
    # J_phi with R over the surface -- large at tight aspect ratio. dIp/dA does
    # not depend on the radial label, but its two halves do: dIp/drho is on
    # rho_J while the area derivative is on r, so put them on the same label.
    dA_dr = pyro.local_geometry.get_flux_surface_area_volume_derivatives()[0]
    drho_J_dr = pyro.metric_terms.rho / (redl.label_factor * redl.rho_label)
    jphi_area[i] = (redl.dIp_drho * drho_J_dr / dA_dr).to("ampere / m**2").m

    jphi_fsa[i] = redl.Jphi_fsa.to("ampere / m**2").m
    jphi_fsa_bs[i] = redl.Jphi_bs_fsa.to("ampere / m**2").m
    jphi_fsa_ext[i] = redl.Jphi_ext_fsa.to("ampere / m**2").m
    jphi_fsa_psdia[i] = redl.Jphi_psdia_fsa.to("ampere / m**2").m

    jphi_eff[i] = redl.Jphi_eff.to("ampere / m**2").m
    jphi_eff_bs[i] = redl.Jphi_bs_eff.to("ampere / m**2").m
    jphi_eff_ext[i] = redl.Jphi_ext_eff.to("ampere / m**2").m
    jphi_eff_psdia[i] = redl.Jphi_psdia_eff.to("ampere / m**2").m
    label_factor[i] = redl.label_factor.m

    # The pieces the radial integral needs, from this same merged surface
    rho_m[i] = redl.rho_label.to("meter").m
    dIp_drho_bs[i] = redl.dIp_bs_drho.to("ampere / meter").m
    dIp_drho_ext[i] = redl.dIp_ext_drho.to("ampere / meter").m
    dIp_drho_psdia[i] = redl.dIp_psdia_drho.to("ampere / meter").m
    Ip_ampere[i] = redl.Ip.to("ampere").m

    jbsdotb_b0[i] = (redl.JbsdotB / B0).to("ampere / m**2").m
    pyro_ftrap[i] = redl.trapped_fraction

    sauter = Sauter1999(pyro)
    sauter_jbsdotb_b0[i] = (sauter.JbsdotB / B0).to("ampere / m**2").m

    valid[i] = True

if not valid.any():
    raise RuntimeError("No flux surface loaded successfully, check the inputs")

# --------------------------------------------------------------------------
# Jphi_eff is already on JETTO's radial label
# --------------------------------------------------------------------------
# Redl2021 was built with radial_coordinate="rho_tor", so Jphi_eff is already
# (1 / 2 pi rho_J) dIp/drho_J and compares with JZ directly. Dividing the
# library's own label_factor back out recovers the r-based value, which is kept
# only to show below what the label choice is worth.
jz_on_scan = np.interp(psi_ns, jetto_psin, jetto_jz)

jphi_eff_on_r = np.divide(
    jphi_eff, label_factor, out=np.zeros_like(jphi_eff), where=label_factor != 0
)

ratio_jz = np.abs(jphi_eff[valid]) / np.abs(jz_on_scan[valid])
ratio_eff_raw = np.abs(jphi_eff_on_r[valid]) / np.abs(jz_on_scan[valid])
ratio_fsa = np.abs(jphi_fsa[valid]) / np.abs(jz_on_scan[valid])
ratio_area = np.abs(jphi_area[valid]) / np.abs(jz_on_scan[valid])

print("\nToroidal current density against JETTO's JZ")
print(f"  radial label factor d(r^2)/d(rho_J^2): {label_factor[valid].min():.3f}")
print(f"                                     to  {label_factor[valid].max():.3f}")
print(
    f"\n  |Jphi_eff on rho_tor|/ |JZ| : "
    f"mean {ratio_jz.mean():6.3f}  std {ratio_jz.std():6.3f}   <- the comparison"
)
print("\n  for reference, without the radial label conversion:")
print(
    f"    |Jphi_eff on r| / |JZ|    : "
    f"mean {ratio_eff_raw.mean():6.3f}  std {ratio_eff_raw.std():6.3f}"
)
print(
    f"    |<J_phi>|    / |JZ|       : "
    f"mean {ratio_fsa.mean():6.3f}  std {ratio_fsa.std():6.3f}"
)
print(
    f"    |<J_phi>_area|/|JZ|       : "
    f"mean {ratio_area.mean():6.3f}  std {ratio_area.std():6.3f}"
)
print(
    "    (<J_phi>_area is dIp/dS, the note's J_tor: a different, also valid\n"
    "     quantity, related to Jz by eq 63, not an alternative to it)"
)

scatter = ratio_jz.std() / max(ratio_jz.mean(), 1e-30)
offset = abs(ratio_jz.mean() - 1)
if scatter > 0.1 or offset > 0.15:
    print(
        f"\n  WARNING: the rho_tor ratio is not flat and near 1 "
        f"(offset {offset:.2f}, relative scatter {scatter:.2f})."
    )
    print("           Check the equilibrium really is the one that made this JSP.")
else:
    print(
        f"\n  Agreement on rho_tor: {100 * offset:.1f}% offset, "
        f"{scatter:.3f} scatter"
    )

# --------------------------------------------------------------------------
# Radially integrated currents vs JETTO CUR / CUBS
# --------------------------------------------------------------------------
# No radial label conversion is needed here. Ip is the current enclosed by a
# flux surface, which does not depend on the coordinate used to label it, so
# these compare directly with JETTO however each code parametrises the radius.
#
# See the module docstring: this is the same integral
# pyrokinetics' integrate_toroidal_current performs, but evaluated on the scan
# above, which applied the merge. Calling that function instead would re-load
# every surface and silently drop the merge, integrating a different plasma
# from the one every figure above describes.
#
# The innermost surface contributes pi rho_0^2 Jphi_eff(rho_0), i.e. the current
# density is taken as constant between the axis and the first surface, exactly
# as integrate_toroidal_current does.
rho_scan = rho_m[valid] * ureg.meter


def enclosed(dIp_drho):
    derivative = dIp_drho[valid] * ureg.ampere / ureg.meter
    axis = 0.5 * rho_scan[0] * derivative[0]
    return (
        cumulative_trapezoid(derivative.m, rho_scan.m, initial=0.0) * ureg.ampere + axis
    )


integrated = {
    "psi_n": psi_ns[valid],
    "rho": rho_scan,
    "Ip_ampere": Ip_ampere[valid] * ureg.ampere,
    "Ip_bs": enclosed(dIp_drho_bs),
    "Ip_ext": enclosed(dIp_drho_ext),
    "Ip_psdia": enclosed(dIp_drho_psdia),
}
integrated["Ip"] = integrated["Ip_bs"] + integrated["Ip_ext"] + integrated["Ip_psdia"]

psi_n_edge = integrated["psi_n"][-1]
jetto_cur_edge = np.interp(psi_n_edge, jetto_psin, jetto_cur)
jetto_cubs_edge = np.interp(psi_n_edge, jetto_psin, jetto_cubs)
jetto_cunb_edge = np.interp(psi_n_edge, jetto_psin, jetto_cunb)

# JETTO tracks bootstrap and each drive separately but has no key for the
# ohmic or the Pfirsch-Schlueter + diamagnetic current, so everything that is
# not bootstrap or an explicit drive is the remainder CUR - CUBS - CUNB. Group
# our external and ps+dia together to compare like with like.
jetto_other_edge = jetto_cur_edge - jetto_cubs_edge - jetto_cunb_edge
ours_other = (integrated["Ip_ext"][-1] + integrated["Ip_psdia"][-1]).to("MA").m

# Signs follow each code's own current convention, compare magnitudes
print(f"\nCurrent enclosed by psi_n = {psi_n_edge:.3f}, pyro vs JETTO")
print("  (magnitudes, since the two use different current sign conventions)\n")
print(f"  {'':<36} {'pyro':>9} {'JETTO':>9} {'ratio':>7}")


def compare(label, mine, theirs):
    ratio = abs(mine) / abs(theirs) if theirs else float("nan")
    theirs_s = f"{abs(theirs):9.4f}" if theirs else f"{'n/a':>9}"
    ratio_s = f"{ratio:7.3f}" if theirs else f"{'':>7}"
    print(f"  {label:<36} {abs(mine):9.4f} {theirs_s} {ratio_s}")


compare("bootstrap", integrated["Ip_bs"][-1].to("MA").m, jetto_cubs_edge / 1e6)
compare("NBI", 0.0, jetto_cunb_edge / 1e6)
compare("ohmic + ps/dia (the remainder)", ours_other, jetto_other_edge / 1e6)
compare("total", integrated["Ip"][-1].to("MA").m, jetto_cur_edge / 1e6)
print(
    f"\n  of our remainder: external (aux+ohmic) "
    f"{abs(integrated['Ip_ext'][-1].to('MA').m):.4f} MA, "
    f"ps+dia {abs(integrated['Ip_psdia'][-1].to('MA').m):.4f} MA"
)

amp = integrated["Ip_ampere"][-1].to("MA").m
print(
    f"  {'Ampere law, same surface':<34} {amp:10.4f}   {'':>10}    (independent check)"
)

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
v = valid


def finish(name, title, ylabel=r"$A\,m^{-2}$"):
    plt.title(title)
    plt.xlabel(r"$\psi_N$")
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(PLOT_PREFIX + name)
    plt.close()
    print(f"  wrote {PLOT_PREFIX}{name}")


print("\nWriting plots ...")

# Trapped fraction
plt.figure()
plt.plot(jetto_psin, jetto_ftrap, lw=2, label="JETTO FT")
plt.plot(psi_ns[v], pyro_ftrap[v], ls="--", lw=2, label="Pyro (Redl 2021)")
finish("jetto_trapped_fraction.pdf", "Trapped particle fraction", r"$f_{trap}$")

# Bootstrap parallel current
plt.figure()
plt.plot(jetto_psin, np.abs(jetto_jpbs), lw=2, label="JETTO JPBS")
plt.plot(psi_ns[v], np.abs(jbsdotb_b0[v]), ls="--", lw=2, label="Redl (2021) Pyro")
plt.plot(
    psi_ns[v], np.abs(sauter_jbsdotb_b0[v]), ls=":", lw=2, label="Sauter (1999) Pyro"
)
finish("jetto_jbsdotb.pdf", r"$\langle J_{bs} \cdot B\rangle / B_0$")

# Toroidal current density on JETTO's radial label: total and bootstrap
plt.figure()
plt.plot(jetto_psin, np.abs(jetto_jz), lw=2, color="k", label="JETTO JZ (total)")
plt.plot(
    jetto_psin, np.abs(jetto_jzbs), lw=2, color="C2", label="JETTO JZBS (bootstrap)"
)
plt.plot(
    psi_ns[v],
    np.abs(jphi_eff[v]),
    ls="--",
    lw=2,
    color="C0",
    label=r"Pyro $J_\phi^{eff}$ (on $\rho_J$)",
)
plt.plot(
    psi_ns[v],
    np.abs(jphi_eff_bs[v]),
    ls="-.",
    lw=2,
    color="C3",
    label=r"Pyro bootstrap (on $\rho_J$)",
)
finish("jetto_jz_comparison.pdf", "Toroidal current density vs JETTO (JETTO label)")

# The same comparison without the conversion, showing what the radial label
# choice is worth: these curves are the ones that sit a factor of a few off
plt.figure()
plt.plot(jetto_psin, np.abs(jetto_jz), lw=2, color="k", label="JETTO JZ")
plt.plot(
    psi_ns[v], np.abs(jphi_fsa[v]), ls="--", lw=2, label=r"Pyro $\langle J_\phi\rangle$"
)
plt.plot(
    psi_ns[v],
    np.abs(jphi_eff_on_r[v]),
    ls="-.",
    lw=2,
    label=r"Pyro $J_\phi^{eff}$ (on $r$)",
)
plt.plot(
    psi_ns[v], np.abs(jphi_eff[v]), lw=2, color="C0", label=r"Pyro $J_z$ (on $\rho_J$)"
)
finish("jetto_jphi_conventions.pdf", "Effect of the radial label choice")

# Decomposition, effective convention
plt.figure()
plt.plot(psi_ns[v], jphi_eff[v], lw=2, color="k", label="Total")
plt.plot(psi_ns[v], jphi_eff_bs[v], ls="--", lw=2, label="Bootstrap")
plt.plot(psi_ns[v], jphi_eff_ext[v], ls="--", lw=2, label="External (aux + ohmic)")
plt.plot(psi_ns[v], jphi_eff_psdia[v], ls="--", lw=2, label="Pfirsch-Schluter + dia")
plt.plot(
    psi_ns[v],
    jphi_eff_bs[v] + jphi_eff_ext[v] + jphi_eff_psdia[v],
    ls=":",
    lw=2,
    color="C3",
    label="Sum of components",
)
finish("jetto_jphi_eff_decomposition.pdf", r"$J_\phi^{eff}$ decomposition (Redl 2021)")

# Decomposition, flux surface average convention
plt.figure()
plt.plot(psi_ns[v], jphi_fsa[v], lw=2, color="k", label="Total")
plt.plot(psi_ns[v], jphi_fsa_bs[v], ls="--", lw=2, label="Bootstrap")
plt.plot(psi_ns[v], jphi_fsa_ext[v], ls="--", lw=2, label="External (aux + ohmic)")
plt.plot(psi_ns[v], jphi_fsa_psdia[v], ls="--", lw=2, label="Pfirsch-Schluter + dia")
finish("jetto_jphi_fsa_decomposition.pdf", r"$\langle J_\phi \rangle$ decomposition")

# Cumulative current
plt.figure()
plt.plot(
    jetto_psin, np.abs(jetto_cur) / 1e6, lw=2, color="k", label="JETTO CUR (total)"
)
plt.plot(
    jetto_psin,
    np.abs(jetto_cubs) / 1e6,
    lw=2,
    color="C2",
    label="JETTO CUBS (bootstrap)",
)
plt.plot(
    integrated["psi_n"],
    np.abs(integrated["Ip"].to("MA").m),
    ls="--",
    lw=2,
    label="Pyro total",
)
plt.plot(
    integrated["psi_n"],
    np.abs(integrated["Ip_bs"].to("MA").m),
    ls="--",
    lw=2,
    label="Pyro bootstrap",
)
plt.plot(
    integrated["psi_n"],
    np.abs(integrated["Ip_ampere"].to("MA").m),
    ls=":",
    lw=2,
    color="C3",
    label="Pyro Ampere law",
)
finish("jetto_enclosed_current.pdf", "Enclosed toroidal current", "MA")

if USING_FALLBACK:
    # Repeat it here: the opening banner is easily lost above the pyrokinetics
    # warnings that the equilibrium and geometry fits emit during the scan.
    print(FALLBACK_BANNER)

print("\nDone.")
