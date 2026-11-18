"""
Prepare nonlinear (or linear) gyrokinetic simulations (default: GENE) on all or
selected radial flux surfaces directly from a Trinity3D (T3D) simulation output.

Usage examples
--------------
1. Self-contained demo (synthesises a mock T3D log from Pyrokinetics' template GEQDSK):
     python example_t3d_to_gene.py

2. Prepare nonlinear GENE inputs on selected T3D flux-tube surfaces (0-based indices):
     python example_t3d_to_gene.py --t3d-dir /path/to/t3d_run --tubes 5 16 21 23 --out-dir gene_nl_decks

3. Prepare nonlinear GENE inputs on ALL T3D flux-tube surfaces (e.g. all 24 tubes):
     python example_t3d_to_gene.py --t3d-dir /path/to/t3d_run --out-dir gene_nl_all24

4. Prepare linear GENE k_y scans on user-specified r/a surfaces:
     python example_t3d_to_gene.py --t3d-dir /path/to/t3d_run --rho 0.35 0.65 0.85 --linear --ky-scan 0.05 0.1 0.2 0.4
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from pyrokinetics import Pyro, PyroScan, template_dir
from pyrokinetics.kinetics.t3d import load_t3d_log_npy

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def _find_eq_file(t3d_dir: Path) -> Path:
    """Locate a GEQDSK equilibrium file in ``t3d_dir`` or sibling ``restart00``/parent directories."""
    search_dirs = [t3d_dir]
    if t3d_dir.parent != t3d_dir:
        r00 = t3d_dir.parent / "restart00"
        if r00.is_dir() and r00 != t3d_dir:
            search_dirs.append(r00)
        search_dirs.append(t3d_dir.parent)

    patterns = ("*.eqdsk_out", "*.geqdsk", "*.eqdsk", "g*")
    for d in search_dirs:
        for pat in patterns:
            matches = [
                p
                for p in sorted(d.glob(pat))
                if p.is_file() and not p.name.endswith((".in", ".py", ".out", ".err", ".src", ".npy"))
            ]
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Could not automatically locate a GEQDSK equilibrium file in {t3d_dir}; pass --eq-file explicitly."
    )


def _find_t3d_log(t3d_dir: Path) -> Path:
    matches = sorted(t3d_dir.glob("*.log.npy"))
    if not matches:
        matches = sorted(t3d_dir.glob("*.nc"))
    if not matches:
        raise FileNotFoundError(f"No T3D *.log.npy or *.nc file found in {t3d_dir}.")
    return matches[0]


def _read_t3d_gamma_e(t3d_dir: Path) -> Optional[np.ndarray]:
    """Read optional per-tube ExB shearing rate ``gammaE`` (in T3D/GS2 units ``v_t,D / a``) from ``*.in``."""
    in_files = [
        p
        for p in sorted(t3d_dir.glob("*.in"))
        if "gs2" not in p.name.lower() and "neo" not in p.name.lower()
    ]
    for in_path in in_files:
        try:
            with open(in_path, "rb") as f:
                cfg = tomllib.load(f)
        except Exception:
            continue
        for model in cfg.get("model", []):
            if "gammaE" in model:
                val = model["gammaE"]
                if isinstance(val, list):
                    return np.asarray(val, dtype=float)
                return np.asarray([float(val)], dtype=float)
    return None


def _apply_ken_checkpoint_override(log_file: Path, ken_npz: Path, tmp_dir: Path) -> Path:
    """Write a temporary T3D .log.npy whose final slice replaces (n_e, T_e, T_D) from a KEN state/ckpt .npz."""
    log = load_t3d_log_npy(log_file)
    z = np.load(ken_npz, allow_pickle=True)
    x = np.asarray(z["pt_x_chk"] if "pt_x_chk" in z.files else z["x"], dtype=float)

    grid_obj = log["grid"]
    grid_dict = grid_obj.__dict__ if hasattr(grid_obj, "__dict__") else grid_obj
    n_int = int(grid_dict["N_radial"]) - 1
    if len(x) != 3 * n_int:
        raise ValueError(f"Expected KEN state vector of length {3 * n_int}, got {len(x)}.")

    def _edge(key: str) -> float:
        v = log[key][-1]
        arr = v.__dict__["profile"] if hasattr(v, "__dict__") and "profile" in v.__dict__ else np.asarray(v)
        return float(arr[-1])

    ne = np.append(x[0:n_int], _edge("n_e"))
    pe = np.append(x[n_int : 2 * n_int], _edge("n_e") * _edge("T_e"))
    pD = np.append(x[2 * n_int : 3 * n_int], _edge("n_D") * _edge("T_D"))

    patched = {
        "t": [float(log["t"][-1])],
        "norms": log.get("norms", {}),
        "grid": grid_dict,
        "species_tags": ["e", "D"],
        "n_e": [ne],
        "n_D": [ne.copy()],
        "T_e": [pe / ne],
        "T_D": [pD / ne],
    }
    out_path = tmp_dir / "ken_override.log.npy"
    np.save(out_path, patched, allow_pickle=True)
    return out_path


def prepare_gk_from_t3d(
    kinetics_file: Path,
    eq_file: Path,
    out_dir: Path,
    *,
    t3d_dir: Optional[Path] = None,
    ken_npz: Optional[Path] = None,
    time_index: int = -1,
    time: Optional[float] = None,
    rho_surfaces: Optional[Sequence[float]] = None,
    tube_indices: Optional[Sequence[int]] = None,
    gk_code: str = "GENE",
    local_geometry: str = "Miller",
    nonlinear: bool = True,
    electromagnetic: bool = True,
    nkx: int = 128,
    nky: int = 24,
    kymin: float = 0.05,
    ky_linear: float = 0.2,
    ky_scan: Optional[Sequence[float]] = None,
    ntheta: int = 32,
    nenergy: int = 12,
    npitch: int = 16,
    delta_time: float = 0.005,
    max_time: float = 150.0,
    template_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Build nonlinear or linear GK input decks (default: GENE) across selected T3D flux surfaces.

    Returns a list of per-surface metadata dictionaries (also written to ``<out_dir>/manifest.json``).
    """
    kinetics_file = Path(kinetics_file)
    eq_file = Path(eq_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if t3d_dir is None:
        t3d_dir = kinetics_file.parent

    with tempfile.TemporaryDirectory() as tmp:
        eff_kinetics = (
            _apply_ken_checkpoint_override(kinetics_file, Path(ken_npz), Path(tmp))
            if ken_npz is not None
            else kinetics_file
        )
        log = load_t3d_log_npy(eff_kinetics) if eff_kinetics.suffix == ".npy" else None

        # Determine target r/a surfaces (default: T3D's flux-tube midpoints)
        midpoints: Optional[np.ndarray] = None
        if log is not None:
            g_obj = log["grid"]
            g_dict = g_obj.__dict__ if hasattr(g_obj, "__dict__") else g_obj
            if "midpoints" in g_dict:
                midpoints = np.asarray(g_dict["midpoints"], dtype=float)
            else:
                rho_nodes = np.asarray(g_dict["rho"], dtype=float)
                midpoints = 0.5 * (rho_nodes[:-1] + rho_nodes[1:])

        selected_tubes: List[Optional[int]] = []
        if rho_surfaces is not None:
            target_rhos = [float(r) for r in rho_surfaces]
            for r in target_rhos:
                if midpoints is not None and len(midpoints) > 0:
                    j = int(np.argmin(np.abs(midpoints - r)))
                    selected_tubes.append(j if abs(midpoints[j] - r) < 1e-3 else None)
                else:
                    selected_tubes.append(None)
        elif tube_indices is not None:
            if midpoints is None:
                raise ValueError("--tubes requires T3D midpoint grid metadata; pass --rho instead.")
            target_rhos = [float(midpoints[int(j)]) for j in tube_indices]
            selected_tubes = [int(j) for j in tube_indices]
        elif midpoints is not None:
            target_rhos = [float(r) for r in midpoints]
            selected_tubes = list(range(len(midpoints)))
        else:
            target_rhos = [0.3, 0.5, 0.7, 0.85]
            selected_tubes = [None] * len(target_rhos)

        gamma_e_arr = _read_t3d_gamma_e(t3d_dir)

        kinetics_kwargs: Dict[str, Any] = {}
        if time is not None:
            kinetics_kwargs["time"] = float(time)
        else:
            kinetics_kwargs["time_index"] = int(time_index)

        base_pyro = Pyro(
            eq_file=eq_file,
            eq_type="GEQDSK",
            kinetics_file=eff_kinetics,
            kinetics_type="T3D",
            kinetics_kwargs=kinetics_kwargs,
        )

    # Build monotonic r/a -> psi_n mapping from the loaded equilibrium
    psi_raw = base_pyro.eq["psi_n"].data
    rmin_raw = base_pyro.eq["r_minor"].data
    psi_eq = np.asarray(psi_raw.magnitude if hasattr(psi_raw, "magnitude") else psi_raw, dtype=float)
    rmin_eq = np.asarray(rmin_raw.magnitude if hasattr(rmin_raw, "magnitude") else rmin_raw, dtype=float)
    rho_eq = rmin_eq / rmin_eq[-1]
    _, uniq = np.unique(rho_eq, return_index=True)
    uniq = np.sort(uniq)
    psi_of_rho = InterpolatedUnivariateSpline(rho_eq[uniq], psi_eq[uniq], k=min(3, len(uniq) - 1))

    manifest: List[Dict[str, Any]] = []
    default_fname = "parameters_0001" if gk_code.upper() == "GENE" else f"input.{gk_code.lower()}"

    for surf_idx, (rho_val, tube_idx) in enumerate(zip(target_rhos, selected_tubes)):
        psi_n_val = float(np.clip(psi_of_rho(rho_val), 1e-5, 0.9999))
        pyro = base_pyro

        pyro.load_local(psi_n=psi_n_val, local_geometry=local_geometry)
        pyro.gk_code = gk_code.upper()

        norm_pyro = pyro.norms.pyrokinetics
        pyro.numerics.phi = True
        pyro.numerics.apar = bool(electromagnetic)
        pyro.numerics.bpar = bool(electromagnetic)
        pyro.numerics.ntheta = int(ntheta)
        pyro.numerics.nenergy = int(nenergy)
        pyro.numerics.npitch = int(npitch)
        pyro.numerics.delta_time = float(delta_time) * norm_pyro.lref / norm_pyro.vref
        pyro.numerics.max_time = float(max_time) * norm_pyro.lref / norm_pyro.vref

        if nonlinear:
            pyro.numerics.nonlinear = True
            pyro.numerics.nkx = int(nkx)
            pyro.numerics.nky = int(nky)
            pyro.numerics.ky = float(kymin) / norm_pyro.rhoref
            pyro.numerics.kx = 0.0 / norm_pyro.rhoref
        else:
            pyro.numerics.nonlinear = False
            pyro.numerics.nkx = 1
            pyro.numerics.nky = 1
            pyro.numerics.ky = float(ky_linear) / norm_pyro.rhoref

        # Map T3D's per-surface ExB shear (in GS2 units v_t,ref / a) to Pyrokinetics / GENE units
        gamma_t3d_val = 0.0
        if gamma_e_arr is not None:
            if tube_idx is not None and 0 <= tube_idx < len(gamma_e_arr):
                gamma_t3d_val = float(gamma_e_arr[tube_idx])
            elif midpoints is not None and len(gamma_e_arr) == len(midpoints):
                gamma_t3d_val = float(np.interp(rho_val, midpoints, gamma_e_arr))
        pyro.numerics.gamma_exb = (
            gamma_t3d_val * pyro.norms.gs2.vref / pyro.norms.gs2.lref
        ).to(norm_pyro)

        # Enforce local MHD alpha / beta' consistency with the (possibly KEN-overridden) kinetic pressure gradient
        pyro.enforce_consistent_beta_prime()

        tag = (
            f"tube_{tube_idx:02d}_rho_{rho_val:.4f}"
            if tube_idx is not None
            else f"surf_{surf_idx:02d}_rho_{rho_val:.4f}"
        )
        surf_dir = out_dir / tag
        surf_dir.mkdir(parents=True, exist_ok=True)
        deck_path = surf_dir / default_fname

        if not nonlinear and ky_scan is not None and len(ky_scan) > 0:
            scan = PyroScan(
                pyro,
                {"ky": list(ky_scan)},
                value_fmt=".3f",
                file_name=default_fname,
                base_directory=surf_dir,
            )
            scan.write()
        else:
            write_kw: Dict[str, Any] = {"file_name": deck_path}
            if template_file is not None:
                write_kw["template_file"] = Path(template_file)
            pyro.write_gk_file(**write_kw)

        sp_summary = {}
        for sp_name in pyro.local_species.names:
            sp_obj = pyro.local_species[sp_name]
            sp_summary[sp_name] = {
                "dens_norm": float(np.asarray(sp_obj.dens.m if hasattr(sp_obj.dens, "m") else sp_obj.dens)),
                "temp_norm": float(np.asarray(sp_obj.temp.m if hasattr(sp_obj.temp, "m") else sp_obj.temp)),
                "a_Ln": float(
                    np.asarray(
                        sp_obj.inverse_ln.to(norm_pyro.lref**-1).m
                        if hasattr(sp_obj.inverse_ln, "to")
                        else sp_obj.inverse_ln
                    )
                ),
                "a_LT": float(
                    np.asarray(
                        sp_obj.inverse_lt.to(norm_pyro.lref**-1).m
                        if hasattr(sp_obj.inverse_lt, "to")
                        else sp_obj.inverse_lt
                    )
                ),
            }

        entry = {
            "surface_dir": str(surf_dir),
            "deck_file": str(deck_path),
            "gk_code": gk_code.upper(),
            "nonlinear": bool(nonlinear),
            "tube_index": tube_idx,
            "rho": round(rho_val, 6),
            "psi_n": round(psi_n_val, 6),
            "q": round(float(pyro.local_geometry.q), 6),
            "shat": round(float(pyro.local_geometry.shat), 6),
            "beta_ref": float(pyro.norms.beta.m if hasattr(pyro.norms.beta, "m") else pyro.norms.beta),
            "gamma_E_t3d": gamma_t3d_val,
            "gamma_exb_gene": float(
                pyro.gk_input.data.get("external_contr", {}).get("exbrate", 0.0)
                if gk_code.upper() == "GENE" and pyro.gk_input is not None
                else 0.0
            ),
            "species": sp_summary,
        }
        manifest.append(entry)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _run_self_contained_demo() -> None:
    """Demonstrate T3D -> nonlinear GENE deck generation out-of-the-box using Pyrokinetics' template GEQDSK."""
    with tempfile.TemporaryDirectory(prefix="pyro_t3d_demo_") as tmp:
        tmp_path = Path(tmp)
        rho = np.linspace(0.04, 0.92, 13)
        midpoints = 0.5 * (rho[:-1] + rho[1:])
        mock_log = {
            "t": [0.0, 1.0],
            "norms": {"t_ref": 0.62},
            "grid": {"N_radial": len(rho), "rho": rho, "midpoints": midpoints},
            "species_tags": ["e", "D"],
            "n_e": [1.5 * (1.0 - 0.4 * rho**2), 1.8 * (1.0 - 0.35 * rho**2)],
            "n_D": [1.5 * (1.0 - 0.4 * rho**2), 1.8 * (1.0 - 0.35 * rho**2)],
            "T_e": [12.0 * (1.0 - 0.6 * rho**2), 13.0 * (1.0 - 0.58 * rho**2)],
            "T_D": [11.0 * (1.0 - 0.6 * rho**2), 12.2 * (1.0 - 0.58 * rho**2)],
        }
        log_path = tmp_path / "mock_t3d.log.npy"
        np.save(log_path, mock_log, allow_pickle=True)

        out_dir = tmp_path / "gene_nonlinear_surfaces"
        manifest = prepare_gk_from_t3d(
            kinetics_file=log_path,
            eq_file=template_dir / "test.geqdsk",
            out_dir=out_dir,
            tube_indices=[2, 6, 10],
            gk_code="GENE",
            nonlinear=True,
        )
        print(f"Prepared {len(manifest)} nonlinear GENE flux-tube directories:")
        for m in manifest:
            print(
                f"  tube={m['tube_index']:02d}  r/a={m['rho']:.3f}  psi_n={m['psi_n']:.3f}  "
                f"q={m['q']:.3f}  shat={m['shat']:.3f}  deck={Path(m['deck_file']).name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare nonlinear or linear GK (default: GENE) simulations on radial surfaces from T3D output."
    )
    parser.add_argument("--t3d-dir", type=Path, default=None, help="T3D run directory containing *.log.npy")
    parser.add_argument("--kinetics-file", type=Path, default=None, help="Explicit T3D .log.npy or .nc path")
    parser.add_argument("--eq-file", type=Path, default=None, help="Explicit GEQDSK equilibrium path")
    parser.add_argument("--ken-npz", type=Path, default=None, help="Optional KEN state.npz / ckpt.npz override")
    parser.add_argument("--out-dir", type=Path, default=Path("t3d_gene_decks"), help="Output directory")
    parser.add_argument("--time-index", type=int, default=-1, help="T3D time slice index (default: -1 = latest)")
    parser.add_argument("--time", type=float, default=None, help="Target T3D physical time in seconds")
    parser.add_argument("--rho", type=float, nargs="+", default=None, help="Target r/a surfaces")
    parser.add_argument("--tubes", type=int, nargs="+", default=None, help="Target 0-based T3D flux-tube indices")
    parser.add_argument("--gk-code", type=str, default="GENE", help="Target GK code (default: GENE)")
    parser.add_argument("--geometry", type=str, default="Miller", help="Local geometry model (Miller, MXH, ...)")
    parser.add_argument("--linear", action="store_true", help="Prepare linear runs instead of nonlinear")
    parser.add_argument("--ky-scan", type=float, nargs="+", default=None, help="Optional linear k_y scan values")
    parser.add_argument("--nkx", type=int, default=128, help="Nonlinear radial grid size nx0 (default: 128)")
    parser.add_argument("--nky", type=int, default=24, help="Nonlinear binormal mode count nky0 (default: 24)")
    parser.add_argument("--kymin", type=float, default=0.05, help="Minimum k_y * rho_ref (default: 0.05)")
    args = parser.parse_args()

    if args.t3d_dir is None and args.kinetics_file is None:
        _run_self_contained_demo()
        return

    t3d_dir = args.t3d_dir if args.t3d_dir is not None else args.kinetics_file.parent
    kinetics_file = args.kinetics_file if args.kinetics_file is not None else _find_t3d_log(t3d_dir)
    eq_file = args.eq_file if args.eq_file is not None else _find_eq_file(t3d_dir)

    manifest = prepare_gk_from_t3d(
        kinetics_file=kinetics_file,
        eq_file=eq_file,
        out_dir=args.out_dir,
        t3d_dir=t3d_dir,
        ken_npz=args.ken_npz,
        time_index=args.time_index,
        time=args.time,
        rho_surfaces=args.rho,
        tube_indices=args.tubes,
        gk_code=args.gk_code,
        local_geometry=args.geometry,
        nonlinear=not args.linear,
        ky_scan=args.ky_scan,
        nkx=args.nkx,
        nky=args.nky,
        kymin=args.kymin,
    )
    print(f"Wrote {len(manifest)} {args.gk_code.upper()} surface decks to {args.out_dir} (manifest.json saved).")


if __name__ == "__main__":
    main()
