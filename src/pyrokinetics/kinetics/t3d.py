"""Kinetics reader for Trinity3D (T3D) transport simulation outputs (.log.npy and NetCDF)."""

from __future__ import annotations

import pickle
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Optional, Tuple

import netCDF4 as nc
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from ..constants import (
    deuterium_mass,
    electron_mass,
    hydrogen_mass,
    tritium_mass,
)
from ..equilibrium import Equilibrium
from ..file_utils import FileReader
from ..species import Species
from ..typing import PathLike
from ..units import UnitSpline
from ..units import ureg as units
from .kinetics import Kinetics

_TAG_TO_SPECIES_NAME: Dict[str, str] = {
    "e": "electron",
    "electron": "electron",
    "H": "hydrogen",
    "hydrogen": "hydrogen",
    "D": "deuterium",
    "deuterium": "deuterium",
    "T": "tritium",
    "tritium": "tritium",
    "He": "helium",
    "helium": "helium",
    "He3": "helium3",
    "helium3": "helium3",
    "He4": "helium",
    "C": "carbon",
    "carbon": "carbon",
    "N": "nitrogen",
    "Ne": "neon",
    "Ar": "argon",
    "W": "tungsten",
    "tungsten": "tungsten",
}

_DEFAULT_CHARGE_MASS_AMU: Dict[str, Tuple[float, float]] = {
    "electron": (-1.0, float((electron_mass / (0.5 * deuterium_mass)).magnitude)),
    "hydrogen": (1.0, 1.0),
    "deuterium": (1.0, 2.0),
    "tritium": (1.0, 3.0),
    "helium3": (2.0, 3.0),
    "helium": (2.0, 4.0),
    "carbon": (6.0, 12.0),
    "nitrogen": (7.0, 14.0),
    "neon": (10.0, 20.0),
    "argon": (18.0, 40.0),
    "tungsten": (74.0, 183.84),
}


class _T3DStub:
    """Lightweight placeholder allowing T3D .log.npy pickles to be read without importing t3d or adios2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_T3DStub":
        return self

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif (
            isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict)
        ):
            self.__dict__.update(state[1])
        else:
            self._state = state

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _T3DStub()


class _T3DUnpickler(pickle.Unpickler):
    """Safe unpickler that stubs out t3d.* and adios2.* classes while preserving NumPy arrays and dicts."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(("t3d", "adios2")):
            return type(name, (_T3DStub,), {"__module__": module})
        return super().find_class(module, name)


def _unwrap_profile(val: Any) -> np.ndarray:
    if hasattr(val, "__dict__") and "profile" in val.__dict__:
        return np.asarray(val.__dict__["profile"], dtype=float)
    return np.asarray(val, dtype=float)


def load_t3d_log_npy(filename: PathLike) -> Dict[str, Any]:
    """Load a T3D ``.log.npy`` dictionary without requiring ``t3d`` or ``adios2`` to be installed."""
    path = Path(filename)
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a NumPy .npy file.")
        f.seek(0)
        version = np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, version)
        obj = _T3DUnpickler(f).load()
    data = obj.item() if isinstance(obj, np.ndarray) else obj
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict inside T3D .log.npy file {path}, got {type(data)!r}."
        )
    return data


def _species_mass_quantity(species_name: str, mass_amu: Optional[float]):
    if species_name == "electron":
        return electron_mass
    if species_name == "hydrogen":
        return hydrogen_mass
    if species_name == "deuterium":
        return deuterium_mass
    if species_name == "tritium":
        return tritium_mass
    if mass_amu is not None and mass_amu > 0.0:
        return (float(mass_amu) / 2.0) * deuterium_mass
    _, default_amu = _DEFAULT_CHARGE_MASS_AMU.get(species_name, (1.0, 2.0))
    return (default_amu / 2.0) * deuterium_mass


def _resolve_time_index(
    t_arr: np.ndarray,
    t_ref: float,
    time_index: int = -1,
    time: Optional[float] = None,
) -> int:
    if time_index != -1 and time is not None:
        raise ValueError("Cannot set both `time` and `time_index`")
    if len(t_arr) == 0:
        raise ValueError("T3D output contains zero time slices.")
    if time is not None:
        t_seconds = t_arr * t_ref
        return int(np.argmin(np.abs(t_seconds - float(time))))
    return int(np.arange(len(t_arr))[time_index])


def _read_t3d_data(
    filename: PathLike,
    time_index: int = -1,
    time: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Return ``(rho_t3d, species_dict)`` with SI magnitudes (``dens`` in m^-3, ``temp`` in eV)."""
    path = Path(filename)
    with open(path, "rb") as f:
        head = f.read(8)

    if head[:6] == b"\x93NUMPY":
        log = load_t3d_log_npy(path)
        if "t" not in log or "grid" not in log:
            raise ValueError(
                f"{path} is a NumPy file, but is missing T3D 't'/'grid' keys."
            )

        grid_obj = log["grid"]
        grid_dict = grid_obj.__dict__ if hasattr(grid_obj, "__dict__") else grid_obj
        rho_t3d = np.asarray(grid_dict["rho"], dtype=float)

        norms = log.get("norms", {})
        if hasattr(norms, "__dict__"):
            norms = norms.__dict__
        t_ref = float(norms.get("t_ref", 1.0)) if isinstance(norms, dict) else 1.0
        t_arr = np.asarray(log["t"], dtype=float)
        idx = _resolve_time_index(t_arr, t_ref, time_index=time_index, time=time)

        # Extract per-species charge & mass metadata if SpeciesDict was logged
        sp_meta_by_tag: Dict[str, Dict[str, Any]] = {}
        if (
            "species" in log
            and isinstance(log["species"], list)
            and len(log["species"]) > 0
        ):
            sp_container = log["species"][idx]
            sp_cdict = (
                sp_container.__dict__ if hasattr(sp_container, "__dict__") else {}
            )
            for sp_type, sp_obj in sp_cdict.get("species_dict", {}).items():
                sd = sp_obj.__dict__ if hasattr(sp_obj, "__dict__") else {}
                tag = str(sd.get("type_tag", sp_type))
                sp_meta_by_tag[tag] = {
                    "species_type": str(sd.get("type", sp_type)),
                    "Z": float(sd["Z"]) if "Z" in sd else None,
                    "mass_amu": float(sd["mass"]) if "mass" in sd else None,
                }

        tags = list(log.get("species_tags", []))
        if not tags:
            tags = [
                k[2:] for k in log.keys() if k.startswith("n_") and f"T_{k[2:]}" in log
            ]
        if not tags:
            raise ValueError(
                f"No T3D species profiles ('n_<tag>', 'T_<tag>') found in {path}."
            )

        species_out: Dict[str, Dict[str, Any]] = {}
        for tag in tags:
            n_key, t_key = f"n_{tag}", f"T_{tag}"
            if n_key not in log or t_key not in log:
                continue
            dens_m3 = _unwrap_profile(log[n_key][idx]) * 1e20
            temp_ev = _unwrap_profile(log[t_key][idx]) * 1e3
            meta = sp_meta_by_tag.get(tag, {})
            raw_name = meta.get("species_type", tag)
            sp_name = _TAG_TO_SPECIES_NAME.get(
                raw_name, _TAG_TO_SPECIES_NAME.get(tag, str(raw_name))
            )
            if sp_name in species_out:
                sp_name = f"{sp_name}_{tag}"
            z_val = meta.get("Z")
            if z_val is None:
                z_val = _DEFAULT_CHARGE_MASS_AMU.get(sp_name, (1.0, 2.0))[0]
            mass_qty = _species_mass_quantity(sp_name, meta.get("mass_amu"))
            species_out[sp_name] = {
                "charge": float(z_val),
                "mass": mass_qty,
                "dens_m3": dens_m3,
                "temp_ev": temp_ev,
            }
        return rho_t3d, species_out

    # Otherwise try NetCDF4 T3D output (.nc)
    with nc.Dataset(path, "r") as ds:
        if "grid" in ds.groups and "rho" in ds.groups["grid"].variables:
            rho_t3d = np.asarray(ds.groups["grid"].variables["rho"][:], dtype=float)
        elif "rho" in ds.variables:
            rho_t3d = np.asarray(ds.variables["rho"][:], dtype=float)
        else:
            raise ValueError(
                f"{path} is not a recognised T3D NetCDF file (missing 'grid/rho')."
            )

        if "time" in ds.groups and "t" in ds.groups["time"].variables:
            t_arr = np.asarray(ds.groups["time"].variables["t"][:], dtype=float)
        elif "t" in ds.variables:
            t_arr = np.asarray(ds.variables["t"][:], dtype=float)
        else:
            raise ValueError(f"{path} is missing T3D time variable 't'.")

        t_ref = 1.0
        if "norms" in ds.groups and "t_ref" in ds.groups["norms"].variables:
            t_ref = float(np.asarray(ds.groups["norms"].variables["t_ref"][:]))
        elif hasattr(ds, "t_ref"):
            t_ref = float(ds.t_ref)

        idx = _resolve_time_index(t_arr, t_ref, time_index=time_index, time=time)
        sp_src = (
            ds.groups["species"].variables if "species" in ds.groups else ds.variables
        )
        tags = [
            k[2:]
            for k in sp_src.keys()
            if k.startswith("n_") and f"T_{k[2:]}" in sp_src
        ]
        if not tags:
            raise ValueError(
                f"No T3D species profiles ('n_<tag>', 'T_<tag>') found in {path}."
            )

        species_out = {}
        for tag in tags:
            n_var = sp_src[f"n_{tag}"]
            t_var = sp_src[f"T_{tag}"]
            dens_m3 = np.asarray(n_var[idx, :], dtype=float) * 1e20
            temp_ev = np.asarray(t_var[idx, :], dtype=float) * 1e3
            sp_name = _TAG_TO_SPECIES_NAME.get(tag, str(tag))
            z_val = float(
                getattr(
                    n_var, "Z", _DEFAULT_CHARGE_MASS_AMU.get(sp_name, (1.0, 2.0))[0]
                )
            )
            m_amu = getattr(n_var, "mass", None)
            mass_qty = _species_mass_quantity(
                sp_name, float(m_amu) if m_amu is not None else None
            )
            species_out[sp_name] = {
                "charge": z_val,
                "mass": mass_qty,
                "dens_m3": dens_m3,
                "temp_ev": temp_ev,
            }
        return rho_t3d, species_out


def _even_rho_spline(
    rho: np.ndarray, values: np.ndarray
) -> InterpolatedUnivariateSpline:
    """Construct an even-symmetric cubic spline in rho = r/a so df/drho(0) = 0 cleanly."""
    order = np.argsort(rho)
    r_pos = rho[order]
    v_pos = values[order]
    if r_pos[0] > 0.0:
        r_sym = np.concatenate((-r_pos[::-1], r_pos))
        v_sym = np.concatenate((v_pos[::-1], v_pos))
    else:
        r_sym = np.concatenate((-r_pos[:0:-1], r_pos))
        v_sym = np.concatenate((v_pos[:0:-1], v_pos))
    k = min(3, max(1, len(r_sym) - 1))
    return InterpolatedUnivariateSpline(r_sym, v_sym, k=k)


class KineticsReaderT3D(FileReader, file_type="T3D", reads=Kinetics):
    """Read radial kinetic profiles from Trinity3D (T3D) ``.log.npy`` or NetCDF output files."""

    def read_from_file(
        self,
        filename: PathLike,
        eq: Optional[Equilibrium] = None,
        time_index: int = -1,
        time: Optional[float] = None,
    ) -> Kinetics:
        if eq is None:
            raise ValueError(dedent(f"""\
                    {self.__class__.__name__} must be provided with an Equilibrium object via
                    the keyword argument 'eq'. Please load an Equilibrium.
                    """))

        rho_t3d, species_raw = _read_t3d_data(
            filename, time_index=time_index, time=time
        )

        psi_raw = eq["psi_n"].data
        rmin_raw = eq["r_minor"].data
        psi_eq = np.asarray(
            psi_raw.magnitude if hasattr(psi_raw, "magnitude") else psi_raw, dtype=float
        )
        rmin_eq = np.asarray(
            rmin_raw.magnitude if hasattr(rmin_raw, "magnitude") else rmin_raw,
            dtype=float,
        )
        rho_eq = rmin_eq / rmin_eq[-1]

        # Build monotonic rho -> psi_n map so T3D's exact radial grid nodes are included in the spline knots
        _, uniq_r = np.unique(rho_eq, return_index=True)
        uniq_r = np.sort(uniq_r)
        psi_of_rho = InterpolatedUnivariateSpline(
            rho_eq[uniq_r], psi_eq[uniq_r], k=min(3, len(uniq_r) - 1)
        )
        psi_t3d = np.clip(psi_of_rho(rho_t3d), 0.0, 1.0)

        # Merge equilibrium psi_n knots with T3D's exact psi_n knots (dropping near-duplicates < 1e-8)
        psi_merged = np.sort(np.concatenate((psi_eq, psi_t3d)))
        keep = np.concatenate(([True], np.diff(psi_merged) > 1e-8))
        psi_grid = psi_merged[keep]

        _, uniq_p = np.unique(psi_eq, return_index=True)
        uniq_p = np.sort(uniq_p)
        rho_of_psi = InterpolatedUnivariateSpline(
            psi_eq[uniq_p], rho_eq[uniq_p], k=min(3, len(uniq_p) - 1)
        )
        rho_grid = np.clip(rho_of_psi(psi_grid), 0.0, None)

        psi_q = psi_grid * units.dimensionless
        rho_func = UnitSpline(psi_q, rho_grid * units.lref_minor_radius)
        omega_func = UnitSpline(
            psi_q, np.zeros_like(psi_grid) * units.radians / units.second
        )
        ones = np.ones_like(psi_grid)

        result: Dict[str, Species] = {}
        for sp_name, sp_info in species_raw.items():
            dens_spl = _even_rho_spline(rho_t3d, sp_info["dens_m3"])
            temp_spl = _even_rho_spline(rho_t3d, sp_info["temp_ev"])
            dens_vals = np.maximum(dens_spl(rho_grid), 1e10) * units.meter**-3
            temp_vals = np.maximum(temp_spl(rho_grid), 1e-2) * units.eV
            charge_func = UnitSpline(
                psi_q, sp_info["charge"] * ones * units.elementary_charge
            )
            result[sp_name] = Species(
                species_type=sp_name,
                charge=charge_func,
                mass=sp_info["mass"],
                dens=UnitSpline(psi_q, dens_vals),
                temp=UnitSpline(psi_q, temp_vals),
                omega0=omega_func,
                rho=rho_func,
            )

        return Kinetics(kinetics_type="T3D", **result)

    def verify_file_type(self, filename: PathLike) -> None:
        path = Path(filename)
        with open(path, "rb") as f:
            head = f.read(8)

        if head[:6] == b"\x93NUMPY":
            log = load_t3d_log_npy(path)
            has_species = bool(log.get("species_tags")) or any(
                k.startswith("n_") for k in log.keys()
            )
            if "t" not in log or "grid" not in log or not has_species:
                raise ValueError(f"{path} is not a valid T3D .log.npy file.")
            return

        if head[:3] == b"CDF" or head == b"\x89HDF\r\n\x1a\n":
            try:
                with nc.Dataset(path, "r") as ds:
                    is_t3d_nc = (
                        "grid" in ds.groups
                        and "species" in ds.groups
                        and "rho" in ds.groups["grid"].variables
                    ) or hasattr(ds, "trinity_infile")
            except Exception as exc:
                raise ValueError(
                    f"{path} could not be read as a T3D NetCDF file."
                ) from exc
            if not is_t3d_nc:
                raise ValueError(f"{path} is not a T3D NetCDF output file.")
            return

        raise ValueError(f"{path} is not a T3D .log.npy or NetCDF output file.")
