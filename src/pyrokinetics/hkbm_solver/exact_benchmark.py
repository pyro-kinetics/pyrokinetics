"""Resumable, GENE-seeded benchmarks of the collisionless full GK solver.

Accepts the STEP-EM-TRANSITION Figure 16 CSV directly. This is an eigenvalue
benchmark, not an exhaustive search for all unstable roots. Non-convergence is
recorded as unresolved, never as stability. --collisionless is required because
the current full solver does not implement the nominal GENE collision operator.

Example (run through the site's scheduler, with BLAS threads set to one)::

    python -m pyrokinetics.hkbm_solver.exact_benchmark --collisionless \
        --reference figure16a/data.csv --template parameters \
        --output fig16_collisionless.jsonl --indices 360,396 --workers 2

Output records contain input/source hashes, the original collision parameter,
resolution, parity, convergence residual, endpoint amplitude and wall time.
An existing output can only be resumed with exactly the same inputs and code.
"""

import argparse
import csv
import hashlib
import json
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import f90nml
import numpy as np

from .exact import ExactSolver
from .gene_io import Deck


def read_reference(path):
    """Read the paper CSV without rounding, reordering or filling missing cells."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = ("beta", "ky", "growth_rate", "mode_frequency")
    result = []
    for index, row in enumerate(rows):
        values = {key: float(row[key]) for key in required}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"reference row {index} contains a non-finite value")
        if values["ky"] <= 0 or values["beta"] < 0:
            raise ValueError(f"reference row {index} has invalid beta or ky")
        result.append(
            dict(
                index=index,
                **values,
                Ctear=float(row.get("Ctear") or 0.0),
                growth_rate_tolerance=(
                    float(row["growth_rate_tolerance"])
                    if row.get("growth_rate_tolerance")
                    else None
                ),
            )
        )
    if not result:
        raise ValueError("reference grid is empty")
    keys = [(row["beta"], row["ky"]) for row in result]
    if len(set(keys)) != len(keys):
        raise ValueError("reference has duplicate beta-ky cells")
    return result


def configuration(
    template, reference, resolution, maxit, timeout, collision_model="none"
):
    """Hash numerical settings and all solver sources, including local modifications."""
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    config = dict(
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        source_sha256=sources,
        resolution=resolution,
        maxit=maxit,
        timeout=timeout,
        collision_model=collision_model,
        seed_method="GENE eigenvalue",
    )
    config["id"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return config


def completed_records(path, config_id):
    """Reject incompatible resumes before launching any work; do not silently mix runs."""
    done = set()
    if Path(path).exists():
        with Path(path).open() as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"incomplete record on line {number}; preserve and repair it first"
                    ) from exc
                if row["configuration"]["id"] != config_id:
                    raise ValueError(
                        "output belongs to different inputs, resolution or source; choose a new output"
                    )
                done.add((row["reference"]["index"], row["parity"]))
    return done


def solve_case(reference, parity, template, config):
    started = time.monotonic()
    nml = f90nml.reads(template)
    nominal_coll = float(nml["general"].get("coll", 0.0))
    collision_model = config.get("collision_model", "none")
    if collision_model not in ("none", "legacy"):
        raise ValueError("unsupported collision_model")
    if collision_model == "legacy" and parity != "twisting":
        raise ValueError("legacy collisions require twisting parity")
    nml["general"]["beta"] = reference["beta"]
    if collision_model == "none":
        nml["general"]["coll"] = 0.0
    nml["box"]["kymin"] = reference["ky"]
    record = dict(
        reference=reference,
        parity=parity,
        configuration=config,
        reference_coll=nominal_coll,
        collision_physics_match=nominal_coll == 0,
        collision_model=collision_model,
        status="unresolved",
        converged=False,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="exact-benchmark-") as tmp:
            deck_path = Path(tmp) / "parameters"
            nml.write(deck_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                deck = Deck(deck_path)
                solver = ExactSolver.from_deck(
                    deck,
                    reference["ky"],
                    collision_model=collision_model,
                    **config["resolution"],
                )
            record["input_warnings"] = [str(w.message) for w in caught]
            cs = deck.units["c_s_over_c_ref"]
            seed = (
                complex(
                    -reference["mode_frequency"], max(reference["growth_rate"], 0.005)
                )
                / cs
            )
            result = solver.find_root(
                seed, parity, maxit=config["maxit"], timeout=config["timeout"]
            )
            record.update(
                status=result["status"],
                converged=result["converged"],
                omega=result["omega_gene"] * cs,
                gamma=result["gamma"] * cs,
                relative_residual=result["relative_residual"],
                edge_phi=result.get("edge_phi"),
                Ctear=result.get("C_tear"),
                kperp2_phi=result.get("kperp2_phi"),
                kperp2_apar=result.get("kperp2_apar"),
                iterations=result["iters"],
                setup_seconds=solver.t_setup,
                root_seconds=result["seconds"],
                collision_parameter=result["collision_parameter"],
            )
            for key in (
                "omega",
                "gamma",
                "relative_residual",
                "edge_phi",
                "Ctear",
                "kperp2_phi",
                "kperp2_apar",
            ):
                if record[key] is not None and not np.isfinite(record[key]):
                    record[key] = None
                    record.update(
                        status="error",
                        converged=False,
                        error="non-finite eigenvalue or eigenfunction diagnostics",
                    )
    except TimeoutError as exc:
        record.update(status="timeout", error=str(exc))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record["total_seconds"] = time.monotonic() - started
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collisionless", action="store_true")
    parser.add_argument("--collision-model", choices=("none", "legacy"), default="none")
    parser.add_argument(
        "--indices", help="comma-separated zero-based CSV row indices (default all)"
    )
    parser.add_argument(
        "--parity",
        choices=("reference", "twisting", "tearing", "both"),
        default="reference",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maxit", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=120.0)
    for name, default in (
        ("npt", 32),
        ("nturns", 1),
        ("nE", 12),
        ("nlp", 12),
        ("nlt", 4),
        ("nq", 6),
        ("nbs", 4),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    if not args.collisionless and args.collision_model == "none":
        parser.error(
            "the current exact solver is collisionless; pass --collisionless explicitly"
        )
    if args.collision_model == "legacy" and (
        args.collisionless or args.parity != "twisting"
    ):
        parser.error(
            "--collision-model legacy requires --parity twisting and no --collisionless"
        )
    if args.workers < 1 or args.maxit < 1 or args.timeout <= 0:
        parser.error("workers, maxit and timeout must be positive")
    rows = read_reference(args.reference)
    indices = (
        set(map(int, args.indices.split(",")))
        if args.indices
        else set(range(len(rows)))
    )
    if not indices.issubset(range(len(rows))):
        parser.error("indices must select existing CSV rows")
    template = args.template.read_text()
    resolution = {
        key: getattr(args, key)
        for key in ("npt", "nturns", "nE", "nlp", "nlt", "nq", "nbs")
    }
    config = configuration(
        template,
        args.reference,
        resolution,
        args.maxit,
        args.timeout,
        args.collision_model,
    )
    done = completed_records(args.output, config["id"])
    jobs = []
    for ref in rows:
        if ref["index"] not in indices:
            continue
        parity = "tearing" if ref["Ctear"] > 0.5 else "twisting"
        parities = (
            ("twisting", "tearing")
            if args.parity == "both"
            else (parity if args.parity == "reference" else args.parity,)
        )
        jobs.extend((ref, par) for par in parities if (ref["index"], par) not in done)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(rows)} reference cells; {len(jobs)} pending solves; collision model={args.collision_model}",
        flush=True,
    )
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        args.output.open("a") as handle,
    ):
        futures = [
            pool.submit(solve_case, ref, par, template, config) for ref, par in jobs
        ]
        for future in as_completed(futures):
            record = future.result()
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            ref = record["reference"]
            print(
                f"row {ref['index']}: beta={ref['beta']} ky={ref['ky']} {record['parity']} "
                f"{record['status']} omega={record.get('omega')} gamma={record.get('gamma')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
