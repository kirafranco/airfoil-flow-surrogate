"""AirfRANS VTK → parquet ETL with a per-simulation quality gate.

    python -m src.etl.pipeline               # every simulation of the raw snapshot
    python -m src.etl.pipeline --workers 1   # in one process
    python -m src.etl.pipeline --limit 10    # smoke run: 10 simulations, no dataset-level outputs

In the container:

    docker compose --profile ml run --rm ml python -m src.etl.pipeline

Reads the one complete raw snapshot under data/raw/airfrans/ and writes:

    data/interim/airfrans/<snapshot-date>/
        etl.json                 run record: schema version, raw SHA-256, counts, failures, timings
        simulations.parquet      one row per simulation
        simulations/<name>/      nodes, edges and surface tables (see src.etl.schema)
    data/processed/airfrans/<snapshot-date>/
        splits.json              train / val / test by simulation
        normalisation.json       statistics over the training simulations' nodes only

Each simulation is read, checked by the quality gate and written on its own. One
that fails is logged with the check it failed and skipped, and the batch goes on.
Finished simulations are checkpoints, so a re-run only does what is missing.
The dataset-level outputs are written only from a complete set, never from part
of one, and a re-run with nothing left to do writes nothing at all.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import resource
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AIRFRANS_RAW_ROOT, INTERIM_ROOT, PROCESSED_ROOT
from src.etl.names import CHORD_M, NU_M2_S, SimulationNameError, parse_simulation_name
from src.etl.normalise import fit_normalisation
from src.etl.quality import check_simulation
from src.etl.schema import (
    EDGES_FILE,
    MANIFEST_SPLITS,
    META_FILE,
    NODES_FILE,
    PARTIAL_SUFFIX,
    SCHEMA_VERSION,
    SIMULATIONS_SCHEMA,
    SURFACE_FILE,
    build_table,
    edges_table,
    is_complete,
    nodes_table,
    surface_table,
    write_json,
    write_parquet_atomic,
    write_simulation,
)
from src.etl.splits import make_splits, names_sha256, splits_record
from src.etl.vtk_io import VtkReadError, read_simulation
from src.ingest.airfrans import git_commit

logger = logging.getLogger(__name__)

SOURCE_FILE = "source.json"
DATASET_DIR = "Dataset"
MANIFEST_FILE = "manifest.json"
ETL_FILE = "etl.json"
SIMULATIONS_FILE = "simulations.parquet"
SIMULATIONS_DIR = "simulations"
SPLITS_FILE = "splits.json"
NORMALISATION_FILE = "normalisation.json"
DEFAULT_WORKERS = 2  # the ml container's CPU limit; decoding VTK is CPU-bound
PROGRESS_STEP_PCT = 5


class EtlError(Exception):
    """A problem with the whole run, not one simulation: the run stops."""


@dataclass(frozen=True)
class SimOutcome:
    name: str
    ok: bool
    #: (check, detail) pairs; "name", "read" and "error" for failures before or outside the gate.
    violations: tuple[tuple[str, str], ...] = ()
    bytes_written: int = 0
    read_s: float = 0.0
    total_s: float = 0.0


@dataclass(frozen=True)
class EtlResult:
    interim_dir: Path
    processed_dir: Path
    expected: int
    processed: int
    skipped: int
    failures: tuple[SimOutcome, ...]
    #: True when the dataset-level outputs were (re)written in this run.
    wrote_dataset: bool
    seconds: float
    outcomes: tuple[SimOutcome, ...] = field(default=(), repr=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_snapshot(raw_root: Path) -> tuple[Path, dict[str, Any]]:
    """The single complete raw snapshot: a directory with source.json and an extracted Dataset/."""
    candidates = (
        sorted(d for d in raw_root.iterdir() if (d / SOURCE_FILE).is_file() and (d / DATASET_DIR).is_dir())
        if raw_root.is_dir()
        else []
    )
    if len(candidates) != 1:
        found = ", ".join(d.name for d in candidates) or "none"
        raise EtlError(f"expected exactly one complete raw snapshot under {raw_root}, found {found}")
    source = _read_json(candidates[0] / SOURCE_FILE)
    if not source.get("sha256"):
        raise EtlError(f"{candidates[0] / SOURCE_FILE} has no sha256: the raw snapshot is incomplete")
    return candidates[0], source


def process_simulation(sim_dir: Path, sims_root: Path) -> SimOutcome:
    """Read, gate and write one simulation. Never raises: every failure becomes an outcome."""
    started = time.perf_counter()
    name = sim_dir.name

    def failed(check: str, detail: str, read_s: float = 0.0) -> SimOutcome:
        return SimOutcome(
            name, False, ((check, detail),), read_s=read_s, total_s=time.perf_counter() - started
        )

    try:
        try:
            params = parse_simulation_name(name)
        except SimulationNameError as exc:
            return failed("name", str(exc))
        try:
            raw = read_simulation(sim_dir)
        except VtkReadError as exc:
            return failed("read", str(exc))
        read_s = time.perf_counter() - started

        gate = check_simulation(raw, params)
        if not gate.passed or gate.derived is None:
            return SimOutcome(
                name,
                False,
                tuple((v.check, v.detail) for v in gate.violations),
                read_s=read_s,
                total_s=time.perf_counter() - started,
            )
        derived = gate.derived
        meta = {
            "name": name,
            "u_inf": params.u_inf,
            "aoa_deg": params.aoa_deg,
            "inlet_u_x": params.inlet_u_x,
            "inlet_u_y": params.inlet_u_y,
            "reynolds": params.reynolds,
            "naca_digits": params.naca_digits,
            "naca_params": list(params.naca_params),
            "n_nodes": raw.mesh.n_points,
            "n_edges": int(len(derived.edges)),
            "n_surface": int(derived.is_surface.sum()),
            "stagnation_ratio": derived.stagnation_ratio,
            "inlet_speed_rel_err": derived.inlet_speed_rel_err,
            "inlet_angle_err_deg": derived.inlet_angle_err_deg,
            "read_s": read_s,
        }
        tables = {
            NODES_FILE: nodes_table(raw, derived),
            EDGES_FILE: edges_table(derived),
            SURFACE_FILE: surface_table(derived),
        }
        written = write_simulation(sims_root, name, tables, meta, started)
        return SimOutcome(name, True, (), written, read_s, time.perf_counter() - started)
    except Exception as exc:  # one simulation's surprise must not stop the other 999
        return failed("error", f"{type(exc).__name__}: {exc}")


def _process_all(sim_dirs: list[Path], sims_root: Path, workers: int) -> list[SimOutcome]:
    total = len(sim_dirs)
    outcomes: list[SimOutcome] = []
    started = time.perf_counter()
    next_pct = PROGRESS_STEP_PCT

    def record(outcome: SimOutcome) -> None:
        nonlocal next_pct
        outcomes.append(outcome)
        if outcome.ok:
            logger.info(
                "sim_done name=%s read_s=%.2f total_s=%.2f bytes=%d",
                outcome.name,
                outcome.read_s,
                outcome.total_s,
                outcome.bytes_written,
            )
        else:
            for check, detail in outcome.violations:
                logger.warning("simulation_failed name=%s check=%s detail=%s", outcome.name, check, detail)
        pct = len(outcomes) * 100 // total
        if pct >= next_pct:
            elapsed = time.perf_counter() - started
            logger.info(
                "etl_progress done=%d failed=%d of=%d elapsed_s=%.0f eta_s=%.0f",
                len(outcomes),
                sum(not o.ok for o in outcomes),
                total,
                elapsed,
                elapsed / len(outcomes) * (total - len(outcomes)),
            )
            next_pct = (pct // PROGRESS_STEP_PCT + 1) * PROGRESS_STEP_PCT

    if workers <= 1 or total <= 1:
        for sim_dir in sim_dirs:
            record(process_simulation(sim_dir, sims_root))
    else:
        # Workers start from a clean server process, not a fork of this one:
        # pyarrow and VTK run threads, and forking a threaded process can deadlock.
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(process_simulation, sim_dir, sims_root) for sim_dir in sim_dirs]
            for future in as_completed(futures):
                record(future.result())
    return outcomes


def _check_record(interim_dir: Path, raw_sha256: str) -> dict[str, Any] | None:
    """Refuse to mix layouts: an existing interim directory must match this schema and raw snapshot."""
    path = interim_dir / ETL_FILE
    if not path.is_file():
        return None
    record = _read_json(path)
    if record.get("schema_version") != SCHEMA_VERSION or record.get("raw_sha256") != raw_sha256:
        raise EtlError(
            f"{interim_dir} was built with schema {record.get('schema_version')!r} from raw SHA-256 "
            f"{record.get('raw_sha256')!r}; this run is {SCHEMA_VERSION!r} from {raw_sha256!r}. "
            "Delete that directory to rebuild it rather than mixing two layouts."
        )
    return record


def _simulations_table(sims_root: Path, names: list[str], manifest: dict[str, list[str]]) -> tuple[Any, list[dict]]:
    rows = [_read_json(sims_root / name / META_FILE) for name in names]
    membership = {split: set(manifest.get(split, [])) for split in MANIFEST_SPLITS}
    columns: dict[str, Any] = {key: [row[key] for row in rows] for key in rows[0]}
    for split in MANIFEST_SPLITS:
        columns[split] = [row["name"] in membership[split] for row in rows]
    table = build_table(columns, SIMULATIONS_SCHEMA)

    # Dataset-level checks: unique names, every manifest name present, split sizes as listed.
    if len(set(columns["name"])) != len(rows):
        raise EtlError("simulation names are not unique")
    for split in MANIFEST_SPLITS:
        listed = membership[split] & set(names)
        if sum(columns[split]) != len(membership[split]) or len(listed) != len(membership[split]):
            raise EtlError(f"split {split}: {sum(columns[split])} rows vs {len(membership[split])} in the manifest")
    return table, rows


def _peak_rss_mb() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    workers = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(own, workers) / 1024  # ru_maxrss is in KiB on Linux


def _timing(outcomes: Iterable[SimOutcome], wall_s: float, workers: int) -> dict[str, Any]:
    totals = np.array([o.total_s for o in outcomes if o.ok])
    return {
        "wall_s": round(wall_s, 1),
        "workers": workers,
        "s_per_sim_median": round(float(np.median(totals)), 3) if totals.size else None,
        "s_per_sim_p95": round(float(np.percentile(totals, 95)), 3) if totals.size else None,
        "sims_per_min": round(totals.size / wall_s * 60, 1) if wall_s > 0 else None,
    }


def run(
    *,
    raw_root: Path = AIRFRANS_RAW_ROOT,
    interim_root: Path = INTERIM_ROOT,
    processed_root: Path = PROCESSED_ROOT,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> EtlResult:
    started = time.perf_counter()
    started_at = now()
    snapshot, source = find_snapshot(raw_root)
    raw_sha256 = source["sha256"]
    dataset = snapshot / DATASET_DIR
    manifest = _read_json(dataset / MANIFEST_FILE)
    expected = sorted(set(manifest["full_train"]) | set(manifest["full_test"]))

    interim_dir = interim_root / "airfrans" / snapshot.name
    processed_dir = processed_root / "airfrans" / snapshot.name
    sims_root = interim_dir / SIMULATIONS_DIR
    record = _check_record(interim_dir, raw_sha256)
    sims_root.mkdir(parents=True, exist_ok=True)
    if record is None:
        # Written before any simulation, so an interrupted first run still
        # carries the schema version a later run checks against.
        write_json(
            interim_dir / ETL_FILE,
            {"schema_version": SCHEMA_VERSION, "raw_sha256": raw_sha256, "status": "running"},
        )
    for leftover in sorted(sims_root.glob(f"*{PARTIAL_SUFFIX}")):
        logger.info("partial_removed dir=%s", leftover.name)
        shutil.rmtree(leftover)

    done = {name for name in expected if is_complete(sims_root / name)}
    todo = [name for name in expected if name not in done]
    if limit is not None:
        todo = todo[:limit]
    logger.info(
        "etl_start snapshot=%s raw_sha256=%s expected=%d already_done=%d to_process=%d workers=%d",
        snapshot.name,
        raw_sha256,
        len(expected),
        len(done),
        len(todo),
        workers,
    )

    outcomes = _process_all([dataset / name for name in todo], sims_root, workers)
    failures = tuple(o for o in outcomes if not o.ok)
    processed = len(outcomes) - len(failures)
    bytes_written = sum(o.bytes_written for o in outcomes)
    wall_s = time.perf_counter() - started

    def result(wrote_dataset: bool) -> EtlResult:
        seconds = time.perf_counter() - started
        logger.info(
            "etl_done processed=%d skipped=%d failed=%d seconds=%.1f bytes_written=%d peak_rss_mb=%.0f wrote_dataset=%s",
            processed,
            len(done),
            len(failures),
            seconds,
            bytes_written,
            _peak_rss_mb(),
            wrote_dataset,
        )
        return EtlResult(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            expected=len(expected),
            processed=processed,
            skipped=len(done),
            failures=failures,
            wrote_dataset=wrote_dataset,
            seconds=seconds,
            outcomes=tuple(outcomes),
        )

    run_record = {
        "schema_version": SCHEMA_VERSION,
        "raw_snapshot": snapshot.name,
        "raw_sha256": raw_sha256,
        "etl_code_commit": git_commit(),
        "started_at_utc": started_at.isoformat(timespec="seconds"),
        "simulations": {
            "expected": len(expected),
            "processed": processed,
            "skipped": len(done),
            "failed": len(failures),
        },
        "failures": [
            {"name": o.name, "check": check, "detail": detail} for o in failures for check, detail in o.violations
        ],
        "timing": _timing(outcomes, wall_s, workers),
        "bytes_written": bytes_written,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "constants": {"chord_m": CHORD_M, "nu_m2_s": NU_M2_S},
    }

    if limit is not None:
        logger.info("etl_smoke_run limit=%d: per-simulation outputs only", limit)
        return result(wrote_dataset=False)

    if failures:
        write_json(
            interim_dir / ETL_FILE,
            {**run_record, "status": "incomplete", "finished_at_utc": now().isoformat(timespec="seconds")},
        )
        logger.error(
            "etl_incomplete failed=%d: no dataset-level outputs written; fix or settle the failures and re-run",
            len(failures),
        )
        return result(wrote_dataset=False)

    outputs = (
        interim_dir / ETL_FILE,
        interim_dir / SIMULATIONS_FILE,
        processed_dir / SPLITS_FILE,
        processed_dir / NORMALISATION_FILE,
    )
    previous = _read_json(interim_dir / ETL_FILE)
    if processed == 0 and previous.get("status") == "complete" and all(path.is_file() for path in outputs):
        logger.info("etl_up_to_date interim=%s: nothing written", interim_dir)
        return result(wrote_dataset=False)

    table, rows = _simulations_table(sims_root, expected, manifest)
    write_parquet_atomic(table, interim_dir / SIMULATIONS_FILE)

    splits = make_splits(manifest)
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_json(processed_dir / SPLITS_FILE, splits_record(splits, raw_sha256))

    inlet = {row["name"]: (row["inlet_u_x"], row["inlet_u_y"]) for row in rows}
    norm_started = time.perf_counter()
    normalisation = fit_normalisation(
        (name, sims_root / name, *inlet[name]) for name in splits.train
    )
    normalisation["fitted_on"].update(
        {"names_sha256": names_sha256(splits.train), "raw_sha256": raw_sha256, "schema_version": SCHEMA_VERSION}
    )
    write_json(processed_dir / NORMALISATION_FILE, normalisation)
    logger.info(
        "normalisation_done simulations=%d nodes=%d seconds=%.1f",
        len(splits.train),
        normalisation["channels"]["x"]["count"],
        time.perf_counter() - norm_started,
    )

    write_json(
        interim_dir / ETL_FILE,
        {**run_record, "status": "complete", "finished_at_utc": now().isoformat(timespec="seconds")},
    )
    return result(wrote_dataset=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="worker processes (default: %(default)s)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N simulations and write no dataset-level outputs (a smoke run)",
    )
    args = parser.parse_args(argv)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    try:
        result = run(workers=args.workers, limit=args.limit)
    except EtlError as exc:
        logger.error("etl_stopped error=%s", exc)
        return 1
    logger.info(
        "etl_result interim=%s processed=%d skipped=%d failed=%d",
        result.interim_dir,
        result.processed,
        result.skipped,
        len(result.failures),
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
